"""Bounded, ordered per-process PNG loading; no CUDA work in reader threads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path

from .core import uniform_indices


def configure_cpu(cores=8, slot=0):
    """Pin this new process to one logical CPU per physical core, before ML imports.

    Slots select disjoint groups within the inherited affinity. Affinity limits
    placement; it does not reserve these CPUs against other users' processes.
    """
    if cores < 1 or slot < 0:
        raise ValueError('CPU core count must be positive and slot nonnegative')
    available = sorted(os.sched_getaffinity(0))
    physical = {}
    for cpu in available:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        package = (topology/'physical_package_id').read_text().strip()
        core = (topology/'core_id').read_text().strip()
        physical.setdefault((package, core), cpu)
    cpus = list(physical.values())[slot*cores:(slot+1)*cores]
    if len(cpus) != cores:
        raise ValueError(f'CPU slot {slot} needs {cores} physical cores; only {len(physical)} available')
    os.sched_setaffinity(0, cpus)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(cores)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    return cpus


@dataclass
class LoadedRow:
    row: dict
    images: list
    references: list
    indices: list


class ImagePrefetcher:
    def __init__(self, root, *, adapter, workers=8, prefetch_batches=2, official_frames=32):
        if workers < 1 or prefetch_batches < 0:
            raise ValueError('Reader workers must be positive and prefetch batches nonnegative')
        if adapter not in ('icl', 'official') or official_frames not in (8, 16, 32):
            raise ValueError('Unsupported adapter or official frame count')
        self.root = Path(root)
        self.adapter, self.official_frames = adapter, official_frames
        self.prefetch_batches = prefetch_batches
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='png-reader')
        self.references = {}

    def _read(self, path):
        from PIL import Image
        with Image.open(self.root/path) as image:
            return image.convert('RGB')

    def _submit(self, rows):
        batch = []
        for row in rows:
            indices = (list(range(row['num_frames'])) if self.adapter == 'icl' else
                       uniform_indices(row['num_frames'], self.official_frames))
            references = []
            if self.adapter == 'icl':
                for path in row['icl_reference']['image_paths']:
                    if path not in self.references:
                        self.references[path] = self.pool.submit(self._read, path)
                    references.append(self.references[path])
            images = [self.pool.submit(self._read, row['image_paths'][i]) for i in indices]
            batch.append((row, images, references, indices))
        return batch

    def batches(self, rows, batch_size):
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        pending = deque()
        cursor = 0
        while cursor < len(rows) or pending:
            while cursor < len(rows) and len(pending) < self.prefetch_batches + 1:
                pending.append(self._submit(rows[cursor:cursor+batch_size]))
                cursor += batch_size
            batch = pending.popleft()
            # Futures preserve manifest ordering even when disk reads finish out of order.
            yield [LoadedRow(row, [f.result() for f in images], [f.result() for f in refs], indices)
                   for row, images, refs, indices in batch]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.references.clear()
