"""Metadata-only success filtering and deterministic, prebatched ICL sampling."""
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from robometer.data.dataset_category import DATASET_MAP
from .targets import collect_batch, select_frames


def validation_member(dataset_key, trajectory_id, fraction, seed):
    key = f'{seed}:{dataset_key}:{trajectory_id}'.encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') / 2**64
    return value < fraction


class TrajectoryStore:
    def __init__(self, cfg):
        self.root = Path(cfg.root)
        names = []
        for key in cfg.datasets:
            names.extend(DATASET_MAP[key]['train'] if key in DATASET_MAP else [key])
        names = list(dict.fromkeys(names))
        excluded = set(cfg.exclude_datasets)
        unknown = excluded - set(names)
        if unknown:
            raise ValueError(f'Excluded dataset keys are not in the requested pool: {sorted(unknown)}')
        self.datasets = {}
        self.indices = {'train': {}, 'validation': {}}
        self.summary = []
        for name in names:
            if name in excluded:
                continue
            path = self.root / name / 'processed_dataset'
            ds = Dataset.load_from_disk(str(path))
            columns = [k for k in ('id', 'task', 'data_source', 'quality_label', 'partial_success',
                                   'num_frames', 'frames_shape', 'frames') if k in ds.column_names]
            ds = ds.select_columns(columns)
            train, validation = [], []
            counts = dict(total=len(ds), non_success=0, short=0, excluded_source=0, partial_conflict=0)
            row_offset = 0
            for batch in ds.iter(batch_size=8192):
                n = len(batch['quality_label'])
                partial_scores = batch.get('partial_success')
                identifiers = batch.get('id')
                for j in range(n):
                    if batch['quality_label'][j] != 'successful':
                        counts['non_success'] += 1
                        continue
                    # A conflicting partial score must not reintroduce partial failures.
                    partial = partial_scores[j] if partial_scores is not None else None
                    if partial is not None and not np.isclose(partial, 1.0):
                        counts['partial_conflict'] += 1
                        continue
                    source = batch['data_source'][j]
                    if source in cfg.exclude_sources:
                        counts['excluded_source'] += 1
                        continue
                    frame_count = (batch['frames_shape'][j][0] if 'frames_shape' in batch
                                   else batch['num_frames'][j])
                    if frame_count < cfg.min_keyframes + 1:
                        counts['short'] += 1
                        continue
                    row_id = row_offset + j
                    identifier = (identifiers[j] if identifiers is not None else None) or str(row_id)
                    dest = validation if validation_member(name, identifier, cfg.validation_fraction,
                                                           cfg.split_seed) else train
                    dest.append(row_id)
                row_offset += n
            counts.update(dataset=name, train_eligible=len(train), validation_eligible=len(validation))
            cap = cfg.max_trajectories_per_source
            if cap is not None:
                train, validation = train[:int(cap)], validation[:int(cap)]
            counts.update(train_selected=len(train), validation_selected=len(validation))
            self.summary.append(counts)
            for split, indices in [('train', train), ('validation', validation)]:
                if indices:
                    self.indices[split][name] = np.asarray(indices, dtype=np.int64)
            if train or validation:
                self.datasets[name] = ds
        if not self.indices['train']:
            raise ValueError('No eligible successful training trajectories remain')

    def save_summary(self, output):
        Path(output).write_text(json.dumps(self.summary, ensure_ascii=False, indent=2) + '\n')


class ICLBatchDataset(TorchDataset):
    """One item is one GPU's complete B-sample batch, before image decoding.

    A sequential outer DataLoader uses batch_size=1. Accelerate shards these
    complete batches across ranks (split_batches=False), preserving refill size.
    """
    def __init__(self, store, cfg, batch_size, length, seed, split='train', offset=0):
        self.store, self.cfg = store, cfg
        self.batch_size, self.length, self.seed = batch_size, length, seed
        self.split, self.offset = split, offset
        self.sources = sorted(store.indices[split])
        if not self.sources:
            raise ValueError(f'No {split} trajectories; use more data or a larger validation_fraction')
        self.weights = None
        if not cfg.source_balanced:
            sizes = np.asarray([len(store.indices[split][s]) for s in self.sources], dtype=np.float64)
            self.weights = sizes / sizes.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.offset + index]))

        def draw():
            source = self.sources[int(rng.choice(len(self.sources), p=self.weights))]
            pool = self.store.indices[self.split][source]
            row = self.store.datasets[source][int(pool[int(rng.integers(len(pool)))])]
            count = row['frames_shape'][0] if 'frames_shape' in row else row['num_frames']
            selected = select_frames(count, self.cfg.min_keyframes, self.cfg.max_keyframes, rng)
            if selected is None:
                return None
            return {'trajectory': row, 'selection': selected, 'dataset_key': source}

        samples, attempts = collect_batch(draw, self.batch_size, self.cfg.max_candidate_rounds)
        return {'samples': samples, 'candidate_attempts': attempts}
