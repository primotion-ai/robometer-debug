"""Ordering, boundedness, failures and serial/parallel inference equivalence."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from .adapters import ICLAdapter, OfficialAdapter
from .loaded_inference import predict_icl, predict_official
from .prefetch import ImagePrefetcher


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for j in range(4):
            paths = []
            for i in range(32):
                path = f'{j}_{i}.png'
                Image.fromarray(np.full((28,28,3), j*40+i, np.uint8)).save(self.root/path)
                paths.append(path)
            self.rows.append(dict(trajectory_id=str(j), instruction='task', num_frames=32,
                                  image_paths=paths))
        for row in self.rows:
            row['icl_reference'] = dict(trajectory_id='0', image_paths=[self.rows[0]['image_paths'][i] for i in (0,8,16,24,31)])

    def test_order_reference_reuse_and_decoded_pixels(self):
        with ImagePrefetcher(self.root, adapter='icl', workers=8, prefetch_batches=2) as loader:
            batches = list(loader.batches(self.rows, 1))
            self.assertEqual(len(loader.references), 5)
        for j, batch in enumerate(batches):
            loaded = batch[0]
            self.assertEqual(loaded.row['trajectory_id'], str(j))
            self.assertEqual([x.getpixel((0,0))[0] for x in loaded.images], list(range(j*40,j*40+32)))
            self.assertEqual([x.getpixel((0,0))[0] for x in loaded.references], [0,8,16,24,31])
            self.assertIs(loaded.references[0], batches[0][0].references[0])

    def test_prefetch_is_bounded_and_can_be_disabled(self):
        for depth in (0, 2):
            with ImagePrefetcher(self.root, adapter='official', prefetch_batches=depth, official_frames=8) as loader:
                with patch.object(loader, '_submit', wraps=loader._submit) as submit:
                    batches = loader.batches(self.rows, 1)
                    next(batches)
                    self.assertEqual(submit.call_count, depth+1)
                    next(batches)
                    self.assertEqual(submit.call_count, min(depth+2, len(self.rows)))
                    batches.close()

    def test_reads_overlap_before_consumer_requests_next_batch(self):
        # Barrier proves eight readers can be active, rather than relying on timing.
        barrier = threading.Barrier(8, timeout=10)
        with ImagePrefetcher(self.root, adapter='official', workers=8, official_frames=8) as loader:
            read = loader._read
            def concurrent_read(path):
                barrier.wait()
                return read(path)
            with patch.object(loader, '_read', side_effect=concurrent_read):
                batches = list(loader.batches(self.rows[:2], 1))
            self.assertEqual(len(batches), 2)

    def test_missing_image_is_reported_without_silent_skip(self):
        (self.root/self.rows[0]['image_paths'][4]).unlink()
        with ImagePrefetcher(self.root, adapter='icl') as loader:
            with self.assertRaises(FileNotFoundError):
                next(loader.batches(self.rows, 1))

    def test_official_matches_serial_for_all_sampling_modes(self):
        class Model:
            def compute_batched_progress(self, samples):
                return [[float(f[0,0,0])/255 for f in s.trajectory.frames] for s in samples]
        for count in (8,16,32):
            adapter = OfficialAdapter.__new__(OfficialAdapter)
            adapter.root, adapter.sample_frames, adapter.model = self.root, count, Model()
            adapter.report = {'interpolation':'none' if count == 32 else 'linear_in_stored_frame_index'}
            expected = adapter.predict_many(self.rows[:2])
            with ImagePrefetcher(self.root, adapter='official', official_frames=count) as loader:
                actual = predict_official(adapter, next(loader.batches(self.rows[:2], 2)))
            self.assertEqual(expected, actual)

    def test_icl_inference_matches_serial_across_partial_frame_batches(self):
        import torch
        adapter = ICLAdapter.__new__(ICLAdapter)
        adapter.root, adapter.batch_size = self.root, 7
        adapter.reference_cache = {}
        adapter.processor = None
        adapter.cfg = SimpleNamespace(data=SimpleNamespace(max_keyframes=6,max_pixels=None),
                                      model=SimpleNamespace(precision='bf16'))
        def build(processor, task, references, queries, max_keyframes, max_pixels):
            self.assertEqual([x.getpixel((0,0))[0] for x in references], [0,8,16,24,31])
            return {'query':torch.tensor([x.getpixel((0,0))[0]/255 for x in queries])}
        def model(query):
            logits = query[:, None]*torch.arange(5)[None, :]
            return {'logits':torch.cat([logits,torch.full((len(query),1),float('-inf'))],dim=1)}
        adapter.model = model
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == 'cuda:0':
                args = ('cpu',) + args[1:]
            return original_to(tensor, *args, **kwargs)
        with patch('eval_astra_bench.adapters.build_icl_inputs', side_effect=build), \
             patch('eval_astra_bench.loaded_inference.build_icl_inputs', side_effect=build), \
             patch.object(torch.Tensor, 'to', cpu_to), patch('torch.autocast', return_value=nullcontext()):
            expected = adapter.predict(self.rows[1])
            with ImagePrefetcher(self.root, adapter='icl') as loader:
                actual = predict_icl(adapter, next(loader.batches(self.rows[1:2], 1))[0])
        self.assertEqual(expected, actual)

    def test_cpu_slots_use_disjoint_physical_cores(self):
        groups = []
        for slot in (0, 1):
            code = ('import json,os; from eval_astra_bench.prefetch import configure_cpu; '
                    f'cpus=configure_cpu(8,{slot}); '
                    'print(json.dumps([cpus,sorted(os.sched_getaffinity(0))]))')
            cpus, affinity = json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
            self.assertEqual(cpus, affinity)
            self.assertEqual(len(cpus), 8)
            groups.append(set(cpus))
        self.assertFalse(groups[0] & groups[1])


if __name__ == '__main__':
    unittest.main()
