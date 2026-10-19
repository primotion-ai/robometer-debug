"""Adapter tests; run with the repository ML environment (.venv/bin/python)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from eval_astra_bench.adapters import OfficialAdapter, build_icl_inputs
from eval_astra_bench.core import uniform_indices


class OfficialProtocolTests(unittest.TestCase):
    def test_whole_trajectory_outputs_and_sparse_interpolation(self):
        class FakeModel:
            def __init__(self):
                self.calls = []
            def compute_batched_progress(self, samples):
                self.calls.append(samples)
                return [[float(f[0,0,0])/31 for f in s.trajectory.frames] for s in samples]
        images = [Image.fromarray(np.full((4,4,3),i,dtype=np.uint8)) for i in range(32)]
        row = dict(trajectory_id='test',instruction='pick',num_frames=32,image_paths=[str(i) for i in range(32)])
        for count in (8,16,32):
            adapter = OfficialAdapter.__new__(OfficialAdapter)
            adapter.root = Path('/unused')
            adapter.sample_frames = count
            adapter.model = FakeModel()
            adapter.report = {'interpolation':'none' if count==32 else 'linear_in_stored_frame_index'}
            with patch('eval_astra_bench.adapters.read_images', side_effect=lambda root,paths:[images[int(p)] for p in paths]):
                values = adapter.predict(row)
            np.testing.assert_allclose(values,np.arange(32)/31,rtol=0,atol=1e-15)
            self.assertEqual(len(adapter.model.calls),1)
            self.assertEqual(len(adapter.model.calls[0][0].trajectory.frames),count)
            self.assertEqual(adapter.last_details['sampled_frame_indices'],uniform_indices(32,count))

    def test_multiple_trajectories_keep_their_own_values_and_frame_mapping(self):
        class Model:
            def __init__(self): self.calls=0
            def compute_batched_progress(self, samples):
                self.calls+=1
                return [[float(f[0,0,0])/100 for f in sample.trajectory.frames] for sample in samples]
        adapter=OfficialAdapter.__new__(OfficialAdapter)
        adapter.root=Path('/unused');adapter.sample_frames=8;adapter.model=Model()
        adapter.report={'interpolation':'linear_in_stored_frame_index'}
        rows=[dict(trajectory_id=str(j),instruction='task '+str(j),num_frames=32,
                   image_paths=[str(40*j+i) for i in range(32)]) for j in range(2)]
        with patch('eval_astra_bench.adapters.read_images',side_effect=lambda root,paths:[
                Image.fromarray(np.full((4,4,3),int(p),dtype=np.uint8)) for p in paths]):
            results=adapter.predict_many(rows)
        self.assertEqual(adapter.model.calls,1)
        self.assertEqual(len(results),2)
        for j,(progress,details) in enumerate(results):
            np.testing.assert_allclose(progress,(40*j+np.arange(32))/100,atol=1e-15)
            self.assertEqual(details['sampled_frame_indices'],uniform_indices(32,8))

    def test_single_terminal_output_is_not_broadcast(self):
        class BadModel:
            def compute_batched_progress(self,samples):
                return [[.7]]
        adapter=OfficialAdapter.__new__(OfficialAdapter)
        adapter.root=Path('/unused');adapter.sample_frames=8;adapter.model=BadModel()
        row=dict(trajectory_id='test',instruction='pick',num_frames=32,image_paths=['image']*32)
        with patch('eval_astra_bench.adapters.read_images',return_value=[Image.new('RGB',(4,4))]*8):
            with self.assertRaisesRegex(ValueError,'one progress value per sampled frame'):
                adapter.predict(row)


class ICLInputTests(unittest.TestCase):
    def test_exact_training_collator_equivalence(self):
        import torch
        from transformers import AutoProcessor
        from robometer.icl_progress.collator import ICLCollator
        from robometer.icl_progress.targets import FrameSelection, soft_target
        checkpoint=Path('/mnt/liuzihao/project/robometer/logs/icl_4b_001/checkpoint-15000/processor')
        if not checkpoint.exists():
            self.skipTest('Local checkpoint processor is not available')
        processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
        rng=np.random.default_rng(5)
        frames=rng.integers(0,256,(32,56,56,3),dtype=np.uint8)
        indices=(0,8,16,24,31)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frames.npz'
            np.savez(path,frames=frames)
            selection=FrameSelection(indices,10,soft_target(indices,10))
            expected=ICLCollator(processor,tmp,11,None)([{'samples':[{
                'trajectory':{'frames':str(path),'task':'Pick up the object'},'selection':selection}]}])
            actual=build_icl_inputs(processor,'Pick up the object',
                [ICLCollator._pil(frames[i]) for i in indices],[ICLCollator._pil(frames[10])],11,None)
            self.assertEqual(set(expected)-{'labels'},set(actual))
            for key in actual:
                torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
