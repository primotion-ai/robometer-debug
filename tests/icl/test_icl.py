import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from omegaconf import OmegaConf

from robometer.icl_progress.targets import soft_target, select_frames, collect_batch
from robometer.icl_progress.model import ICLProgressModel, progress_loss
from robometer.icl_progress.config import load_config
from robometer.icl_progress.data import TrajectoryStore, ICLBatchDataset, validation_member
from robometer.icl_progress.collator import build_conversation


class ICLTests(unittest.TestCase):
    def test_temporal_targets_and_reference_relative_scale(self):
        self.assertEqual(soft_target([0,30,90,100,180],60), (0,0.5,0.5,0,0))
        self.assertEqual(soft_target([10,20],15), (0.5,0.5))
        self.assertEqual(soft_target([0,20],15), (0.25,0.75))
        for query in (0,20,21):
            with self.assertRaises(ValueError):
                soft_target([0,20],query)

    def test_random_sampling_invariants_and_empty_interiors(self):
        rng = np.random.default_rng(42)
        sizes=set(); non_endpoints=False
        for _ in range(2000):
            sample=select_frames(80,2,11,rng)
            if sample is None:
                continue
            sizes.add(len(sample.keyframes))
            non_endpoints |= sample.keyframes[0]>0 and sample.keyframes[-1]<79
            self.assertNotIn(sample.query,sample.keyframes)
            self.assertLess(sample.keyframes[0],sample.query)
            self.assertLess(sample.query,sample.keyframes[-1])
            self.assertAlmostEqual(sum(sample.target),1)
            self.assertEqual(sum(x>0 for x in sample.target),2)
        self.assertEqual(sizes,set(range(2,12)))
        self.assertTrue(non_endpoints)
        self.assertIsNone(select_frames(2,2,11,rng))
        class Consecutive:
            def integers(self,*a): return 3
            def choice(self,*a,**kw): return np.array([10,11,12])
        self.assertIsNone(select_frames(80,2,11,Consecutive()))

    def test_b_plus_one_then_refill(self):
        candidates=iter([None,'a',None,'b',None,'c'])
        values,count=collect_batch(lambda:next(candidates),3)
        self.assertEqual(values,['a','b','c'])
        self.assertEqual(count,6)
        candidates=iter(range(20))
        values,count=collect_batch(lambda:next(candidates),3)
        self.assertEqual(values,[0,1,2]);self.assertEqual(count,4)
        with self.assertRaises(RuntimeError):
            collect_batch(lambda:None,3,max_rounds=2)

    def test_padding_does_not_affect_loss_or_gradient(self):
        logits=torch.tensor([[0.,0.,1000.],[0.,0.,0.]],requires_grad=True)
        labels=torch.tensor([[.5,.5,0.],[0.,.25,.75]])
        mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
        loss,masked=progress_loss(logits,labels,mask)
        self.assertAlmostEqual(loss.item(),float((np.log(2)+np.log(3))/2),places=6)
        loss.backward()
        self.assertEqual(logits.grad[0,2].item(),0)
        self.assertEqual(masked.softmax(-1)[0,2].item(),0)

    def test_variable_head_backward_has_no_legacy_heads(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__();self.emb=nn.Embedding(32,16)
                self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
            def forward(self,input_ids,**kw):
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))
        model=ICLProgressModel(Backbone(),8)
        result=model(input_ids=torch.arange(24).reshape(2,12),
                     anchor_positions=torch.tensor([[1,3,0],[1,3,5]]),query_positions=torch.tensor([9,11]),
                     anchor_mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool),
                     labels=torch.tensor([[.5,.5,0.],[0.,.25,.75]]))
        result['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(any('success_head' in n or 'preference_head' in n for n,_ in model.named_parameters()))

    def test_success_filter_and_trajectory_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);name='toy'
            records=[]
            for i in range(100):
                records.append(dict(id=str(i),task='task',data_source='toy',quality_label='successful',
                                    partial_success=None,frames_shape=[20,32,32,3],num_frames=20,frames='unused.npz'))
            records += [{**records[0],'id':'failure','quality_label':'failure'},
                        {**records[0],'id':'partial','partial_success':.5},
                        {**records[0],'id':'short','frames_shape':[2,32,32,3]}]
            Dataset.from_list(records).save_to_disk(str(root/name/'processed_dataset'))
            cfg=load_config('4b');cfg.data.root=str(root);cfg.data.datasets=[name];cfg.data.validation_fraction=.2
            store=TrajectoryStore(cfg.data)
            a=set(store.indices['train'][name]);b=set(store.indices['validation'][name])
            self.assertFalse(a&b);self.assertEqual(a|b,set(range(100)))
            self.assertEqual(store.summary[0]['non_success'],1)
            self.assertEqual(store.summary[0]['partial_conflict'],1)
            self.assertEqual(store.summary[0]['short'],1)
            dataset=ICLBatchDataset(store,cfg.data,4,100,123)
            one=dataset[9];two=dataset[9]
            self.assertEqual(one,two)
            self.assertEqual(len(one['samples']),4)
            resumed=ICLBatchDataset(store,cfg.data,4,100,123,offset=8)
            self.assertEqual(dataset[9],resumed[1])

    def test_prompt_has_roles_and_no_timestamps(self):
        conversation=build_conversation('task',['image1','image2'],'query')
        content=conversation[0]['content']
        text=' '.join(x['text'] for x in content if x['type']=='text')
        self.assertIn('END REFERENCE KEYFRAMES',text)
        self.assertIn('CURRENT QUERY IMAGE',text)
        self.assertIn('0.0000%',text);self.assertIn('100.0000%',text)
        self.assertEqual([x['image'] for x in content if x['type']=='image'],['image1','image2','query'])

    def test_all_recipes(self):
        for name in ('4b','27b','397b'):
            cfg=load_config(name)
            self.assertEqual(cfg.data.max_keyframes,11)
            self.assertEqual(cfg.data.datasets,['rbm-1m-id'])
            self.assertEqual(cfg.model.precision,'pure_fp16' if name=='397b' else 'bf16')
        with self.assertRaises(Exception):load_config('4b',['data.max_keyframes=12'])
        with self.assertRaises(Exception):load_config('4b',['data.typo=12'])


if __name__=='__main__':unittest.main()
