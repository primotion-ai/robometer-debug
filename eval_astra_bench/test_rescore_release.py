"""Prevent stale predictions or incompatible frame pairs in label-only rescoring."""
import copy
import unittest
from .rescore_release import validate_reuse, relabel_pairs


class RescoreTests(unittest.TestCase):
    def fixture(self):
        old = dict(trajectory_id='a',instruction='task',reference_group='g',
                   image_paths=[f'old/{i}' for i in range(32)],
                   icl_reference=dict(trajectory_id='a',frame_indices=[0,8,16,24,31],
                                      image_paths=[f'old/{i}' for i in (0,8,16,24,31)]))
        new = dict(trajectory_id='a',instruction='task',group='g',num_frames=32,
                   image_paths=[f'new/{i}' for i in range(32)],image_sha256=[f'hash{i}' for i in range(32)])
        return old,new,{f'old/{i}':f'hash{i}' for i in range(32)},{f'new/{i}':f'hash{i}' for i in range(32)}

    def test_path_rename_is_allowed_but_input_changes_rejected(self):
        old,new,oc,nc=self.fixture()
        validate_reuse([old],[new],oc,nc)
        for field,value in [('instruction','different'),('group','different'),('num_frames',31)]:
            bad=dict(new,**{field:value})
            with self.assertRaises(ValueError):validate_reuse([old],[bad],oc,nc)
        bad=copy.deepcopy(new);bad['image_sha256'][7]='changed'
        with self.assertRaisesRegex(ValueError,'Image changed'):validate_reuse([old],[bad],oc,nc)

    def test_missing_reference_rejects_prediction_reuse(self):
        old,new,oc,nc=self.fixture();old['icl_reference']['trajectory_id']='missing'
        with self.assertRaisesRegex(ValueError,'reference missing'):validate_reuse([old],[new],oc,nc)

    def test_pairs_keep_endpoints_and_ids_but_get_new_gold(self):
        by={uid:dict(instruction='task',reference_sha256='new_scale') for uid in ('a','b')}
        targets={'a':{'progress':[.4]},'b':{'progress':[.3]}}
        pair=dict(pair_id='fixed',mode='cross',a=['a',0],b=['b',0],gold_relation='gt',involves_known_conflict=True)
        missing=dict(pair,pair_id='removed',b=['absent',0])
        result=relabel_pairs([pair,missing],by,targets)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['pair_id'],pair['pair_id'])
        self.assertEqual(result[0]['a'],pair['a']);self.assertEqual(result[0]['b'],pair['b'])
        self.assertEqual(result[0]['gold_relation'],'eq')
        self.assertEqual(pair['gold_relation'],'gt')

    def test_cross_pair_with_changed_scope_is_rejected(self):
        pair=dict(pair_id='fixed',mode='cross',a=['a',0],b=['b',0])
        by={'a':dict(instruction='task',reference_sha256='one'),
            'b':dict(instruction='task',reference_sha256='two')}
        with self.assertRaisesRegex(ValueError,'no longer shares'):relabel_pairs([pair],by,{})


if __name__=='__main__':unittest.main()
