"""CPU tests for protocol semantics, reproducibility, and adapters' input construction."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from eval_astra_bench.core import (file_hash, load_benchmark, prediction_row, uniform_indices, prepare,
                                   read_jsonl, relation, score, validate_predictions, write_json, write_jsonl)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'source'
        (self.data/'manifests').mkdir(parents=True)
        write_json(self.data/'dataset.json', {'release':'synthetic-v1'})
        (self.data/'checksums.sha256').write_text('synthetic fixture\n')
        rows = []
        spec = [('a','g1','r1','pick','source','successful'),
                ('b','g1','r1','pick','source','failure'),
                ('c','g1','r2','pick','source','successful'),
                ('d','g2','r3','Fold the towel in half','other','successful'),
                ('excluded','g3','r4','Fold the towel in half','rfm_new_mit_franka_rfm','successful')]
        for uid, group, ref, task, source, quality in spec:
            progress = [i/31 for i in range(32)] if uid != 'b' else [1-i/31 for i in range(32)]
            path = self.data/f'{uid}.json'
            write_json(path, dict(trajectory_id=uid, reference_sha256=ref,
                frame_labels=[{'frame_index':i,'progress':p} for i,p in enumerate(progress)]))
            rows.append(dict(trajectory_id=uid, reference_group=group, group_name=group, reference_key=ref,
                             instruction=task, source=source, source_quality_label=quality,
                             num_frames=32, image_paths=[f'images/{uid}/{i}.png' for i in range(32)],
                             label_path=path.name, label_sha256=file_hash(path),
                             human_model_conflict=False, success_basis='visual', prompt_version='test'))
        write_jsonl(self.data/'manifests/trajectories.jsonl', rows)
        self.benchmark = self.root/'bench'
        with redirect_stdout(io.StringIO()):
            self.info = prepare(self.data, self.benchmark, seed=17, within_per_trajectory=9, cross_per_group=13)
        self.rows = read_jsonl(self.benchmark/'inputs.jsonl')

    def predictions(self, transform=lambda p:p):
        gold = {r['trajectory_id']:r['progress'] for r in read_jsonl(self.benchmark/'targets.jsonl')}
        return [prediction_row(self.info, row, [transform(p) for p in gold[row['trajectory_id']]]) for row in self.rows]

    def test_threshold_boundary_is_symmetric_and_inclusive(self):
        self.assertEqual(relation(.4,.3,.1),'eq')
        self.assertEqual(relation(.3,.4,.1),'eq')
        self.assertEqual(relation(.4000001,.3,.1),'gt')
        self.assertEqual(relation(.3,.4000001,.1),'lt')
        self.assertEqual(relation(.3,.3,0),'eq')

    def test_uniform_sampling_covers_full_trajectory(self):
        self.assertEqual(uniform_indices(32,8),[0,4,8,13,17,22,26,31])
        self.assertEqual(uniform_indices(32,32),list(range(32)))
        for n in [8,16,32]:
            indices = uniform_indices(32,n)
            self.assertEqual(indices[0],0)
            self.assertEqual(indices[-1],31)
            self.assertEqual(len(set(indices)),n)
        with self.assertRaises(ValueError):
            uniform_indices(32,33)

    def test_seeded_artifacts_reproducible_and_immutable(self):
        other = self.root/'copy'
        with redirect_stdout(io.StringIO()):
            info = prepare(self.data,other,seed=17,within_per_trajectory=9,cross_per_group=13)
        self.assertEqual(info['benchmark_id'],self.info['benchmark_id'])
        self.assertEqual((other/'pairs.jsonl').read_bytes(),(self.benchmark/'pairs.jsonl').read_bytes())
        with self.assertRaises(FileExistsError):
            prepare(self.data,self.benchmark)
        with (other/'pairs.jsonl').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError,'Frozen benchmark changed'):
            load_benchmark(other)

    def test_exclusion_pair_validity_reference_retention_and_scope(self):
        by_id = {r['trajectory_id']:r for r in self.rows}
        self.assertEqual(set(by_id),{'a','b','c','d'})
        pairs = read_jsonl(self.benchmark/'pairs.jsonl')
        seen = set()
        for p in pairs:
            a,b = p['a'],p['b']
            self.assertNotEqual(a,b)
            key = tuple(sorted([tuple(a),tuple(b)]))
            self.assertNotIn(key,seen)
            seen.add(key)
            self.assertEqual(by_id[a[0]]['instruction'],by_id[b[0]]['instruction'])
            if p['mode']=='cross':
                self.assertNotEqual(a[0],b[0])
                self.assertEqual(by_id[a[0]]['reference_key'],by_id[b[0]]['reference_key'])
            else:
                self.assertEqual(a[0],b[0])
        self.assertEqual(Counter(p['mode'] for p in pairs),{'within':36,'cross':13})
        refs = {r['icl_reference']['trajectory_id'] for r in self.rows}
        self.assertTrue(refs<=set(by_id))
        self.assertTrue(any(p['involves_icl_reference'] for p in pairs))
        self.assertEqual(len(self.info['skipped_cross_groups']),2)

    def test_perfect_predictions_and_constant_baseline(self):
        path = self.root/'predictions.jsonl'
        write_jsonl(path,self.predictions())
        result = score(self.benchmark,path,self.root/'perfect')
        self.assertEqual(result['overall']['accuracy'],1)
        write_jsonl(path,self.predictions(lambda p: .5))
        result = score(self.benchmark,path,self.root/'constant')
        pairs=read_jsonl(self.benchmark/'pairs.jsonl')
        eq=sum(p['gold_relation']=='eq' for p in pairs)
        self.assertEqual(result['overall']['correct'],eq)
        self.assertEqual(result['overall']['total'],len(pairs))
        self.assertEqual(result['confusion_matrix']['gt']['gt'],0)

    def test_invalid_predictions_fail_instead_of_changing_denominator(self):
        for mutation in ('missing','duplicate','nan','infinity','out_of_range','wrong_id','wrong_benchmark','wrong_frames'):
            rows=self.predictions()
            if mutation=='missing': rows.pop()
            elif mutation=='duplicate': rows.append(rows[0])
            elif mutation=='nan': rows[0]['progress'][0]=float('nan')
            elif mutation=='infinity': rows[0]['progress'][0]=float('inf')
            elif mutation=='out_of_range': rows[0]['progress'][0]=1.1
            elif mutation=='wrong_id': rows[0]['trajectory_id']='unknown'
            elif mutation=='wrong_benchmark': rows[0]['benchmark_id']='wrong'
            elif mutation=='wrong_frames': rows[0]['frame_indices'].reverse()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_predictions(self.info,self.rows,rows)

    def test_bad_source_annotation_hash_rejected(self):
        (self.data/'a.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Source annotation hash mismatch'):
            prepare(self.data,self.root/'bad')

    def test_task_scope_crosses_scales_only_when_explicit(self):
        with redirect_stdout(io.StringIO()):
            prepare(self.data,self.root/'task',pair_scope='task',within_per_trajectory=0,cross_per_group=100000)
        pairs=read_jsonl(self.root/'task/pairs.jsonl')
        self.assertEqual(len(pairs),3*1024)
        self.assertTrue(any('c' in (p['a'][0],p['b'][0]) for p in pairs))


if __name__ == '__main__':
    unittest.main()
