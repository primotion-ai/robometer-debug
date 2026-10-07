import tempfile
import unittest

import torch
import torch.distributed.checkpoint as dcp

from robometer.evals.checkpoint_loader import validate_shapes
from robometer.evals.success_metrics import final_success_metrics


class OfflineEvalTests(unittest.TestCase):
    def test_always_choose_a_scores_half_on_balanced_pairs(self):
        from robometer.data.dataset_types import Trajectory, PreferenceSample
        from robometer.evals.run_baseline_eval import process_batched_rbm_samples
        from robometer.evals.compile_results import run_quality_preference_eval
        class AlwaysA:
            def compute_batched_preference(self, batch):
                return [{'prediction_prob': .9, 'preference_pred': 1., 'is_correct': True} for _ in batch]
        dataset = [PreferenceSample(
            chosen_trajectory=Trajectory(id='a', task='pick', quality_label='successful'),
            rejected_trajectory=Trajectory(id='b', task='pick', quality_label='failure'))]
        results = process_batched_rbm_samples(dataset, AlwaysA(), balanced_preferences=True)
        self.assertEqual([r['preference_labels'] for r in results], [1., 0.])
        metrics, _, _ = run_quality_preference_eval(results)
        self.assertEqual(metrics['preference_accuracy'], .5)

    def test_preference_accepts_json_scalars(self):
        from robometer.evals.compile_results import run_quality_preference_eval
        rows = [{'preference_pred': 1.0, 'preference_labels': 1.0, 'task': 'pick',
                 'metadata': {'chosen_metadata': {'quality_label': 'successful'},
                              'rejected_metadata': {'quality_label': 'failure'}}}]
        metrics, _, _ = run_quality_preference_eval(rows)
        self.assertEqual(metrics['preference_accuracy'], 1.0)

    def test_dcp_cast_preserves_every_tensor(self):
        original = {'progress_head.0.weight': torch.randn(4, 3),
                    'model.language_model.embed_tokens.weight': torch.randn(17, 3)}
        with tempfile.TemporaryDirectory() as path:
            dcp.save({'model': original}, checkpoint_id=path)
            shapes = {k[6:]: v.size for k, v in dcp.FileSystemReader(path).read_metadata().state_dict_metadata.items()}
            weights = {k: torch.empty(s, dtype=torch.bfloat16) for k, s in shapes.items()}
            validate_shapes(original, shapes)
            dcp.load({'model': weights}, checkpoint_id=path)
            for k in original:
                torch.testing.assert_close(weights[k], original[k].bfloat16(), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                validate_shapes(original, dict(shapes, unexpected=(1,)))
            with self.assertRaises(ValueError):
                validate_shapes(original, {})

    def test_success_uses_last_frame_and_handles_missing_class(self):
        rows = [{'id': 'a', 'quality_label': 'successful', 'success_probs': [.1, .9]},
                {'id': 'b', 'quality_label': 'failure', 'success_probs': [.9, .1]},
                {'id': 'c', 'quality_label': 'unknown', 'success_probs': [.9]}]
        result = final_success_metrics(rows)
        self.assertEqual(result['success_n'], 2)
        self.assertEqual(result['success_skipped_n'], 1)
        self.assertEqual(result['success_auroc'], 1)
        self.assertEqual(result['success_accuracy_0.5'], 1)
        self.assertNotIn('success_auroc', final_success_metrics(rows[:1]))
        with self.assertRaises(ValueError):
            final_success_metrics(rows[:1] * 2)


if __name__ == '__main__':
    unittest.main()
