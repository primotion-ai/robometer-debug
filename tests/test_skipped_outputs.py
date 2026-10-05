import unittest
from types import SimpleNamespace

import torch

from robometer.models.utils import ModelOutput
from robometer.trainers.rbm_heads_trainer import RBMHeadsTrainer
from robometer.utils.skipped_updates import SkippedUpdates


class SkippedOutputTests(unittest.TestCase):
    def test_production_loss_keeps_rejected_graph_and_bypasses_metrics(self):
        for sample_type in ("preference", "progress"):
            with self.subTest(sample_type=sample_type):
                trainer = object.__new__(RBMHeadsTrainer)
                trainer.args = SimpleNamespace(device=torch.device("cpu"))
                trainer.config = SimpleNamespace(model=SimpleNamespace(train_preference_head=True, train_progress_head=True))
                trainer._ddp_static_graph_set = True
                trainer.timing_raw = {}
                trainer.skipped_updates = SkippedUpdates(trainer)
                x = torch.tensor([1.0], requires_grad=True)
                output = ModelOutput(pref_logits=x * float("nan"))
                trainer.forward_model = lambda *a, **kw: (output, {})
                inputs = {f"{sample_type}_inputs": {"present": True},
                          "num_preferences": int(sample_type == "preference"),
                          "num_progress": int(sample_type == "progress")}
                loss = trainer.compute_loss(SimpleNamespace(training=True), inputs)
                self.assertTrue(torch.isnan(loss))
                self.assertTrue(loss.requires_grad)
                loss.backward()  # Graph remains connected for FSDP cleanup.
                self.assertIsNotNone(x.grad)
                self.assertEqual(trainer.skipped_updates.output_errors, ["outputs.pref_logits"])
                self.assertEqual(trainer.log_metadata, {})

    def test_finite_outputs_are_not_modified(self):
        trainer = SimpleNamespace(args=SimpleNamespace(device=torch.device("cpu")))
        guard = SkippedUpdates(trainer)
        x = torch.tensor([1.0], requires_grad=True)
        output = ModelOutput(pref_logits=x)
        self.assertIsNone(guard.rejected_outputs(output))
        self.assertIs(output.pref_logits, x)
        x.square().sum().backward()
        self.assertEqual(x.grad.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
