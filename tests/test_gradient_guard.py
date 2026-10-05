"""Regression: a finite loss can still have non-finite gradients."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from robometer.trainers.rbm_heads_trainer import RBMHeadsTrainer


class GradientGuardTests(unittest.TestCase):
    def trainer(self, directory, enabled=True):
        trainer = object.__new__(RBMHeadsTrainer)
        trainer.config = SimpleNamespace(training=SimpleNamespace(pure_fp16=enabled))
        trainer.args = SimpleNamespace(output_dir=directory)
        trainer.state = SimpleNamespace(global_step=133)
        trainer.accelerator = SimpleNamespace(
            process_index=0, distributed_type="FSDP",
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
        )
        return trainer

    def test_invalid_gradient_never_reaches_adam_update(self):
        for invalid in (float("nan"), float("inf")):
            for explicit_clipping in (True, False):
                with self.subTest(invalid=invalid, explicit_clipping=explicit_clipping):
                    model = torch.nn.Linear(1, 1, bias=False).half()
                    opt = torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-4)
                    model.weight.grad = torch.ones_like(model.weight)
                    opt.step()  # Model and moments already exist, as at step 134.
                    before = model.weight.detach().clone()
                    state_before = copy.deepcopy(opt.state[model.weight])
                    model.weight.grad.fill_(invalid)
                    with tempfile.TemporaryDirectory() as directory:
                        trainer = self.trainer(directory)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10) if explicit_clipping else None
                        with self.assertRaisesRegex(FloatingPointError, "before optimizer update 134"):
                            trainer._get_grad_norm(model, norm)
                            opt.step()
                        self.assertTrue(torch.equal(model.weight, before))
                        for key, value in state_before.items():
                            self.assertTrue(torch.equal(opt.state[model.weight][key], value))
                        report = json.loads((Path(directory) / "nonfinite_gradient_rank_0.json").read_text())
                        self.assertEqual(report["completed_steps"], 133)
                        self.assertFalse(report["optimizer_step_executed"])

    def test_finite_gradients_continue_unchanged(self):
        model = torch.nn.Linear(1, 1, bias=False).half()
        model.weight.grad = torch.full_like(model.weight, 2)
        grad = model.weight.grad.clone()
        with tempfile.TemporaryDirectory() as directory:
            norm = torch.tensor(2.0, dtype=torch.float16)
            self.assertIs(self.trainer(directory)._get_grad_norm(model, norm), norm)
            self.assertTrue(torch.equal(model.weight.grad, grad))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_other_recipes_keep_existing_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            norm = torch.tensor(float("nan"))
            self.assertIs(self.trainer(directory, enabled=False)._get_grad_norm(None, norm), norm)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
