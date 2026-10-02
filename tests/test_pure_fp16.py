import tempfile
import unittest
from types import SimpleNamespace

import torch

from scripts.launch_b200 import compose_config
from robometer.utils.pure_fp16 import create_pure_fp16_optimizer, validate_pure_fp16_config
from robometer.utils.setup_utils import create_training_arguments
from robometer.configs.experiment_configs import TrainingConfig


class PureFP16Tests(unittest.TestCase):
    def test_recipe_isolation(self):
        large = compose_config("qwen397b", [])
        validate_pure_fp16_config(large)
        self.assertTrue(large.training.pure_fp16)
        self.assertEqual(large.model.torch_dtype, "float16")
        for name in ("config", "qwen27b"):
            cfg = compose_config(name, [])
            self.assertFalse(cfg.training.pure_fp16)
            self.assertEqual(cfg.training.gradient_checkpointing_backend, "transformers")
            self.assertEqual(cfg.model.torch_dtype, "bfloat16")
        small = compose_config("qwen27b", [])
        self.assertTrue(small.training.bf16)
        self.assertFalse(small.training.fp16)

    def test_reject_accidental_amp_or_disabled_checkpointing(self):
        for override in ("training.fp16=true", "training.bf16=true", "model.torch_dtype=bfloat16",
                         "training.gradient_checkpointing=false", "training.gradient_checkpointing_backend=transformers"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_pure_fp16_config(compose_config("qwen397b", [override]))

    def test_args_use_fsdp_checkpointing_without_double_wrapping(self):
        cfg = TrainingConfig(pure_fp16=True, bf16=False, fp16=False, gradient_checkpointing=True,
                             gradient_checkpointing_backend="fsdp")
        with tempfile.TemporaryDirectory() as directory:
            args = create_training_arguments(cfg, directory)
            self.assertFalse(args.gradient_checkpointing)
            self.assertFalse(args.fp16 or args.bf16)
            self.assertEqual(args.adam_epsilon, 1e-4)

    def test_half_adam_states_and_finite_zero_gradient_update(self):
        model = torch.nn.Linear(2, 2).half()
        args = SimpleNamespace(weight_decay=0.01, learning_rate=5e-6,
                               adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-4)
        optimizer = create_pure_fp16_optimizer(model, args, ["weight"])
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
        optimizer.step()
        for param, values in optimizer.state.items():
            self.assertEqual(param.dtype, torch.float16)
            self.assertTrue(torch.isfinite(param).all())
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(values[key].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
