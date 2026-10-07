"""The 27B recipe must preserve early checkpoints for evaluation after training."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from transformers import Trainer, TrainingArguments

from robometer.utils.setup_utils import create_training_arguments
from scripts.launch_b200 import compose_config


class CheckpointScheduleTests(unittest.TestCase):
    def make_cpu_arguments(self, cfg, folder):
        training = replace(
            OmegaConf.to_object(cfg.training), bf16=False, fp16=False,
            dataloader_num_workers=0, dataloader_persistent_workers=False,
            dataloader_pin_memory=False,
        )
        with patch(
            "robometer.utils.setup_utils.TrainingArguments",
            side_effect=lambda **kwargs: TrainingArguments(use_cpu=True, **kwargs),
        ):
            return create_training_arguments(training, folder)

    def test_recipes_pass_their_schedule_to_trainer(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, steps, interval, limit in (
                ("qwen27b", 5000, 1000, 5),
                ("qwen397b", 2000, 1000, 3),
                ("qwen4b", 20000, 4000, 5),
            ):
                with self.subTest(recipe=name):
                    cfg = compose_config(name, [])
                    args = self.make_cpu_arguments(cfg, folder)
                    self.assertEqual(args.max_steps, steps)
                    self.assertEqual(args.save_strategy.value, "steps")
                    self.assertEqual(args.save_steps, interval)
                    self.assertEqual(args.save_total_limit, limit)
                    self.assertEqual(args.eval_steps, 500)
                    self.assertEqual(cfg.training.custom_eval_steps, 500)

    def test_4b_production_recipe(self):
        cfg = compose_config("qwen4b", [])
        self.assertEqual(cfg.model.torch_dtype, "float32")
        self.assertEqual(cfg.model.attn_implementation, "sdpa")
        self.assertEqual(cfg.model.linear_attention_backend, "auto")
        self.assertFalse(cfg.model.use_unsloth)
        self.assertTrue(cfg.training.bf16)
        self.assertEqual(cfg.training.per_device_train_batch_size, 32)
        self.assertEqual(cfg.training.gradient_accumulation_steps, 1)
        self.assertEqual(cfg.training.learning_rate, 2e-5)
        self.assertEqual(cfg.training.warmup_ratio, 0.1)
        self.assertEqual(cfg.data.dataset_num_repeats, 1)
        self.assertEqual(cfg.data.max_frames, 8)
        self.assertIsNone(cfg.data.resized_height)
        self.assertEqual(list(cfg.logging.save_best.metric_names), [])
        self.assertIsNone(cfg.logging.save_best.save_every)

    def test_retention_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = compose_config("qwen27b", ["training.save_total_limit=null"])
            self.assertIsNone(self.make_cpu_arguments(cfg, folder).save_total_limit)

    def test_all_five_periodic_checkpoints_remain_resumable(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, value):
                return {"loss": (self.weight * value).square().mean()}

        with tempfile.TemporaryDirectory() as folder:
            # Scale 5000/1000 to 5/1 to exercise real Trainer checkpoint rotation.
            cfg = compose_config("qwen27b", [
                "training.max_steps=5", "training.save_steps=1",
                "training.evaluation_strategy=no", "training.do_eval=false",
                "training.gradient_checkpointing=false",
            ])
            args = self.make_cpu_arguments(cfg, folder)
            args.disable_tqdm = True
            trainer = Trainer(
                model=Model(), args=args,
                train_dataset=[{"value": torch.tensor(1.0)}],
            )
            trainer.train()
            checkpoints = sorted(Path(folder).glob("checkpoint-*"))
            self.assertEqual([p.name for p in checkpoints], [f"checkpoint-{i}" for i in range(1, 6)])
            for checkpoint in checkpoints:
                for filename in ("model.safetensors", "optimizer.pt", "trainer_state.json"):
                    self.assertTrue((checkpoint / filename).is_file(), checkpoint / filename)


if __name__ == "__main__":
    unittest.main()
