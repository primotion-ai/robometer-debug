"""Large max_steps runs must not allocate a permutation of 1000 virtual epochs."""
import unittest
import tempfile
from unittest.mock import patch

import torch

from robometer.configs.experiment_configs import DataConfig
from robometer.data.datasets.repeated_dataset import RepeatedDataset
from robometer.utils.setup_utils import setup_dataset
from scripts.launch_b200 import compose_config


class DatasetRepeatsTests(unittest.TestCase):
    def test_b200_recipes_use_real_epoch_length(self):
        for name, max_steps in (("qwen27b", 5000), ("qwen397b", 15000)):
            cfg = compose_config(name, [])
            self.assertEqual(cfg.data.dataset_num_repeats, 1)
            self.assertEqual(cfg.training.max_steps, max_steps)
        self.assertEqual(DataConfig().dataset_num_repeats, 1000)

    def test_setup_and_sampler_cover_each_real_item(self):
        underlying = list(range(12))
        cfg = DataConfig(dataset_type="rbm", dataset_num_repeats=1)
        with patch("robometer.utils.setup_utils.RBMDataset", return_value=underlying):
            dataset = setup_dataset(cfg)
            self.assertIs(dataset.dataset, underlying)
            self.assertEqual(len(dataset), len(underlying))
            indices = list(torch.utils.data.RandomSampler(dataset))
            self.assertEqual(sorted(indices), underlying)
            self.assertIs(setup_dataset(cfg, is_eval=True), underlying)
            cfg.dataset_num_repeats = 1000
            self.assertEqual(len(setup_dataset(cfg)), 12000)

    def test_repeat_validation_and_modulo(self):
        self.assertEqual(RepeatedDataset(["a", "b"], 3)[5], "b")
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                RepeatedDataset([1], value)

    def test_max_steps_continues_across_real_epochs(self):
        from transformers import Trainer, TrainingArguments

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, value):
                return {"loss": (self.weight * value).square().mean()}

        with tempfile.TemporaryDirectory() as folder:
            dataset = RepeatedDataset([{"value": torch.tensor(1.0)} for _ in range(2)], 1)
            trainer = Trainer(
                model=Model(), train_dataset=dataset,
                args=TrainingArguments(output_dir=folder, use_cpu=True, max_steps=3,
                                       per_device_train_batch_size=2, report_to=[],
                                       save_strategy="no", logging_strategy="no", disable_tqdm=True),
            )
            trainer.train()
            self.assertEqual(trainer.state.global_step, 3)
            self.assertEqual(trainer.state.epoch, 3)


if __name__ == "__main__":
    unittest.main()
