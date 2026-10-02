import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import TrainingArguments

from robometer.configs.experiment_configs import LoggingConfig, TrainingConfig
from robometer.trainers.rbm_heads_trainer import RBMHeadsTrainer


class QuadraticTrainer(RBMHeadsTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = model(inputs["x"]).square().mean()
        return (loss, {}) if return_outputs else loss

    def _log_metadata(self):
        pass


class AccumulationTests(unittest.TestCase):
    def test_two_microbatches_contribute_to_one_update(self):
        with tempfile.TemporaryDirectory() as folder:
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1)
            trainer = QuadraticTrainer(
                config=SimpleNamespace(training=TrainingConfig(vision_encoder_lr=None), logging=LoggingConfig()),
                model=model,
                args=TrainingArguments(
                    output_dir=folder, use_cpu=True, max_steps=1,
                    per_device_train_batch_size=1, gradient_accumulation_steps=2,
                    optim="sgd", learning_rate=0.1, lr_scheduler_type="constant",
                    max_grad_norm=0, weight_decay=0, report_to=[],
                    save_strategy="no", logging_strategy="no", disable_tqdm=True,
                    remove_unused_columns=False, dataloader_pin_memory=False,
                ),
                train_dataset=[{"x": torch.tensor([1.0])}, {"x": torch.tensor([2.0])}],
            )
            trainer._fsdp_diagnostics_logged = True
            trainer.train()
            # Mean gradient of (w*1)^2 and (w*2)^2 at w=1 is 5.
            torch.testing.assert_close(model.weight, torch.tensor([[0.5]]))


if __name__ == "__main__":
    unittest.main()
