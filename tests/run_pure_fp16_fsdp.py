"""Small two-GPU MoE integration check, launched with the 397B FSDP config.

Uses synthetic images and all three RBM heads; never starts W&B or loads 397B.
PURE_FP16_TEST_DIR selects a temporary checkpoint directory. Optional
PURE_FP16_TEST_RESUME restores a checkpoint in a separate invocation.
"""
import os
import sys
import faulthandler
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import PartialState
from transformers import Trainer, TrainerCallback, Qwen3_5MoeModel

from test_qwen_backbones import tiny_config
from scripts.launch_b200 import compose_config
from robometer.utils.config_utils import convert_hydra_to_dataclass
from robometer.configs.experiment_configs import ExperimentConfig
from robometer.models.rbm import RBM
from robometer.trainers.rbm_heads_trainer import RBMHeadsTrainer
from robometer.utils.setup_utils import create_training_arguments
from robometer.utils.qwen_kernels import (
    configure_qwen_linear_attention, stabilize_qwen_fp16_shared_experts, optimize_qwen_moe_expert_backward,
)
from robometer.utils.logger import Logger


class FiniteUpdate(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        self.before = [p.detach().clone() for group in optimizer.param_groups for p in group["params"]]

    def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        params = [p for group in optimizer.param_groups for p in group["params"]]
        assert all(torch.isfinite(p).all() for p in params)
        assert any(not torch.equal(a, b) for a, b in zip(self.before, params)), "No parameter changed"
        for values in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in values:
                    assert torch.isfinite(values[key]).all()
        del self.before


class SkippedStateAudit(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        # If global_step did not advance since the previous attempt, neither
        # parameters nor AdamW moments/counters may have changed.
        if hasattr(self, "previous_step") and state.global_step == self.previous_step:
            for before, parameter in zip(self.parameters, model.parameters()):
                assert torch.equal(before, parameter), "Skipped batch changed a parameter"
            now = optimizer.state_dict()
            assert now["state"].keys() == self.optimizer_state["state"].keys()
            for key, values in now["state"].items():
                for name, value in values.items():
                    before = self.optimizer_state["state"][key][name]
                    assert torch.equal(before, value) if isinstance(value, torch.Tensor) else before == value
        import copy
        self.previous_step = state.global_step
        self.parameters = [p.detach().clone() for p in model.parameters()]
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())


class TinyTrainer(RBMHeadsTrainer):
    # Keep the real Trainer FSDP/optimizer/callback/save lifecycle, replace only
    # the production dataset/loss metrics with a tiny three-head objective.
    def training_step(self, model, inputs, num_items_in_batch=None):
        return Trainer.training_step(self, model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        output, _ = model(**inputs, sample_type="preference")
        values = [output.pref_logits, *output.progress_logits.values(), *output.success_logits.values()]
        loss = sum(v.float().square().mean() for v in values if v is not None)
        assert torch.isfinite(loss)
        if os.environ.get("PURE_FP16_TEST_SKIPS") == "1":
            attempt = self.skipped_updates.attempts
            if attempt == 5:
                if self.accelerator.process_index == 1:
                    output.pref_logits = output.pref_logits * float("nan")
                rejected = self.skipped_updates.rejected_outputs(output)
                assert rejected is not None
                loss = rejected
            if self.accelerator.process_index == 1:
                if attempt == 1:
                    loss = loss + 11  # Finite loss above the threshold on ONE rank.
                elif attempt == 2:
                    loss = loss * float("nan")
                elif attempt == 3:
                    loss.register_hook(lambda grad: grad * float("inf"))
        return (loss, output) if return_outputs else loss


torch.manual_seed(42)
torch.set_num_threads(2)
faulthandler.dump_traceback_later(45, repeat=True)
# Match train.py: establish the process group before model construction.
PartialState()
cfg = convert_hydra_to_dataclass(compose_config("qwen397b", []), ExperimentConfig)
c = tiny_config("qwen3_5_moe")
c.text_config.linear_key_head_dim = c.text_config.linear_value_head_dim = 128
c.vision_config.hidden_size = 288
c._attn_implementation = "sdpa"
c._experts_implementation = "eager"
c.text_config.use_cache = False
backbone = configure_qwen_linear_attention(Qwen3_5MoeModel(c), "fla").half()
stabilize_qwen_fp16_shared_experts(backbone)
optimize_qwen_moe_expert_backward(backbone)
backbone.visual.requires_grad_(False)
tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda t: {"<|prog_token|>": 5, "<|pref_token|>": 6}[t])
model = RBM(c, SimpleNamespace(tokenizer=tokenizer), tokenizer, base_model=backbone,
            base_model_id=cfg.model.base_model_id, model_config=cfg.model).half()
root = os.environ["PURE_FP16_TEST_DIR"]
cfg.training.max_steps = 3 if os.environ.get("PURE_FP16_TEST_RESUME") else 2
cfg.training.per_device_train_batch_size = 1
cfg.training.evaluation_strategy = "no"
cfg.training.do_eval = False
cfg.training.save_steps = 1
cfg.training.logging_steps = 1
cfg.training.warmup_ratio = 0.0
cfg.training.dataloader_num_workers = 0
cfg.training.dataloader_persistent_workers = False
args = create_training_arguments(cfg.training, root)
ids = torch.tensor([[1, 8, 7, 9, 5, 8, 7, 9, 5, 6]])
inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
              pixel_values=torch.randn(8, 3 * 2 * 16 * 16, dtype=torch.float16),
              image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]))
trainer = TinyTrainer(config=cfg, model=model, args=args, train_dataset=[0] * 8,
                      data_collator=lambda _: inputs, callbacks=[FiniteUpdate(), SkippedStateAudit()],
                      logger=Logger(log_to=[], output_dir=root, is_main_process=False))
trainer.train(resume_from_checkpoint=os.environ.get("PURE_FP16_TEST_RESUME"))
assert trainer.state.global_step == cfg.training.max_steps
if os.environ.get("PURE_FP16_TEST_SKIPS") == "1":
    import json
    assert trainer.skipped_updates.skipped == 4
    assert trainer.skipped_updates.attempts == cfg.training.max_steps + 4
    assert trainer.lr_scheduler.last_epoch == cfg.training.max_steps
    cursor = json.loads((Path(root) / f"checkpoint-{cfg.training.max_steps}" / "skipped_update_state.json").read_text())
    assert cursor["skipped"] == 4 and cursor["attempts"] == cfg.training.max_steps + 4
    print(f"SKIPPED_UPDATES_OK rank={trainer.accelerator.process_index} attempts={trainer.skipped_updates.attempts}", flush=True)

print(f"PURE_FP16_FSDP_OK rank={trainer.accelerator.process_index} steps={trainer.state.global_step}", flush=True)
faulthandler.cancel_dump_traceback_later()
torch.distributed.destroy_process_group()
