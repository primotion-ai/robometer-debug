"""Opt-in 397B pure FP16 storage; other recipes keep their existing AMP path.

Kernel reductions/loss arithmetic may still use FP32 for numerical stability.
There is no FP32 master parameter or parameter-sized FP32 AdamW state here.
"""

import json
from pathlib import Path

import torch
from transformers import TrainerCallback


def validate_pure_fp16_config(cfg):
    if not cfg.training.pure_fp16:
        return
    if cfg.model.torch_dtype != "float16" or cfg.training.bf16 or cfg.training.fp16:
        raise ValueError("pure_fp16 requires model.torch_dtype=float16 and training.bf16=false, training.fp16=false")
    if cfg.model.use_peft or cfg.model.quantization or cfg.model.use_unsloth:
        raise ValueError("The 397B pure_fp16 recipe requires full-parameter training without PEFT/quantization/Unsloth")
    if not cfg.training.gradient_checkpointing or cfg.training.gradient_checkpointing_backend != "fsdp":
        raise ValueError("The 397B pure_fp16 recipe requires gradient_checkpointing=true with backend=fsdp")
    if cfg.training.vision_encoder_lr is not None:
        raise ValueError("The 397B pure_fp16 recipe does not support a separate vision_encoder_lr")


def create_pure_fp16_optimizer(model, args, decay_names):
    decay_names = set(decay_names)
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and (n in decay_names) == decay],
         "weight_decay": args.weight_decay if decay else 0.0}
        for decay in (True, False)
    ]
    # States follow parameter dtype. Non-foreach avoids parameter-sized tensor
    # lists as temporary optimizer storage. No master copy or AMP GradScaler.
    return torch.optim.AdamW(groups, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                             eps=args.adam_epsilon, foreach=False, fused=False)


class PureFP16Audit(TrainerCallback):
    """Fail if wrapping/resume silently changes storage or checkpoint coverage."""

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.audited_update = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if self.accelerator.mixed_precision != "no" or self.accelerator.scaler is not None:
            raise RuntimeError("Pure FP16 requires Accelerate mixed_precision='no' with no GradScaler")
        if plugin is None or plugin.fsdp_version != 1 or not plugin.activation_checkpointing:
            raise RuntimeError("Use the dedicated b200_397b_fp16_fsdp.yaml launcher for pure FP16 + FSDP checkpointing")
        self._check_params(model)
        names = {"Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"}
        blocks = {id(m): type(m).__name__ for m in model.modules() if type(m).__name__ in names}
        wrapped = {id(m._checkpoint_wrapped_module) for m in model.modules() if isinstance(m, CheckpointWrapper)}
        if not blocks or not set(blocks).issubset(wrapped):
            raise RuntimeError("Pure FP16 requires checkpoint wrappers around every MoE decoder and vision block")
        self.coverage = {name: list(blocks.values()).count(name) for name in names}

    @staticmethod
    def _check_params(model):
        for name, parameter in model.named_parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float16:
                raise RuntimeError(f"Pure FP16 parameter {name} unexpectedly has dtype {parameter.dtype}")

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.audited_update:
            return
        self._check_params(model)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and parameter.grad.dtype != torch.float16:
                    raise RuntimeError(f"Pure FP16 gradient unexpectedly has dtype {parameter.grad.dtype}")

    def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        if self.audited_update:
            return
        count = 0
        for values in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in values:
                    count += 1
                    if values[key].dtype != torch.float16:
                        raise RuntimeError(f"Pure FP16 AdamW {key} unexpectedly has dtype {values[key].dtype}")
        if not count:
            raise RuntimeError("No AdamW states found after the first pure FP16 update")
        report = dict(parameter_dtype="float16", gradient_dtype="float16", adamw_moment_dtype="float16",
                      mixed_precision="no", checkpointed_blocks=self.coverage,
                      adamw_moment_tensors=count, rank=self.accelerator.process_index)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / f"pure_fp16_rank_{self.accelerator.process_index}.json").write_text(json.dumps(report, indent=2))
        self.audited_update = True
