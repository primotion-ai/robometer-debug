"""Opt-in module diagnostics and small output checks; never repair invalid values."""

import json
import math
from pathlib import Path

import torch


def _tensors(value, path):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _tensors(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _tensors(child, f"{path}[{index}]")


def check_finite_outputs(value, *, location, output_dir, rank, step=None):
    """Fail before losses/metrics obscure an already-invalid model output."""
    for path, tensor in _tensors(value, location):
        if not tensor.is_floating_point() or tensor.numel() == 0:
            continue
        tensor = tensor.detach()
        if torch.isfinite(tensor).all():
            continue
        report = dict(rank=rank, step=step, location=path, dtype=str(tensor.dtype),
                      shape=list(tensor.shape), nan_count=torch.isnan(tensor).sum().item(),
                      inf_count=torch.isinf(tensor).sum().item())
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / f"first_nonfinite_rank_{rank}.json"
        report_path.write_text(json.dumps(report, indent=2))
        raise FloatingPointError(
            f"Non-finite model output at {path}: rank={rank}, step={step}, "
            f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
            f"NaN={report['nan_count']}, Inf={report['inf_count']}. "
            f"Report: {report_path}. No values were replaced. "
            "Use ROBOMETER_CHECK_NUMERICS=1 to locate the first failing module."
        )


def install_finite_output_hooks(model, *, output_dir, rank):
    """Slow debug-only checks, including block residuals and functional MoE ops."""
    handles = []
    containers = {"Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock",
                  "Qwen3_5MoeGatedDeltaNet", "Qwen3_5MoeSparseMoeBlock"}

    def make_hook(name):
        def hook(module, inputs, output):
            check_finite_outputs(output, location=name, output_dir=output_dir, rank=rank)
        return hook

    for name, module in model.named_modules():
        if not any(module.children()) or type(module).__name__ in containers:
            handles.append(module.register_forward_hook(make_hook(name or type(module).__name__)))
    return handles


def check_finite_grad_norm(grad_norm, *, output_dir, rank, completed_steps):
    """Stop before AdamW can consume invalid gradients; do not skip an update.

    The caller uses the already-computed FSDP global norm, so all ranks see the
    same failure without scanning the 397B parameters a second time.
    """
    value = float(grad_norm)
    if math.isfinite(value):
        return
    report = dict(rank=rank, completed_steps=completed_steps,
                  attempted_update=completed_steps + 1, phase="before_optimizer_step",
                  grad_norm=str(value), optimizer_step_executed=False)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / f"nonfinite_gradient_rank_{rank}.json"
    report_path.write_text(json.dumps(report, indent=2))
    raise FloatingPointError(
        f"Non-finite global gradient norm ({value}) before optimizer update "
        f"{completed_steps + 1}: rank={rank}, completed_steps={completed_steps}. "
        f"AdamW was not executed for this update. Report: {report_path}. "
        "Gradients were not replaced and the update was not silently skipped."
    )
