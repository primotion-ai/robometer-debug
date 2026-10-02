"""Explicit, per-model kernel selection for Qwen3.5 dense and MoE backbones."""

import importlib
from loguru import logger


def configure_qwen_linear_attention(model, backend="auto"):
    """Select reference or fused kernels without changing process-wide globals."""
    if backend not in {"auto", "torch", "fla"}:
        raise ValueError(f"Unknown linear_attention_backend: {backend}")
    count = 0
    for module in model.modules():
        if type(module).__name__ not in {"Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet"}:
            continue
        implementation = importlib.import_module(type(module).__module__)
        if backend == "fla":
            if not implementation.is_fast_path_available:
                raise RuntimeError(
                    "linear_attention_backend=fla requires working fla-core and causal-conv1d kernels. "
                    "Install the b200-kernels extra; use model.linear_attention_backend=torch for a reference run."
                )
            module.causal_conv1d_fn = implementation.causal_conv1d_fn
            module.causal_conv1d_update = implementation.causal_conv1d_update
            module.chunk_gated_delta_rule = implementation.chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = implementation.fused_recurrent_gated_delta_rule
        elif backend == "torch":
            module.causal_conv1d_fn = None
            module.causal_conv1d_update = implementation.torch_causal_conv1d_update
            module.chunk_gated_delta_rule = implementation.torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = implementation.torch_recurrent_gated_delta_rule
        if backend != "auto":
            old_norm = module.norm
            if backend == "torch":
                norm_cls = getattr(implementation, type(module).__name__.replace("GatedDeltaNet", "RMSNormGated"))
                module.norm = norm_cls(module.head_v_dim, eps=module.layer_norm_epsilon)
            else:
                module.norm = implementation.FusedRMSNormGated(
                    module.head_v_dim, eps=module.layer_norm_epsilon, activation=module.activation,
                    device=old_norm.weight.device, dtype=old_norm.weight.dtype,
                )
            # Preserve the loaded parameter, including meta tensors on nonzero ranks.
            module.norm.weight = old_norm.weight
            module.norm.train(old_norm.training)
        count += 1
    if backend != "auto" and not count:
        raise ValueError(f"No supported Qwen GatedDeltaNet layers found for backend={backend}")
    logger.info(f"Qwen linear attention: backend={backend}, layers={count}")
    return model
