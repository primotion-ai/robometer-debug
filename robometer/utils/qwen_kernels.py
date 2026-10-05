"""Explicit, per-model kernel selection for Qwen3.5 dense and MoE backbones."""

import importlib
from types import MethodType

import torch
import torch.nn.functional as F
from loguru import logger


def _eager_experts_unbind_forward(self, hidden_states, top_k_index, top_k_weights):
    # Indexing the packed parameter inside each expert's graph causes one
    # packed-parameter-sized SelectBackward zero/scatter per expert. Unbind
    # once so autograd assembles the packed gradient once instead.
    gate_up_weights = self.gate_up_proj.unbind(0)
    down_weights = self.down_proj.unbind(0)
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
    for expert_idx in expert_hit:
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, gate_up_weights[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = F.linear(self.act_fn(gate) * up, down_weights[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


def optimize_qwen_moe_expert_backward(model):
    """Opt-in eager MoE backward optimization; preserves packed parameter keys."""
    count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen3_5MoeExperts":
            module.forward = MethodType(_eager_experts_unbind_forward, module)
            count += 1
    logger.info(f"Qwen eager expert backward: unbind packed weights once; {count} modules")
    return model


def _fp16_shared_expert_forward(self, hidden_states):
    # Real 397B weights can produce >280k before shared-expert gating, while
    # the gated result is only ~39. Keep this intermediate out of FP16.
    gate = self.gate_proj(hidden_states).float()
    up = self.up_proj(hidden_states).float()
    return F.linear(self.act_fn(gate) * up, self.down_proj.weight.float())


def _fp16_sparse_moe_forward(self, hidden_states):
    batch, length, width = hidden_states.shape
    inputs = hidden_states.reshape(-1, width)
    shared = self.shared_expert(inputs)
    _, routing_weights, selected_experts = self.gate(inputs)
    routed = self.experts(inputs, selected_experts, routing_weights)
    # Apply the small gate before casting the shared branch back to FP16.
    # FP32 sigmoid also keeps its backward multiplication out of FP16.
    gate = torch.sigmoid(self.shared_expert_gate(inputs).float())
    output = routed.float() + shared * gate
    return output.to(hidden_states.dtype).reshape(batch, length, width)


def stabilize_qwen_fp16_shared_experts(model):
    """Local FP32 intermediates only; parameter/state dtypes and keys are unchanged."""
    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3_5MoeSparseMoeBlock":
            continue
        if module.shared_expert.down_proj.weight.dtype != torch.float16:
            continue
        module.shared_expert.forward = MethodType(_fp16_shared_expert_forward, module.shared_expert)
        module.forward = MethodType(_fp16_sparse_moe_forward, module)
        count += 1
    if count:
        logger.info(f"Qwen FP16 shared-expert stabilization: {count} blocks; FP32 intermediates through gating")
    return model


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
