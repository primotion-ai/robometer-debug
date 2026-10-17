import torch
from torch import nn
import torch.nn.functional as F


def progress_loss(logits, labels, anchor_mask):
    # Finite padding logit avoids 0 * -inf in soft-target CE; its probability
    # underflows to zero. Mask before softmax, not after normalization.
    masked = logits.float().masked_fill(~anchor_mask, torch.finfo(torch.float32).min)
    loss = -(labels.float() * F.log_softmax(masked, dim=-1)).sum(-1).mean()
    return loss, masked


class ICLProgressModel(nn.Module):
    def __init__(self, backbone, head_dim=256):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.text_config.hidden_size
        self.anchor_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, head_dim), nn.GELU())
        self.query_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, head_dim), nn.GELU())
        # The same score function is applied to every real anchor, for any K.
        self.scorer = nn.Sequential(nn.Linear(head_dim * 3 + 1, head_dim), nn.GELU(), nn.Linear(head_dim, 1))
        dtype = next(backbone.parameters()).dtype
        self.anchor_projection.to(dtype=dtype)
        self.query_projection.to(dtype=dtype)
        self.scorer.to(dtype=dtype)

    def forward(self, anchor_positions, query_positions, anchor_mask, labels=None, **inputs):
        outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        anchors = hidden[batch_index[:, None], anchor_positions]
        query = hidden[batch_index, query_positions]
        a = self.anchor_projection(anchors)
        q = self.query_projection(query)[:, None, :].expand_as(a)
        positions = torch.arange(a.shape[1], device=a.device)[None, :].expand(a.shape[:2])
        scale = positions / (anchor_mask.sum(-1, keepdim=True) - 1).clamp_min(1)
        features = torch.cat([a, q, a * q, scale[..., None].to(a.dtype)], dim=-1)
        logits = self.scorer(features).squeeze(-1)
        if labels is None:
            labels = torch.zeros_like(logits)
        loss, logits = progress_loss(logits, labels, anchor_mask)
        return {'loss': loss, 'logits': logits}


def load_model(cfg):
    from types import SimpleNamespace
    from transformers import AutoProcessor
    from robometer.utils.setup_utils import _load_base_model_standard

    dtype = torch.float16 if cfg.precision == 'pure_fp16' else torch.float32
    # Reuse tested Qwen3.5 weight key mapping, rank-0/meta loading, FLA and
    # 397B eager-expert/shared-expert fixes, but never instantiate RBM heads.
    model_cfg = SimpleNamespace(base_model_id=cfg.path, trust_remote_code=False,
                                linear_attention_backend=cfg.linear_attention,
                                experts_implementation='eager' if cfg.name == '397b' else None)
    backbone = _load_base_model_standard(model_cfg, dtype,
                                    {'attn_implementation': cfg.attention, 'local_files_only': True}, None)
    backbone.config.use_cache = False
    backbone.config.text_config.use_cache = False
    if cfg.freeze_vision:
        backbone.visual.requires_grad_(False)
    if cfg.gradient_checkpointing and cfg.precision != 'pure_fp16':
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = ICLProgressModel(backbone, cfg.head_dim)
    processor = AutoProcessor.from_pretrained(cfg.path, local_files_only=True)
    return model, processor
