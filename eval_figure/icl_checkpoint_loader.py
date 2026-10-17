"""Strict local ICL checkpoint loading; preserve FP32 training weights for BF16 eval."""
import torch


def load_icl_checkpoint(config, checkpoint):
    from safetensors.torch import load_file
    from robometer.icl_progress.model import load_model, ICLProgressModel
    dcp_path = checkpoint/'pytorch_model_fsdp_0'
    config.model.gradient_checkpointing = False
    if dcp_path.is_dir():
        from accelerate import init_empty_weights
        from transformers import AutoConfig, Qwen3_5Model
        import torch.distributed.checkpoint as dcp
        from robometer.utils.qwen_kernels import configure_qwen_linear_attention
        if config.model.name != '27b':
            raise ValueError('DCP inference currently supports the ICL 27B architecture')
        backbone_config = AutoConfig.from_pretrained(config.model.path, local_files_only=True)
        if backbone_config.model_type != 'qwen3_5':
            raise ValueError(f'Unexpected architecture: {backbone_config.model_type}')
        backbone_config._attn_implementation = config.model.attention
        backbone_config.use_cache = backbone_config.text_config.use_cache = False
        with init_empty_weights():
            model = ICLProgressModel(Qwen3_5Model(backbone_config), config.model.head_dim).float()
        reader = dcp.FileSystemReader(dcp_path)
        metadata = reader.read_metadata()
        shapes = {}
        for key, value in metadata.state_dict_metadata.items():
            if not key.startswith('model.') or not hasattr(value, 'size'):
                raise ValueError(f'Unexpected DCP entry: {key}')
            if value.properties.dtype != torch.float32:
                raise ValueError(f'Unexpected saved dtype: {key}: {value.properties.dtype}')
            shapes[key[len('model.'):]] = tuple(value.size)
        expected = model.state_dict()
        if shapes.keys() != expected.keys():
            raise ValueError(f'Checkpoint keys mismatch: missing={expected.keys()-shapes.keys()}, extra={shapes.keys()-expected.keys()}')
        for key, shape in shapes.items():
            if tuple(expected[key].shape) != shape:
                raise ValueError(f'Checkpoint shape mismatch: {key}')
        del expected
        weights = {key:torch.empty(shape,dtype=torch.float32) for key,shape in shapes.items()}
        print(f'Loading {len(weights)} DCP tensors ({sum(t.numel()*t.element_size() for t in weights.values())/2**30:.1f} GiB FP32)',flush=True)
        dcp.load({'model':weights},storage_reader=reader)
        loaded = model.load_state_dict(weights, strict=True, assign=True)
        count = len(weights)
        del weights
        configure_qwen_linear_attention(model.backbone, config.model.linear_attention)
        checkpoint_format = 'fsdp_dcp'
    else:
        model, _ = load_model(config.model)
        weights = load_file(str(checkpoint/'model.safetensors'), device='cpu')
        loaded = model.load_state_dict(weights, strict=True)
        count = len(weights)
        del weights
        checkpoint_format = 'safetensors'
    if any(t.is_meta for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('Unmaterialized model tensors after loading')
    model.eval().requires_grad_(False).to('cuda:0')
    report = dict(checkpoint=str(checkpoint.resolve()), format=checkpoint_format, tensors_loaded=count,
                  missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys,
                  parameter_dtype=str(next(model.parameters()).dtype), compute_autocast='bfloat16')
    return model, report
