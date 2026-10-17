"""Strict offline inference loading, including FSDP DCP saved with any world size."""
import json
from pathlib import Path

import torch
import yaml
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, Qwen3VLModel, Qwen3_5Model, Qwen3_5MoeModel

from robometer.configs.experiment_configs import ExperimentConfig
from robometer.models.rbm import RBM
from robometer.utils.setup_utils import _setup_processor_and_tokenizer, _add_special_tokens_and_resize
from robometer.utils.qwen_kernels import configure_qwen_linear_attention


def validate_shapes(expected, shapes):
    missing, extra = set(expected) - set(shapes), set(shapes) - set(expected)
    mismatched = [k for k in set(expected) & set(shapes) if tuple(expected[k].shape) != tuple(shapes[k])]
    if missing or extra or mismatched:
        raise ValueError(f'Checkpoint mismatch: missing={sorted(missing)}, extra={sorted(extra)}, shapes={mismatched}')


def dispatch_offline_model(model, device, placement):
    from accelerate import dispatch_model
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    model = dispatch_model(model, device_map=placement, main_device=device)
    # RBM extracts token states using input_ids on the main GPU before it
    # calls its heads. Return the backbone outputs to that same GPU.
    add_hook_to_module(model.model, AlignDevicesHook(execution_device=device, io_same_device=True), append=True)
    return model


def load_offline_checkpoint(checkpoint_path, device, base_model_path=None, device_map=None):
    if device_map not in (None, 'auto'):
        raise ValueError('Offline device_map must be null or auto')
    root = Path(checkpoint_path).resolve()
    config_path = root / 'config.yaml'
    if not config_path.exists():
        config_path = root.parent / 'config.yaml'
    exp = ExperimentConfig(**yaml.safe_load(config_path.read_text()))
    if exp.model.use_peft:
        raise ValueError('Strict offline loader requires a full model checkpoint, not PEFT adapters')
    if base_model_path:
        exp.model.base_model_id = base_model_path
    exp.model.use_unsloth = False
    exp.model.attn_implementation = 'sdpa'
    dtype = torch.bfloat16
    dcp_path = root / 'pytorch_model_fsdp_0'
    shapes, shards = {}, {}
    if dcp_path.is_dir():
        from torch.distributed.checkpoint import FileSystemReader
        reader = FileSystemReader(dcp_path)
        metadata = reader.read_metadata()
        for key, value in metadata.state_dict_metadata.items():
            if not key.startswith('model.') or not hasattr(value, 'size'):
                raise ValueError(f'Unexpected DCP entry: {key}')
            shapes[key[len('model.'):]] = tuple(value.size)
    else:
        index = root / 'model.safetensors.index.json'
        names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        for name in names:
            with safe_open(root / name, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key in shapes:
                        raise ValueError(f'Duplicate checkpoint tensor: {key}')
                    shapes[key] = tuple(f.get_slice(key).get_shape())
                    shards[key] = name
    config = AutoConfig.from_pretrained(root if (root / 'config.json').exists() else exp.model.base_model_id,
                                       local_files_only=True)
    config.text_config.vocab_size = shapes['model.language_model.embed_tokens.weight'][0]
    config.text_config.use_cache = False
    config._attn_implementation = 'sdpa'
    if config.model_type == 'qwen3_5_moe' and exp.model.experts_implementation is not None:
        config._experts_implementation = exp.model.experts_implementation
    cls = {'qwen3_vl': Qwen3VLModel, 'qwen3_5': Qwen3_5Model, 'qwen3_5_moe': Qwen3_5MoeModel}[config.model_type]
    processor = _setup_processor_and_tokenizer(exp.model)
    with init_empty_weights():
        backbone = cls(config)
        _add_special_tokens_and_resize(exp.model, processor, backbone)
        if backbone.get_input_embeddings().num_embeddings != shapes['model.language_model.embed_tokens.weight'][0]:
            raise ValueError('Checkpoint vocabulary cannot represent required reward tokens')
        model = RBM(config, processor, processor.tokenizer, base_model=backbone,
                    base_model_id=exp.model.base_model_id, model_config=exp.model).to(dtype=dtype)
    # The published RFM checkpoint includes a retired similarity head; RBM's
    # progress/preference/success evaluation never calls it. No other extras allowed.
    unused = sorted(k for k in shapes if k.startswith('similarity_head.') and k not in model.state_dict())
    for key in unused:
        shapes.pop(key)
    validate_shapes(model.state_dict(), shapes)
    if dcp_path.is_dir():
        import torch.distributed.checkpoint as dcp
        weights = {k: torch.empty(shape, dtype=dtype) for k, shape in shapes.items()}
        print(f'Loading {len(weights)} DCP tensors into CPU memory ({sum(w.numel()*w.element_size() for w in weights.values())/2**30:.1f} GiB)', flush=True)
        dcp.load({'model': weights}, storage_reader=reader)
    else:
        weights = {}
        for name in sorted(set(shards.values())):
            with safe_open(root / name, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key not in unused:
                        weights[key] = f.get_tensor(key).to(dtype=dtype)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    if config.model_type in {'qwen3_5', 'qwen3_5_moe'}:
        configure_qwen_linear_attention(model.model, exp.model.linear_attention_backend)
    model.requires_grad_(False)
    model.eval()
    if device_map == 'auto':
        from accelerate import infer_auto_device_map
        if not torch.cuda.is_available():
            raise ValueError('Automatic offline device mapping requires CUDA')
        # Leave room for activations, attention workspaces and input/output tensors.
        max_memory = {i: max(0, torch.cuda.mem_get_info(i)[0] - 24 * 2**30)
                      for i in range(torch.cuda.device_count())}
        placement = infer_auto_device_map(
            model, max_memory=max_memory, dtype=dtype,
            no_split_module_classes=['Qwen3_5MoeDecoderLayer', 'Qwen3_5DecoderLayer',
                                     'Qwen3VLTextDecoderLayer', 'Qwen3_5MoeVisionModel',
                                     'Qwen3_5VisionModel', 'Qwen3VLVisionModel'])
        if any(value in ('cpu', 'disk') for value in placement.values()):
            raise RuntimeError(f'Insufficient GPU memory for full-resident inference: {placement}')
        print(f'Dispatching model across GPUs: {placement}', flush=True)
        model = dispatch_offline_model(model, device, placement)
    else:
        placement = {'': str(device)}
        model.to(device)
    model.model_dtype = dtype
    model.checkpoint_load_report = {
        'checkpoint': str(root), 'format': 'fsdp_dcp' if dcp_path.is_dir() else 'safetensors',
        'tensors_loaded': len(shapes), 'missing_keys': [], 'unexpected_keys': [],
        'unused_legacy_similarity_head': unused,
        'dtype': str(dtype), 'base_model_path': exp.model.base_model_id,
        'device_map': placement,
        'special_token_ids': {t: processor.tokenizer.convert_tokens_to_ids(t)
                              for t in ['<|pref_token|>', '<|prog_token|>']},
    }
    return exp, processor.tokenizer, processor, model
