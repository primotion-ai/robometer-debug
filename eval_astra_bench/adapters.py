"""Lazy-loaded GPU adapters. They receive inputs only, never Astra target labels."""
from contextlib import nullcontext
import json
from pathlib import Path

from .core import REFERENCE_PROGRESS, uniform_indices, validate_progress


def build_icl_inputs(processor, task, references, queries, max_keyframes, max_pixels):
    import torch
    from qwen_vl_utils import process_vision_info
    from robometer.icl_progress.collator import build_conversation
    if max_keyframes < len(references):
        raise ValueError('Checkpoint max_keyframes is smaller than five')
    processor.tokenizer.padding_side = 'right'
    conversations = [build_conversation(task, references, q, max_pixels) for q in queries]
    texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in conversations]
    images, _ = process_vision_info(conversations, image_patch_size=processor.image_processor.patch_size)
    inputs = processor(text=texts, images=images, padding=True, truncation=False,
                       do_resize=False, return_tensors='pt')
    positions = torch.zeros(len(queries), max_keyframes, dtype=torch.long)
    mask = torch.zeros(len(queries), max_keyframes, dtype=torch.bool)
    end_token = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    for i in range(len(queries)):
        ends = torch.nonzero(inputs['input_ids'][i] == end_token, as_tuple=False).flatten()
        if len(ends) != len(references) + 1:
            raise ValueError('Expected five reference images followed by one query image')
        positions[i, :len(references)] = ends[:len(references)]
        mask[i, :len(references)] = True
    inputs.update(anchor_positions=positions, anchor_mask=mask,
                  query_positions=inputs['attention_mask'].sum(-1) - 1)
    return dict(inputs)


def load_icl(checkpoint, *, device_map='single', base_model=None):
    """Strict full checkpoints, including FSDP DCP and sharded safetensors.

    FP32 training checkpoints stay FP32 with BF16 autocast; 397B keeps FP16.
    Auto mapping spans visible GPUs on one host; it does not launch multi-node jobs.
    """
    import torch
    import psutil
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, Qwen3VLModel, Qwen3_5Model, Qwen3_5MoeModel
    from robometer.icl_progress.model import ICLProgressModel
    from robometer.utils.qwen_kernels import configure_qwen_linear_attention, stabilize_qwen_fp16_shared_experts

    checkpoint = Path(checkpoint).resolve()
    cfg = OmegaConf.load(checkpoint/'icl_config.yaml')
    state = json.loads((checkpoint/'icl_state.json').read_text())
    if cfg.model.name not in ('4b', '27b', '397b') or cfg.model.name != state['model']:
        raise ValueError('Expected matching ICL 4b/27b/397b checkpoint config and state')
    if cfg.model.precision not in ('bf16', 'pure_fp16'):
        raise ValueError(f'Unsupported precision: {cfg.model.precision}')
    if not torch.cuda.is_available():
        raise RuntimeError('ICL inference requires CUDA')
    dtype = torch.float16 if cfg.model.precision == 'pure_fp16' else torch.float32
    config = AutoConfig.from_pretrained(base_model or cfg.model.path, local_files_only=True)
    config._attn_implementation = cfg.model.attention
    config.use_cache = config.text_config.use_cache = False
    if config.model_type == 'qwen3_5_moe':
        config._experts_implementation = 'eager'
    classes = {'qwen3_vl': Qwen3VLModel, 'qwen3_5': Qwen3_5Model, 'qwen3_5_moe': Qwen3_5MoeModel}
    with init_empty_weights():
        model = ICLProgressModel(classes[config.model_type](config), cfg.model.head_dim).to(dtype=dtype)
    expected = {k:tuple(v.shape) for k,v in model.state_dict().items()}
    dcp_path = checkpoint/'pytorch_model_fsdp_0'
    shapes, shards, saved_dtypes = {}, {}, set()
    if dcp_path.is_dir():
        import torch.distributed.checkpoint as dcp
        reader = dcp.FileSystemReader(dcp_path)
        for key, value in reader.read_metadata().state_dict_metadata.items():
            if not key.startswith('model.') or not hasattr(value, 'size'):
                raise ValueError(f'Unexpected DCP entry: {key}')
            shapes[key[6:]] = tuple(value.size)
            saved_dtypes.add(value.properties.dtype)
    else:
        index = checkpoint/'model.safetensors.index.json'
        names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        for name in names:
            with safe_open(checkpoint/name, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key in shapes:
                        raise ValueError(f'Duplicate weight: {key}')
                    shapes[key] = tuple(f.get_slice(key).get_shape())
                    shards[key] = name
    if shapes != expected:
        missing, extra = set(expected)-set(shapes), set(shapes)-set(expected)
        bad = [k for k in expected.keys() & shapes.keys() if expected[k] != shapes[k]]
        raise ValueError(f'Checkpoint mismatch: missing={missing}, extra={extra}, shapes={bad}')
    if saved_dtypes and saved_dtypes != {dtype}:
        raise ValueError(f'Training precision mismatch: saved={saved_dtypes}, expected={dtype}')
    # Plan placement before allocating a potentially 800 GB state dict.
    if device_map == 'auto':
        memory = {i:max(0, torch.cuda.mem_get_info(i)[0]-8*2**30) for i in range(torch.cuda.device_count())}
        memory['cpu'] = 0
        placement = infer_auto_device_map(model, max_memory=memory, dtype=dtype,
            no_split_module_classes=['Qwen3_5MoeDecoderLayer', 'Qwen3_5DecoderLayer', 'Qwen3VLTextDecoderLayer',
                                     'Qwen3_5MoeVisionModel', 'Qwen3_5VisionModel', 'Qwen3VLVisionModel'])
        if any(str(v) in ('cpu', 'disk') for v in placement.values()):
            raise RuntimeError('Insufficient visible GPU memory. Use a host with enough GPUs; no CPU/disk offload is implicit.')
        if '' not in placement:
            for head in ('anchor_projection', 'query_projection', 'scorer'):
                for key in list(placement):
                    if key == head or key.startswith(head+'.'):
                        del placement[key]
                placement[head] = 0
    elif device_map == 'single':
        placement = {'': 0}
    else:
        raise ValueError('device_map must be single or auto')
    import math
    required = sum(math.prod(shape) for shape in shapes.values()) * torch.tensor([], dtype=dtype).element_size()
    if required > psutil.virtual_memory().available * .9:
        raise RuntimeError(f'Loading needs about {required/2**30:.1f} GiB CPU RAM; insufficient available RAM')
    if device_map == 'single' and required > torch.cuda.mem_get_info(0)[0]-4*2**30:
        raise RuntimeError('Checkpoint does not fit GPU 0 with activation reserve; use --device-map auto')
    print(f'Loading ICL {cfg.model.name}: {required/2**30:.1f} GiB weights, {dtype}', flush=True)
    if dcp_path.is_dir():
        weights = {k:torch.empty(shape, dtype=dtype) for k,shape in shapes.items()}
        dcp.load({'model':weights}, storage_reader=reader)
    else:
        weights = {}
        for name in sorted(set(shards.values())):
            with safe_open(checkpoint/name, framework='pt', device='cpu') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if tensor.dtype != dtype:
                        raise ValueError(f'Training precision mismatch: {key}: {tensor.dtype} != {dtype}')
                    weights[key] = tensor
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError('Unmaterialized tensors in checkpoint')
    if config.model_type in ('qwen3_5', 'qwen3_5_moe'):
        configure_qwen_linear_attention(model.backbone, cfg.model.linear_attention)
        stabilize_qwen_fp16_shared_experts(model.backbone)
    model.eval().requires_grad_(False)
    if device_map == 'auto':
        model = dispatch_model(model, device_map=placement, main_device=torch.device('cuda:0'))
        # ICL forward indexes returned hidden states with GPU-0 token positions.
        add_hook_to_module(model.backbone, AlignDevicesHook(execution_device='cuda:0', io_same_device=True), append=True)
    else:
        model.to('cuda:0')
    processor = AutoProcessor.from_pretrained(checkpoint/'processor', local_files_only=True)
    report = dict(checkpoint=str(checkpoint), model=cfg.model.name, parameter_dtype=str(dtype),
                  format='fsdp_dcp' if dcp_path.is_dir() else 'safetensors', tensors_loaded=len(shapes),
                  missing_keys=[], unexpected_keys=[], device_map=placement,
                  training_state=state, input_policy='five_reference_images_plus_independent_query')
    return model, processor, cfg, report


def read_images(root, paths):
    from PIL import Image
    images = []
    for path in paths:
        with Image.open(Path(root)/path) as image:
            images.append(image.convert('RGB'))
    return images


class ICLAdapter:
    def __init__(self, checkpoint, data_root, batch_size=1, device_map='single', base_model=None, **unused):
        self.model, self.processor, self.cfg, self.report = load_icl(checkpoint, device_map=device_map, base_model=base_model)
        self.root, self.batch_size = data_root, batch_size
        self.reference_cache = {}

    def predict(self, row):
        import torch
        ref = row['icl_reference']
        uid = ref['trajectory_id']
        if uid not in self.reference_cache:
            self.reference_cache[uid] = read_images(self.root, ref['image_paths'])
        references = self.reference_cache[uid]
        queries = read_images(self.root, row['image_paths'])
        result = []
        for start in range(0, len(queries), self.batch_size):
            inputs = build_icl_inputs(self.processor, row['instruction'], references,
                                      queries[start:start+self.batch_size], self.cfg.data.max_keyframes,
                                      self.cfg.data.max_pixels)
            inputs = {k:v.to('cuda:0') for k,v in inputs.items()}
            if self.cfg.model.precision == 'pure_fp16':
                inputs = {k:(v.half() if v.is_floating_point() else v) for k,v in inputs.items()}
            context = torch.autocast('cuda', dtype=torch.bfloat16) if self.cfg.model.precision == 'bf16' else nullcontext()
            with torch.inference_mode(), context:
                logits = self.model(**inputs)['logits'].float()
            if not torch.isfinite(logits[:, :5]).all():
                raise ValueError('Nonfinite ICL logits')
            probabilities = logits.softmax(-1)
            if not torch.allclose(probabilities[:, 5:], torch.zeros_like(probabilities[:, 5:])):
                raise ValueError('Nonzero padded anchor probability')
            values = (probabilities[:, :5] * torch.tensor(REFERENCE_PROGRESS, device=logits.device)).sum(-1)
            result.extend(values.cpu().tolist())
        validate_progress(result)
        return result


class OfficialAdapter:
    def __init__(self, checkpoint, data_root, batch_size=1, official_frames=32,
                 base_model=None, device_map='single', **unused):
        from robometer.evals.baselines.rbm_model import RBMModel
        self.model = RBMModel(str(checkpoint), strict_offline=True, base_model_path=base_model,
                              device_map='auto' if device_map == 'auto' else None)
        self.root, self.sample_frames = data_root, official_frames
        if official_frames not in (8, 16, 32):
            raise ValueError('official_frames must be 8, 16 or 32')
        self.report = dict(getattr(self.model.model, 'checkpoint_load_report', {}),
                           input_policy='whole_trajectory_per_frame_progress_tokens',
                           sampled_frames=official_frames,
                           interpolation='none' if official_frames == 32 else 'linear_in_stored_frame_index',
                           training_max_frames=self.model.exp_config.data.max_frames)

    def predict_many(self, rows):
        """One model forward for a batch of trajectories, keeping each frame mapping."""
        import numpy as np
        from robometer.data.dataset_types import Trajectory, ProgressSample
        if not rows:
            return []
        samples, mappings = [], []
        for row in rows:
            indices = uniform_indices(row['num_frames'], self.sample_frames)
            frames = np.stack([np.asarray(img) for img in read_images(
                self.root, [row['image_paths'][i] for i in indices])])
            samples.append(ProgressSample(trajectory=Trajectory(
                id=row['trajectory_id'], task=row['instruction'], frames=frames,
                frames_shape=frames.shape, metadata={'subsequence_length':len(frames)})))
            mappings.append(indices)
        outputs = self.model.compute_batched_progress(samples)
        if len(outputs) != len(rows):
            raise ValueError('Official model returned incorrect trajectory batch size')
        results = []
        for row, indices, output in zip(rows, mappings, outputs):
            if len(output) != len(indices):
                raise ValueError('Official whole-trajectory inference must return one progress value per sampled frame')
            sparse = [float(x) for x in output]
            validate_progress(sparse, len(indices))
            result = np.interp(np.arange(row['num_frames']), indices, sparse).tolist()
            validate_progress(result)
            details = dict(sampled_frame_indices=indices, sampled_progress=sparse,
                           interpolation=self.report['interpolation'])
            results.append((result, details))
        return results

    def predict(self, row):
        result, self.last_details = self.predict_many([row])[0]
        return result
