"""Inference on preloaded RGB images using the existing model/prompt helpers.

The original CLI/adapters are frozen while the 20261018 queue is running. This
entry adds data loading only; regression tests compare it against those adapters.
"""
from contextlib import nullcontext

from .adapters import build_icl_inputs
from .core import REFERENCE_PROGRESS, validate_progress


def predict_icl(adapter, loaded):
    import torch
    result = []
    for start in range(0, len(loaded.images), adapter.batch_size):
        inputs = build_icl_inputs(adapter.processor, loaded.row['instruction'], loaded.references,
                                  loaded.images[start:start+adapter.batch_size],
                                  adapter.cfg.data.max_keyframes, adapter.cfg.data.max_pixels)
        inputs = {k:v.to('cuda:0') for k,v in inputs.items()}
        if adapter.cfg.model.precision == 'pure_fp16':
            inputs = {k:(v.half() if v.is_floating_point() else v) for k,v in inputs.items()}
        context = torch.autocast('cuda', dtype=torch.bfloat16) if adapter.cfg.model.precision == 'bf16' else nullcontext()
        with torch.inference_mode(), context:
            logits = adapter.model(**inputs)['logits'].float()
        if not torch.isfinite(logits[:, :5]).all():
            raise ValueError('Nonfinite ICL logits')
        probabilities = logits.softmax(-1)
        if not torch.allclose(probabilities[:, 5:], torch.zeros_like(probabilities[:, 5:])):
            raise ValueError('Nonzero padded anchor probability')
        values = (probabilities[:, :5] * torch.tensor(REFERENCE_PROGRESS, device=logits.device)).sum(-1)
        result.extend(values.cpu().tolist())
    validate_progress(result)
    return result


def predict_official(adapter, batch):
    import numpy as np
    from robometer.data.dataset_types import Trajectory, ProgressSample
    if not batch:
        return []
    samples = []
    for loaded in batch:
        row = loaded.row
        frames = np.stack([np.asarray(img) for img in loaded.images])
        samples.append(ProgressSample(trajectory=Trajectory(
            id=row['trajectory_id'], task=row['instruction'], frames=frames,
            frames_shape=frames.shape, metadata={'subsequence_length':len(frames)})))
    outputs = adapter.model.compute_batched_progress(samples)
    if len(outputs) != len(batch):
        raise ValueError('Official model returned incorrect trajectory batch size')
    results = []
    for loaded, output in zip(batch, outputs):
        if len(output) != len(loaded.indices):
            raise ValueError('Official whole-trajectory inference must return one progress value per sampled frame')
        sparse = [float(x) for x in output]
        validate_progress(sparse, len(loaded.indices))
        result = np.interp(np.arange(loaded.row['num_frames']), loaded.indices, sparse).tolist()
        validate_progress(result)
        details = dict(sampled_frame_indices=loaded.indices, sampled_progress=sparse,
                       interpolation=adapter.report['interpolation'])
        results.append((result, details))
    return results
