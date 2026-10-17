#!/usr/bin/env python3
"""Evaluate ICL progress for six drawer trajectories with fixed reference images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE_ID = '1d035ec5-5127-4af9-9cca-fe214830d670'
KEYFRAMES = [0, 17, 22, 25, 31]
ANCHORS = [0.0, 0.25, 0.5, 0.75, 1.0]
SUBSET = 'aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking'
TASK = 'Close the drawer'


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def build_inputs(processor, references, queries, max_keyframes, max_pixels):
    """Same tokenization/positions as training, allowing cross-trajectory queries."""
    import torch
    from qwen_vl_utils import process_vision_info
    from robometer.icl_progress.collator import build_conversation
    processor.tokenizer.padding_side = 'right'
    conversations = [build_conversation(TASK, references, query, max_pixels) for query in queries]
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
            raise ValueError('Expected five reference images and one query image')
        positions[i, :len(references)] = ends[:len(references)]
        mask[i, :len(references)] = True
    inputs.update(anchor_positions=positions, anchor_mask=mask,
                  query_positions=inputs['attention_mask'].sum(-1) - 1)
    return dict(inputs)


def read_data(root):
    import numpy as np
    from datasets import load_from_disk
    rows = [dict(r) for r in load_from_disk(str(root/SUBSET/'processed_dataset')) if r['task'] == TASK]
    if len(rows) != 6 or len({r['id'] for r in rows}) != 6:
        raise ValueError('Expected six unique Close the drawer trajectories')
    arrays = {}
    for row in rows:
        path = root/SUBSET/'frames'/Path(row['frames']).name
        with np.load(path, allow_pickle=False) as cached:
            frames = cached['frames'].copy()
        if frames.shape != (32, 240, 320, 3) or frames.dtype != np.uint8:
            raise ValueError(f'Unexpected frames: {path}')
        arrays[row['id']] = frames
        row['resolved_frames'] = str(path)
    if REFERENCE_ID not in arrays:
        raise ValueError('Reference trajectory missing')
    return rows, arrays


def render(result, arrays, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 5, height_ratios=[3.7, 1.1])
    ax = fig.add_subplot(grid[0, :])
    colors = ['#2563eb', '#0891b2', '#d97706', '#9333ea', '#16a34a', '#dc2626']
    ordered = sorted(result['trajectories'], key=lambda r: (r['id'] != REFERENCE_ID, r['quality_label'], r['id']))
    for row, color in zip(ordered, colors):
        is_reference = row['id'] == REFERENCE_ID
        label = row['quality_label'] + (' [reference trajectory]' if is_reference else '') + '\n' + row['id']
        ax.plot(row['frame_indices'], row['progress'], color=color, marker='o', markersize=3,
                linestyle='--' if is_reference else '-', linewidth=2.5 if is_reference else 1.8, label=label)
    reference = next(row for row in result['trajectories'] if row['id'] == REFERENCE_ID)
    reference_values = dict(zip(reference['frame_indices'], reference['progress']))
    keyframe_values = [reference_values[index] for index in KEYFRAMES]
    ax.scatter(KEYFRAMES, keyframe_values, marker='D', s=90, facecolors='#facc15',
               edgecolors='#2563eb', linewidths=1.7, zorder=6,
               label='Reference keyframes (on predicted curve)')
    for index, value in zip(KEYFRAMES, keyframe_values):
        ax.axvline(index, color='#2563eb', linestyle=':', linewidth=.8, alpha=.22, zorder=0)
        dx, align = (8, 'left') if index == 0 else ((-8, 'right') if index == 31 else (0, 'center'))
        ax.annotate(f'k={index}', (index, value), xytext=(dx, 12), textcoords='offset points',
                    ha=align, va='bottom', fontsize=9, color='#1d4ed8',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.5), zorder=7)
    ax.set(xlim=(-.6, 31.6), ylim=(-.025, 1.025), xlabel='Query frame index (zero-based)',
           ylabel='Probability-weighted progress')
    ax.set_xticks([0, 4, 8, 12, 17, 22, 25, 28, 31])
    ax.set_yticks(ANCHORS, ['0%', '25%', '50%', '75%', '100%'])
    ax.grid(alpha=.22)
    ax.legend(loc='upper left', bbox_to_anchor=(1.015, 1), fontsize=8.5, labelspacing=1.5, frameon=False)
    ax.set_title('Each query: the same 5 reference images + one current image; no query history', fontsize=11, pad=12)
    for i, (frame_index, progress) in enumerate(zip(KEYFRAMES, ANCHORS)):
        panel = fig.add_subplot(grid[1, i])
        panel.imshow(arrays[REFERENCE_ID][frame_index])
        panel.set_title(f'Reference index {frame_index}\nAnchor: {progress:.0%}', fontsize=10)
        panel.axis('off')
    step = result['training_state']['updates']
    model_label = result['training_state'].get('model', '4b').upper()
    fig.suptitle(f'ICL {model_label} / step {step} — Close the drawer\n6 trajectories, 32 independent frame queries each', fontsize=15)
    fig.text(.08, .06, 'Fixed reference: ' + REFERENCE_ID, fontsize=10)
    fig.text(.08, .028, 'Progress = Σ probability × [0%, 25%, 50%, 75%, 100%]. Raw predictions; no smoothing or forced endpoints.', fontsize=10)
    fig.subplots_adjust(left=.075, right=.725, top=.86, bottom=.115, hspace=.48, wspace=.12)
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'close_drawer_icl_progress.{extension}', dpi=180)
    plt.close(fig)
    with (output/'predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trajectory_id', 'quality_label', 'is_reference_trajectory', 'frame_index', 'progress',
                         'prob_0', 'prob_25', 'prob_50', 'prob_75', 'prob_100'])
        for row in result['trajectories']:
            writer.writerows([row['id'], row['quality_label'], row['id'] == REFERENCE_ID, index, value, *probs]
                for index, value, probs in zip(row['frame_indices'], row['progress'], row['probabilities']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'logs/icl_4b_001/checkpoint-15000')
    parser.add_argument('--dataset-root', type=Path, default=Path('/mnt/dataset/robometer-1m'))
    parser.add_argument('--output', type=Path, default=ROOT/'eval_figure/output/icl_4b_step15000_close_drawer')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, arrays = read_data(args.dataset_root)
    prediction_path = args.output/'predictions.json'
    if args.plot_only:
        result = json.loads(prediction_path.read_text())
        for uid, frames in arrays.items():
            if result['frame_sha256'][uid] != hashlib.sha256(frames.tobytes()).hexdigest():
                raise ValueError('Stored frames changed since inference')
        render(result, arrays, args.output)
        print('Replotted cached predictions; no model loaded', flush=True)
        return
    if prediction_path.exists():
        raise FileExistsError('Predictions already exist. Use --plot-only or a new --output directory.')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from eval_figure.icl_checkpoint_loader import load_icl_checkpoint
    from robometer.icl_progress.collator import ICLCollator
    from robometer.icl_progress.targets import FrameSelection, soft_target

    config = OmegaConf.load(args.checkpoint/'icl_config.yaml')
    state = json.loads((args.checkpoint/'icl_state.json').read_text())
    if config.model.name not in ('4b', '27b') or config.model.precision != 'bf16':
        raise ValueError('This entry point supports trained BF16 ICL 4B and 27B models')
    if state['model'] != config.model.name:
        raise ValueError('Checkpoint config/state model mismatch')
    torch.manual_seed(42)
    model, load_report = load_icl_checkpoint(config, args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint/'processor', local_files_only=True)
    references = [ICLCollator._pil(arrays[REFERENCE_ID][i]) for i in KEYFRAMES]

    # Verify exact training-format equivalence with a valid same-trajectory query.
    reference_row = next(r for r in rows if r['id'] == REFERENCE_ID)
    selected = FrameSelection(tuple(KEYFRAMES), 1, soft_target(KEYFRAMES, 1))
    training_row = dict(reference_row, frames=reference_row['resolved_frames'])
    expected = ICLCollator(processor, args.dataset_root, config.data.max_keyframes, config.data.max_pixels)([
        {'samples': [{'trajectory': training_row, 'selection': selected}]}])
    actual = build_inputs(processor, references, [ICLCollator._pil(arrays[REFERENCE_ID][1])],
                          config.data.max_keyframes, config.data.max_pixels)
    if set(expected) - {'labels'} != set(actual):
        raise AssertionError('Training/eval input fields differ')
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    load_report['collator_exact_training_equivalence'] = True
    write_json(args.output/'checkpoint_load_report.json', load_report)
    print(f"Loaded {load_report['tensors_loaded']} checkpoint tensors; training-format equivalence passed", flush=True)
    jobs = [(row['id'], index) for row in rows for index in range(32)]
    collected = {r['id']: dict(id=r['id'], task=r['task'], quality_label=r['quality_label'],
        frame_indices=[], progress=[], probabilities=[], logits=[]) for r in rows}
    batch_check = None
    for offset in range(0, len(jobs), args.batch_size):
        batch = jobs[offset:offset+args.batch_size]
        inputs = build_inputs(processor, references, [ICLCollator._pil(arrays[uid][index]) for uid,index in batch],
                              config.data.max_keyframes, config.data.max_pixels)
        inputs = {k: v.to('cuda:0') for k,v in inputs.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(**inputs)['logits']
        probs = logits.float().softmax(-1)
        if not torch.isfinite(logits[:, :5]).all() or not torch.isfinite(probs).all():
            raise ValueError('Nonfinite output')
        torch.testing.assert_close(probs[:,5:], torch.zeros_like(probs[:,5:]), rtol=0, atol=0)
        torch.testing.assert_close(probs[:,:5].sum(-1), torch.ones(len(batch),device=probs.device), rtol=1e-6, atol=1e-6)
        if offset == 0:
            uid, index = batch[0]
            single = build_inputs(processor, references, [ICLCollator._pil(arrays[uid][index])],
                                  config.data.max_keyframes, config.data.max_pixels)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                one = model(**{k:v.to('cuda:0') for k,v in single.items()})['logits'].float().softmax(-1)
            error = float((one[0,:5]-probs[0,:5]).abs().max())
            if error > .02:
                raise ValueError(f'Unexpected batch vs single-query discrepancy: {error}')
            batch_check = dict(trajectory_id=uid, frame_index=index, max_probability_difference=error,
                               tolerance=.02, passed=True)
        values = (probs[:,:5] * torch.tensor(ANCHORS, device=probs.device)).sum(-1)
        for i, (uid,index) in enumerate(batch):
            row = collected[uid]
            row['frame_indices'].append(index)
            row['progress'].append(float(values[i]))
            row['probabilities'].append(probs[i,:5].cpu().tolist())
            row['logits'].append(logits[i,:5].float().cpu().tolist())
        print(f'PROGRESS {offset+len(batch)}/{len(jobs)} independent frame queries', flush=True)
    result = dict(checkpoint=str(args.checkpoint.resolve()), training_state=state, dataset=SUBSET, task=TASK,
        reference_id=REFERENCE_ID, reference_frame_indices=KEYFRAMES, anchor_progress=ANCHORS,
        index_base=0, batch_size=args.batch_size, query_history=False, probabilities='softmax over real anchors only',
        reference_trajectory_included=True, batch_vs_single_check=batch_check,
        frame_sha256={uid:hashlib.sha256(frames.tobytes()).hexdigest() for uid,frames in arrays.items()},
        trajectories=list(collected.values()))
    write_json(prediction_path, result)
    render(result, arrays, args.output)
    write_json(args.output/'COMPLETE.json', dict(trajectories=6, frame_queries=192, reference_frames=KEYFRAMES,
        progress_formula='sum(p_i * [0, 0.25, 0.5, 0.75, 1]_i)',
        finite_predictions=True, probability_sums_verified=True, training_format_verified=True,
        batch_vs_single_check=batch_check))
    print(f'COMPLETE: {args.output}', flush=True)


if __name__ == '__main__':
    main()
