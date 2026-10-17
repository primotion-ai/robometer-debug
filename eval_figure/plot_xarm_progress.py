#!/usr/bin/env python3
"""Sequential offline inference and two-panel progress plots for USC xArm."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = Path('/mnt/dataset/robometer-1m/aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking')
MODELS = [
    dict(name='official4b', label='Official Robometer-4B', checkpoint='/mnt/base_model/Robometer-4B', base='/mnt/base_model/Qwen3-VL-4B-Instruct', gpus='0'),
    dict(name='qwen27b_step3000', label='27B / step 3000', checkpoint=str(ROOT/'logs/rbm_qwen27b_5nodes_003/checkpoint-3000'), base='/mnt/base_model/Qwen3.8-27B', gpus='0'),
    dict(name='trained4b_step12000', label='Trained 4B / step 12000', checkpoint=str(ROOT/'logs/rbm_qwen4b_bs32_8gpu_002/checkpoint-12000'), base='/mnt/base_model/Qwen3-VL-4B-Instruct', gpus='0'),
    dict(name='qwen397b_step1000', label='397B / step 1000', checkpoint=str(ROOT/'logs/rbm_qwen397b_5nodes_2000steps_003/checkpoint-1000'), base='/mnt/base_model/Qwen3.5-397B-A17B', gpus='0,1,2,3,4,5,6,7', device_map='auto'),
]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sample_indices():
    import numpy as np
    return [(f'segment_{i+1}', list(range(8*i, 8*(i+1)))) for i in range(4)] + [
        ('global', np.rint(np.linspace(0, 31, 8)).astype(int).tolist())]


def plot_record(record, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    colors = ['#2563eb', '#d97706', '#059669', '#9333ea']
    for segment, color in zip(record['predictions'][:4], colors):
        x, y = segment['frame_numbers'], segment['progress']
        axes[0].plot(x, y, color=color, linewidth=2, label=f'Frames {x[0]}–{x[-1]}')
        axes[0].plot([x[0], x[-1]], [y[0], y[-1]], linestyle='none', marker='o',
                     markersize=7, markerfacecolor='white', markeredgewidth=2, color=color)
    for boundary in [8.5, 16.5, 24.5]:
        axes[0].axvline(boundary, color='#b0b0b0', linestyle=':', linewidth=1)
    axes[0].set_title('Dense: four independent 8-frame segments (no history across segments)', fontsize=11)
    axes[0].legend(loc='upper left', ncol=4, fontsize=9)
    global_pred = record['predictions'][4]
    axes[1].plot(global_pred['frame_numbers'], global_pred['progress'], '-o', color='#2563eb',
                 linewidth=2, markerfacecolor='white', markeredgewidth=1.8, markersize=6)
    axes[1].set_title('Global: 8 uniformly sampled frames from the entire trajectory', fontsize=11)
    for ax in axes:
        ax.set_ylim(-.04, 1.08)
        ax.set_xlim(.5, 32.5)
        ax.set_ylabel('Predicted progress')
        ax.grid(alpha=.2)
        ax.set_yticks([0, .25, .5, .75, 1])
    axes[1].set_xticks(sorted(set([1, 8, 9, 16, 17, 24, 25, 32] + global_pred['frame_numbers'])))
    axes[1].set_xlabel('Frame number in the stored 32-frame trajectory')
    fig.suptitle(f"{record['model_label']} | {record['quality_label']}\n" +
                 textwrap.fill(record['task'], 90) + f"\nID: {record['id']}", fontsize=12)
    fig.text(.5, .012, 'Raw model outputs; no smoothing or rescaling. Open circles mark segment endpoints above and sampled frames below.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .035, 1, .97))
    fig.savefig(dest / f"{record['id']}.png", dpi=160)
    plt.close(fig)


def write_tables_and_gallery(dest, records):
    with (dest/'predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'task', 'quality_label', 'view', 'frame', 'progress'])
        for r in records:
            for p in r['predictions']:
                writer.writerows((r['id'], r['task'], r['quality_label'], p['view'], x, y)
                                 for x, y in zip(p['frame_numbers'], p['progress']))
    with (dest/'boundary_jumps.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'task', 'quality_label', 'boundary', 'left_progress', 'right_progress', 'right_minus_left'])
        for r in records:
            for left, right in zip(r['predictions'][:3], r['predictions'][1:4]):
                a, b = left['progress'][-1], right['progress'][0]
                writer.writerow([r['id'], r['task'], r['quality_label'],
                                 f"{left['frame_numbers'][-1]}->{right['frame_numbers'][0]}", a, b, b-a])
    cards = ''.join(f'<article><h3>{html.escape(r["task"])} — {html.escape(r["quality_label"])}</h3>'
                    f'<a href="{r["id"]}.png"><img loading="lazy" src="{r["id"]}.png" width="900"></a></article>'
                    for r in sorted(records, key=lambda r: (r['task'], r['quality_label'], r['id'])))
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Progress curves</title>'
        '<style>body{font-family:system-ui;max-width:1000px;margin:24px auto}img{max-width:100%}article{margin:36px 0}</style>' +
        f'<h1>{html.escape(records[0]["model_label"])}</h1><p>{len(records)} trajectories. Independent dense segments; raw predictions.</p>' + cards)


def run_model(args, job):
    import numpy as np
    from datasets import load_from_disk
    dest = args.output / job['name']
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / 'raw'
    raw.mkdir(exist_ok=True)
    dataset = load_from_disk(str(args.dataset/'processed_dataset'))
    rows = list(dataset)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty dataset or duplicate IDs')
    plan = sample_indices()
    for row in rows:
        if row['num_frames'] != 32:
            raise ValueError(f"Expected 32 frames: {row['id']}")
    protocol = dict(version=1, model=job, dataset=str(args.dataset), ids=[r['id'] for r in rows],
                    sampling=plan, dtype='bfloat16', attention='sdpa', independent_segments=True,
                    rescale=False, smooth=False, batch_size=args.batch_size)
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (dest/'protocol.json').exists() and json.loads((dest/'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Existing output uses a different protocol; choose a new --output directory')
    write_json(dest/'protocol.json', protocol)
    missing = [r for r in rows if not (raw/f"{r['id']}.json").exists()]
    if args.plot_only and missing:
        raise ValueError('Missing predictions for --plot-only')
    started = time.time()
    if missing:
        import torch
        from robometer.data.dataset_types import Trajectory, ProgressSample
        from robometer.evals.baselines.rbm_model import RBMModel
        torch.manual_seed(42)
        model = RBMModel(job['checkpoint'], strict_offline=True, base_model_path=job['base'], device_map=job.get('device_map'))
        write_json(dest/'checkpoint_load_report.json', model.model.checkpoint_load_report)
        tasks, partial = [], {}
        for row in missing:
            frame_path = args.dataset/'frames'/Path(row['frames']).name
            with np.load(frame_path, allow_pickle=False) as archive:
                frames = archive['frames'].copy()
            if frames.shape != (32, 240, 320, 3) or frames.dtype != np.uint8:
                raise ValueError(f'Unexpected RGB array: {frame_path}, {frames.shape}, {frames.dtype}')
            partial[row['id']] = dict(id=row['id'], task=row['task'], quality_label=row['quality_label'],
                model=job['name'], model_label=job['label'], checkpoint=job['checkpoint'],
                frame_path=str(frame_path), frame_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                protocol_sha256=signature, predictions=[])
            for view, indices in plan:
                selected = frames[indices].copy()
                sample = ProgressSample(trajectory=Trajectory(frames=selected, frames_shape=selected.shape,
                    task=row['task'], id=row['id'], quality_label=row['quality_label']))
                tasks.append((row['id'], view, indices, sample))
        for offset in range(0, len(tasks), args.batch_size):
            batch = tasks[offset:offset+args.batch_size]
            predictions = model.compute_batched_progress([item[3] for item in batch])
            if len(predictions) != len(batch):
                raise ValueError('Prediction batch length mismatch')
            for (uid, view, indices, _), prediction in zip(batch, predictions):
                values = np.asarray(prediction, dtype=float)
                if values.shape != (8,) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError(f'Invalid progress: {uid} {view}: {prediction}')
                record = partial[uid]
                record['predictions'].append(dict(view=view, frame_numbers=[i+1 for i in indices], progress=values.tolist()))
                if len(record['predictions']) == 5:
                    write_json(raw/f'{uid}.json', record)
                    plot_record(record, dest)
            print(f"PROGRESS {job['name']} {min(offset+len(batch),len(tasks))}/{len(tasks)} inputs", flush=True)
    records = [json.loads((raw/f"{r['id']}.json").read_text()) for r in rows]
    for record in records:
        if record['protocol_sha256'] != signature:
            raise ValueError('Cached prediction protocol mismatch')
        if args.plot_only or not (dest/f"{record['id']}.png").exists():
            plot_record(record, dest)
    write_tables_and_gallery(dest, records)
    write_json(dest/'COMPLETE.json', dict(trajectories=len(records), inputs=len(records)*5,
        predicted_points=len(records)*40, elapsed_this_invocation=time.time()-started))
    print(f"COMPLETE {job['name']}: {len(records)} figures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--output', type=Path, default=ROOT/'eval_figure/output/usc_xarm')
    parser.add_argument('--models', default=','.join(m['name'] for m in MODELS))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size and limit must be positive')
    names = args.models.split(',')
    if set(names) - {m['name'] for m in MODELS}:
        parser.error('Unknown model name')
    jobs = [m for m in MODELS if m['name'] in names]
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(jobs) != 1:
            parser.error('Worker requires exactly one model')
        run_model(args, jobs[0])
        return
    for job in jobs:
        dest = args.output/job['name']
        dest.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--models', job['name'],
                   '--dataset', str(args.dataset), '--output', str(args.output), '--batch-size', str(args.batch_size)]
        if args.limit:
            command += ['--limit', str(args.limit)]
        if args.plot_only:
            command += ['--plot-only']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=job['gpus'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', MPLBACKEND='Agg', PYTHONUNBUFFERED='1',
                   PYTHONHASHSEED='42', TOKENIZERS_PARALLELISM='false')
        write_json(dest/'invocation.json', dict(command=command, gpus=job['gpus']))
        print(f"START {job['name']} GPUs={job['gpus']}", flush=True)
        with (dest/'run.log').open('a') as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"DONE {job['name']}", flush=True)
    links = ''.join(f'<li><a href="{m["name"]}/index.html">{html.escape(m["label"])}</a></li>' for m in jobs)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>USC xArm progress curves</h1><ul>'+links+'</ul>')


if __name__ == '__main__':
    main()
