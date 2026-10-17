"""Reproduce the six-trajectory Close the drawer annotation experiment."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra_label.backend import CodexBackend, write_json
from astra_label.connection import Connection, add_connection_arguments, connection_overrides
from astra_label.data import load_trajectory
from astra_label.pipeline import make_anchors, run, save_frame, validate_reference
from astra_label.prompts import REFERENCE_SCHEMA, reference_prompt

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE / 'outputs' / 'close_the_drawer'
API_OUTPUT = PACKAGE / 'outputs_api' / 'close_the_drawer'
DATA = Path('/mnt/workspace/dataset/robometer-1m')
CATEGORY = 'aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking'
REFERENCE_ID = '1d035ec5-5127-4af9-9cca-fe214830d670'
KEYFRAMES = [0, 17, 22, 25, 31]
IDS = [
    '9726c08b-3b48-495a-9aae-f90cebd83330',
    '9a3c5b08-296d-4650-a388-759dc39c8b92',
    '83118d44-d77d-4266-abb8-417b60b54f57',
    '4aef5821-1062-4bac-8251-6a85af493614',
    REFERENCE_ID,
    '859a08b0-fc9a-47a2-af6b-112087cc7863',
]


def calibrate(output=OUTPUT, options=None):
    options = options or {}
    connection = Connection.from_config(options)
    trajectory = load_trajectory(DATA, CATEGORY, REFERENCE_ID)
    anchors = make_anchors(KEYFRAMES, len(trajectory.frames))
    indices = sorted({j for i in KEYFRAMES for j in range(max(0, i-1), min(len(trajectory.frames), i+2))})
    folder = output / 'reference_calibration'
    images_dir = folder / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    images = [save_frame(trajectory.frames, i, 'REFERENCE', images_dir, 768) for i in indices]
    manifest = [{'role': 'reference', 'frame_index': i, 'file': path.name} for i, path in zip(indices, images)]
    prompt = reference_prompt('Close the drawer', trajectory.task, anchors, manifest)
    rubric = CodexBackend(folder, model=options.get('model', 'gpt-6-astra'),
                          reasoning=options.get('reasoning', 'high'), connection=connection).ask(
                              'reference', prompt, images, REFERENCE_SCHEMA)
    validate_reference(rubric, anchors)
    write_json(folder / 'rubric.json', rubric)
    write_json(folder / 'shared_input.json', {
        'prompt': prompt,
        'image_hashes': [hashlib.sha256(p.read_bytes()).hexdigest() for p in images],
        'connection': connection.public(),
        'model': options.get('model', 'gpt-6-astra'),
        'reasoning': options.get('reasoning', 'high'),
    })


class SharedReferenceBackend:
    def __init__(self, output, run_root=OUTPUT, options=None):
        self.output = output
        self.run_root = run_root
        options = options or {}
        self.connection = Connection.from_config(options)
        self.target = CodexBackend(output, model=options.get('model', 'gpt-6-astra'),
                                  reasoning=options.get('reasoning', 'high'), connection=self.connection)

    def ask(self, name, prompt, images, schema):
        if name == 'reference':
            folder = self.run_root / 'reference_calibration'
            source = json.loads((folder / 'shared_input.json').read_text())
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in images]
            if prompt != source['prompt'] or hashes != source['image_hashes']:
                raise ValueError('Reference inputs differ from the shared calibration')
            if source.get('connection', Connection().public()) != self.connection.public():
                raise ValueError('Reference calibration used a different API provider or authentication mode')
            if (source.get('model', 'gpt-6-astra') != self.target.model
                    or source.get('reasoning', 'high') != self.target.reasoning):
                raise ValueError('Reference calibration used a different model or reasoning setting')
            write_json(self.output / 'shared_reference_call.json', {
                'source': str(folder / 'calls' / 'reference'),
                'note': 'Exact same reference prompt and image bytes; reuse one rubric for all six targets.',
            })
            return json.loads((folder / 'rubric.json').read_text())
        return self.target.ask(name, prompt, images, schema)


def annotate(trajectory_id, output=OUTPUT, options=None):
    folder = output / trajectory_id
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        'data_root': str(DATA),
        'reference': {'category': CATEGORY, 'id': REFERENCE_ID, 'keyframes': KEYFRAMES},
        'target': {'category': CATEGORY, 'id': trajectory_id},
        'instruction': 'Close the drawer',
        'model': 'gpt-6-astra', 'reasoning': 'high', 'batch_size': 32,
        'context_frames': 2, 'overview_frames': 8, 'reference_neighbors': 1,
        'max_image_side': 768, 'timeout': 900,
    }
    config.update(options or {})
    write_json(folder / 'job.json', config)
    run(config, folder, backend=SharedReferenceBackend(folder, output, options))
    return trajectory_id


def mark_reference_keyframes(ax, labels):
    """Annotate existing reference scores; never change the label values."""
    values = [labels[index]['progress'] for index in KEYFRAMES]
    ax.scatter(KEYFRAMES, values, marker='D', facecolors='white',
               edgecolors='black', s=40, label='Reference keyframes', zorder=6, clip_on=False)
    for index, value in zip(KEYFRAMES, values):
        ax.axvline(index, color='tab:purple', linestyle=':', linewidth=.9, alpha=.35)
        offset = (-10, -20) if index == KEYFRAMES[-1] else (8, 12)
        ax.annotate(f'idx {index}', (index, value), xytext=offset, textcoords='offset points',
                    ha='right' if index == KEYFRAMES[-1] else 'left', fontsize=9,
                    color='black', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1.5},
                    zorder=7)


def summarize(output=OUTPUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = []
    results = []
    colors = plt.get_cmap('tab10').colors
    fig, ax = plt.subplots(figsize=(12, 6))
    panels, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for n, trajectory_id in enumerate(IDS):
        result = json.loads((output / trajectory_id / 'labels.json').read_text())
        results.append(result)
        labels = result['labels']
        assert [v['frame_index'] for v in labels] == list(range(32))
        x = [v['frame_index'] for v in labels]
        y = [v['progress'] for v in labels]
        is_reference = trajectory_id == REFERENCE_ID
        title = trajectory_id[:8] + (' (reference)' if is_reference else '')
        ax.plot(x, y, color=colors[n], label=title, linewidth=2.5 if is_reference else 1.8)
        panel = axes.flat[n]
        panel.plot(x, y, color=colors[n], marker='.', markersize=4)
        if is_reference:
            mark_reference_keyframes(ax, labels)
            mark_reference_keyframes(panel, labels)
            panel.legend(fontsize=8)
            single, reference_ax = plt.subplots(figsize=(12, 4))
            reference_ax.plot(x, y, color=colors[n], linewidth=2, label=f"Reference progress ({result['model']})")
            mark_reference_keyframes(reference_ax, labels)
            reference_ax.set(title=REFERENCE_ID + ' | reference keyframe indices',
                             xlabel='Stored frame index (zero-based)', ylabel='Progress',
                             ylim=(-.03, 1.03), xlim=(0, 31))
            reference_ax.set_xticks(sorted(set([0, 5, 10, 15, 20, 25, 30] + KEYFRAMES)))
            reference_ax.grid(alpha=.2)
            reference_ax.legend(loc='upper left')
            single.tight_layout()
            single.savefig(output / trajectory_id / 'progress.png', dpi=180)
            single.savefig(output / trajectory_id / 'progress.svg')
            plt.close(single)
        review = [v for v in labels if v['needs_review']]
        if review:
            panel.scatter([v['frame_index'] for v in review], [v['progress'] for v in review],
                          marker='x', color='red', s=30, zorder=6)
        panel.set(title=title, ylim=(-.03, 1.03), xlim=(0, 31), xlabel='Frame index', ylabel='Progress')
        panel.grid(alpha=.2)
        summary.append({
            'id': trajectory_id, 'is_reference': is_reference, 'frames': len(labels),
            'initial_progress': y[0], 'final_progress': y[-1], 'max_progress': max(y),
            'mean_confidence': sum(v['confidence'] for v in labels) / len(labels),
            'needs_review_frames': [v['frame_index'] for v in review],
            'decrease_count': sum(b < a for a, b in zip(y, y[1:])),
        })
    for p in [0, .25, .5, .75, 1]:
        ax.axhline(p, color='gray', alpha=.2, linestyle='--')
    models = ', '.join(sorted({result['model'] for result in results}))
    ax.set(title=f'Close the drawer | {models} direct progress | shared reference rubric',
           xlabel='Stored frame index (not physical time)', ylabel='Progress', ylim=(-.03, 1.03), xlim=(0,31))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=180)
    fig.savefig(output / 'comparison.svg')
    panels.suptitle('Direct model labels (red crosses: model requests review)', fontsize=14)
    panels.tight_layout()
    panels.savefig(output / 'all_trajectories.png', dpi=180)
    plt.close('all')
    write_json(output / 'summary.json', summary)
    with (output / 'all_labels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['trajectory_id', *results[0]['labels'][0]])
        writer.writeheader()
        for result in results:
            for label in result['labels']:
                writer.writerow({'trajectory_id': result['target']['id'], **label})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true', help='redraw existing labels without any model calls')
    parser.add_argument('--output', type=Path, help='output root; API defaults to astra_label/outputs_api/close_the_drawer')
    parser.add_argument('--model', help='explicit model override; default remains gpt-6-astra')
    parser.add_argument('--reasoning', choices=('low', 'medium', 'high', 'xhigh', 'max'))
    add_connection_arguments(parser)
    args = parser.parse_args()
    options = connection_overrides(args)
    for key in ('model', 'reasoning'):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if options.get('auth_mode') == 'local':
        options.pop('auth_mode')  # Preserve the original implicit-local manifests and caches.
    output = (args.output or (API_OUTPUT if args.auth_mode == 'api' else OUTPUT)).resolve()
    if args.plot_only:
        summarize(output)
        raise SystemExit(0)
    if args.auth_mode == 'api' and output == OUTPUT:
        parser.error('Use a separate --output directory for API results to preserve the original local experiment.')
    calibrate(output, options)
    errors = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(annotate, ident, output, options): ident for ident in IDS}
        for future in as_completed(futures):
            ident = futures[future]
            try:
                print(f'COMPLETE {future.result()}', flush=True)
            except Exception as exc:
                errors.append({'id': ident, 'error': str(exc)})
                print(f'FAILED {ident}: {exc}', flush=True)
    write_json(output / 'errors.json', errors)
    if errors:
        raise SystemExit('Some trajectories failed; inspect errors.json and rerun to resume cached calls.')
    summarize(output)
