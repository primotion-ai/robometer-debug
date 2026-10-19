"""Benchmark preparation and scoring. Standard library only; no model imports."""
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import random

DEFAULT_DATA = '/mnt_slow/wangchunfeng/dataset/robometer-1m-ood-label/current'
REFERENCE_FRAMES = [0, 8, 16, 24, 31]
REFERENCE_PROGRESS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCHEMA = 'astra_order_bench/v1'
PREDICTION_SCHEMA = 'astra_progress_predictions/v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(canonical(row) + '\n')
    temp.replace(path)


def unique_by_id(rows, key='trajectory_id'):
    result = {}
    for row in rows:
        uid = row[key]
        if uid in result:
            raise ValueError(f'Duplicate {key}: {uid}')
        result[uid] = row
    return result


def validate_progress(values, length=32):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f'Expected {length} progress values')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Progress must contain finite numbers in [0,1]; no clipping/normalization is implicit')


def relation(a, b, threshold):
    # Decimal removes boundary artifacts such as 0.4 - 0.3 > 0.1.
    delta = Decimal(str(a)) - Decimal(str(b))
    if abs(delta) <= Decimal(str(threshold)):
        return 'eq'
    return 'gt' if delta > 0 else 'lt'


def uniform_indices(num_frames, sample_frames):
    """Uniform whole-trajectory sample, matching numpy.linspace(..., dtype=int)."""
    if not 2 <= sample_frames <= num_frames:
        raise ValueError('Require 2 <= sample_frames <= num_frames')
    return [i * (num_frames - 1) // (sample_frames - 1) for i in range(sample_frames)]


def scoped_rng(seed, scope):
    return random.Random(int(digest([seed, scope]), 16))


def prepare(data_root, output, *, seed=20261018, threshold=0.1,
            within_per_trajectory=32, cross_per_group=256, pair_scope='reference',
            exclude_known_conflicts=False):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    if within_per_trajectory < 0 or cross_per_group < 0 or within_per_trajectory + cross_per_group == 0:
        raise ValueError('Pair budgets must be nonnegative and not both zero')
    if pair_scope not in ('reference', 'task'):
        raise ValueError('pair_scope must be reference or task')
    data_root, output = Path(data_root).resolve(), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Frozen benchmark output must be empty: {output}')
    source_rows = read_jsonl(data_root / 'manifests/trajectories.jsonl')
    unique_by_id(source_rows)
    rows, excluded = [], []
    for row in sorted(source_rows, key=lambda r: r['trajectory_id']):
        reason = None
        if row['source'] == 'rfm_new_mit_franka_rfm' and row['instruction'] == 'Fold the towel in half':
            reason = 'excluded_mit_franka_towel'
        elif exclude_known_conflicts and row['human_model_conflict']:
            reason = 'known_human_model_conflict'
        (excluded if reason else rows).append(dict(row, exclusion_reason=reason) if reason else row)
    if not rows:
        raise ValueError('No trajectories selected')
    groups = defaultdict(list)
    for row in rows:
        groups[row['reference_group']].append(row)
    references = {}
    for group, members in sorted(groups.items()):
        # Select using source success, not Astra progress or model predictions.
        candidates = [r for r in members if r['source_quality_label'] == 'successful']
        if not candidates:
            raise ValueError(f'No source-successful ICL reference in {group}; supply a curated dataset')
        chosen = scoped_rng(seed, ['icl_reference', group]).choice(candidates)
        references[group] = dict(trajectory_id=chosen['trajectory_id'], frame_indices=REFERENCE_FRAMES,
                                 progress=REFERENCE_PROGRESS, selection_basis='source_quality_label=successful',
                                 image_paths=[chosen['image_paths'][i] for i in REFERENCE_FRAMES])
    inputs, targets = [], []
    for index, row in enumerate(rows):
        uid = row['trajectory_id']
        if row['num_frames'] != 32 or len(row['image_paths']) != 32:
            raise ValueError(f'Expected 32 frames: {uid}')
        label_path = data_root / row['label_path']
        if file_hash(label_path) != row['label_sha256']:
            raise ValueError(f'Source annotation hash mismatch: {uid}')
        annotation = json.loads(label_path.read_text())
        if annotation['trajectory_id'] != uid or annotation['reference_sha256'] != row['reference_key']:
            raise ValueError(f'Annotation/reference identity mismatch: {uid}')
        frames = annotation['frame_labels']
        if [x['frame_index'] for x in frames] != list(range(32)):
            raise ValueError(f'Invalid frame indices: {uid}')
        progress = [x['progress'] for x in frames]
        validate_progress(progress)
        inputs.append({**{k: row[k] for k in ('trajectory_id', 'instruction', 'source', 'reference_group',
                                               'group_name', 'reference_key', 'num_frames', 'image_paths')},
                       'icl_reference': references[row['reference_group']]})
        targets.append(dict(trajectory_id=uid, progress=progress, label_sha256=row['label_sha256'],
                            human_model_conflict=row['human_model_conflict'],
                            source_quality_label=row['source_quality_label'],
                            success_basis=row['success_basis'], prompt_version=row['prompt_version']))
        if (index + 1) % 100 == 0:
            print(f'Validated labels {index + 1}/{len(rows)}', flush=True)
    gold = unique_by_id(targets)
    by_id = unique_by_id(inputs)
    pairs = []

    def add_pair(mode, group, a, b, rng):
        if rng.randrange(2):
            a, b = b, a
        pa, pb = gold[a[0]]['progress'][a[1]], gold[b[0]]['progress'][b[1]]
        ref_ids = {references[by_id[u]['reference_group']]['trajectory_id'] for u in (a[0], b[0])}
        pairs.append(dict(pair_id=digest([mode, group, sorted([a, b])])[:24], mode=mode, group=group,
                          instruction=by_id[a[0]]['instruction'], a=list(a), b=list(b),
                          gold_relation=relation(pa, pb, threshold),
                          involves_icl_reference=a[0] in ref_ids or b[0] in ref_ids,
                          involves_known_conflict=gold[a[0]]['human_model_conflict'] or gold[b[0]]['human_model_conflict']))

    frame_combinations = list(combinations(range(32), 2))
    for row in inputs:
        uid = row['trajectory_id']
        rng = scoped_rng(seed, ['within', uid])
        for i, j in rng.sample(frame_combinations, min(within_per_trajectory, len(frame_combinations))):
            add_pair('within', uid, (uid, i), (uid, j), rng)
    pair_groups = defaultdict(list)
    for row in inputs:
        key = row['reference_key'] if pair_scope == 'reference' else row['instruction']
        pair_groups[key].append(row['trajectory_id'])
    skipped = []
    for key, ids in sorted(pair_groups.items()):
        if len(ids) < 2:
            skipped.append(dict(group=key, trajectory_ids=ids, reason='only_one_trajectory'))
            continue
        # All trajectories contain 32 frames: each trajectory pair has exactly 1024 frame pairs.
        trajectory_pairs = list(combinations(sorted(ids), 2))
        capacity = len(trajectory_pairs) * 32 * 32
        rng = scoped_rng(seed, ['cross', key])
        for flat in rng.sample(range(capacity), min(cross_per_group, capacity)):
            pair_index, frame_pair = divmod(flat, 1024)
            i, j = divmod(frame_pair, 32)
            a, b = trajectory_pairs[pair_index]
            add_pair('cross', key, (a, i), (b, j), rng)
    if not pairs:
        raise ValueError('No pairs generated')
    unique_by_id(pairs, 'pair_id')
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('inputs', inputs), ('targets', targets), ('pairs', pairs)]:
        write_jsonl(output / f'{name}.jsonl', values)
    info = dict(schema_version=SCHEMA, source_release=json.loads((data_root/'dataset.json').read_text())['release'],
                source_manifest_sha256=file_hash(data_root/'manifests/trajectories.jsonl'),
                source_checksums_sha256=file_hash(data_root/'checksums.sha256'),
                seed=seed, tie_threshold=threshold, tie_rule='abs(a-b)<=threshold', pair_scope=pair_scope,
                within_per_trajectory=within_per_trajectory, cross_per_group=cross_per_group,
                reference_frames=REFERENCE_FRAMES, reference_progress=REFERENCE_PROGRESS,
                reference_policy='seeded source-successful reference per scene group; references included in scoring',
                exclude_known_conflicts=exclude_known_conflicts,
                excluded=[{'trajectory_id':r['trajectory_id'], 'reason':r['exclusion_reason']} for r in excluded],
                skipped_cross_groups=skipped,
                counts=dict(trajectories=len(inputs), frames=32*len(inputs), tasks=len({r['instruction'] for r in inputs}),
                            icl_references=len(references), reference_scales=len(pair_groups),
                            pairs=len(pairs), modes=dict(Counter(p['mode'] for p in pairs)),
                            gold_relations=dict(Counter(p['gold_relation'] for p in pairs))),
                files={name:file_hash(output/name) for name in ('inputs.jsonl', 'targets.jsonl', 'pairs.jsonl')})
    info['benchmark_id'] = digest(info)
    # Location is not identity; a copied data release can be used with --data-root.
    info['data_root'] = str(data_root)
    write_json(output/'benchmark.json', info)
    return info


def load_benchmark(root):
    root = Path(root)
    info = json.loads((root/'benchmark.json').read_text())
    identity = {k:v for k,v in info.items() if k not in ('data_root', 'benchmark_id')}
    if info['schema_version'] != SCHEMA or digest(identity) != info['benchmark_id']:
        raise ValueError('Invalid benchmark identity')
    for name, expected in info['files'].items():
        if file_hash(root/name) != expected:
            raise ValueError(f'Frozen benchmark changed: {name}')
    rows = read_jsonl(root/'inputs.jsonl')
    unique_by_id(rows)
    return info, rows


def prediction_row(info, row, progress):
    validate_progress(progress, row['num_frames'])
    return dict(schema_version=PREDICTION_SCHEMA, benchmark_id=info['benchmark_id'],
                trajectory_id=row['trajectory_id'], frame_indices=list(range(row['num_frames'])), progress=progress)


def validate_predictions(info, rows, predictions, *, allow_partial=False):
    expected = unique_by_id(rows)
    actual = unique_by_id(predictions)
    if set(actual)-set(expected):
        raise ValueError(f'Unexpected trajectory IDs: {sorted(set(actual)-set(expected))}')
    if not allow_partial and set(actual) != set(expected):
        raise ValueError(f'Missing {len(set(expected)-set(actual))} trajectories')
    for uid, row in actual.items():
        if row.get('schema_version') != PREDICTION_SCHEMA or row.get('benchmark_id') != info['benchmark_id']:
            raise ValueError(f'Prediction schema/benchmark mismatch: {uid}')
        indices = row.get('frame_indices')
        if indices != list(range(expected[uid]['num_frames'])) or any(type(i) is not int for i in indices):
            raise ValueError(f'Frame indices must be complete, ordered and zero-based: {uid}')
        validate_progress(row['progress'], expected[uid]['num_frames'])
    return actual


def score(benchmark, predictions_path, output):
    info, rows = load_benchmark(benchmark)
    predictions = validate_predictions(info, rows, read_jsonl(predictions_path))
    pairs = read_jsonl(Path(benchmark)/'pairs.jsonl')
    buckets = defaultdict(lambda: [0, 0])
    confusion = {g:{p:0 for p in ('lt','eq','gt')} for g in ('lt','eq','gt')}
    details = []
    for pair in pairs:
        a, b = pair['a'], pair['b']
        pa, pb = predictions[a[0]]['progress'][a[1]], predictions[b[0]]['progress'][b[1]]
        pred = relation(pa, pb, info['tie_threshold'])
        gold = pair['gold_relation']
        correct = int(pred == gold)
        confusion[gold][pred] += 1
        keys = [('overall', 'all'), ('by_mode', pair['mode']), ('by_task', pair['instruction']),
                ('by_gold_relation', gold), ('by_reference_exposure', str(pair['involves_icl_reference'])),
                ('by_known_conflict', str(pair['involves_known_conflict']))]
        for category, key in keys:
            buckets[category, key][0] += correct
            buckets[category, key][1] += 1
        details.append(dict(pair_id=pair['pair_id'], mode=pair['mode'], gold_relation=gold,
                            predicted_relation=pred, progress_a=pa, progress_b=pb, correct=correct))
    report = dict(schema_version=SCHEMA, benchmark_id=info['benchmark_id'],
                  predictions_sha256=file_hash(predictions_path), tie_threshold=info['tie_threshold'],
                  confusion_matrix=confusion, confusion_axes='rows=gold; columns=prediction')
    for (category, key), (correct, total) in sorted(buckets.items()):
        report.setdefault(category, {})[key] = dict(correct=correct, total=total, accuracy=correct/total)
    report['overall'] = report['overall']['all']
    report['task_macro_accuracy'] = sum(x['accuracy'] for x in report['by_task'].values())/len(report['by_task'])
    report['mode_macro_accuracy'] = sum(x['accuracy'] for x in report['by_mode'].values())/len(report['by_mode'])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Score output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'metrics.json', report)
    write_jsonl(output/'pair_results.jsonl', details)
    return report
