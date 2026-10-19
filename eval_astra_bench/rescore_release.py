"""Rescore a relabeled subset only after verifying identical model inputs.

Retains the original ICL references and actual frame pairs; never resamples pairs
based on new labels. Image SHA256s are checked against both releases' catalogs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import csv
import json
from pathlib import Path

from .core import (digest, file_hash, load_benchmark, read_jsonl, relation, score,
                   unique_by_id, validate_predictions, validate_progress, write_json, write_jsonl)


def checksum_catalog(root):
    result = {}
    for line in (root/'checksums.sha256').read_text().splitlines():
        checksum, path = line.split(maxsplit=1)
        path = path.lstrip('*')
        if path.startswith('./'):
            path = path[2:]
        if path in result:
            raise ValueError(f'Duplicate checksum entry: {path}')
        result[path] = checksum
    return result


def verified_file(root, relative, catalog):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'Invalid release-relative path: {relative}')
    if file_hash(root/relative) != catalog[relative]:
        raise ValueError(f'Release file checksum mismatch: {relative}')


def validate_reuse(old_rows, new_rows, old_catalog, new_catalog):
    old_by = unique_by_id(old_rows)
    new_by = unique_by_id(new_rows)
    for uid, row in new_by.items():
        if uid not in old_by:
            raise ValueError(f'Trajectory needs fresh inference: {uid}')
        old = old_by[uid]
        if row['instruction'] != old['instruction'] or row['group'] != old['reference_group']:
            raise ValueError(f'Task/group changed: {uid}')
        if row['num_frames'] != 32 or len(row['image_paths']) != 32 or len(row['image_sha256']) != 32:
            raise ValueError(f'Frame count changed: {uid}')
        for previous, current, expected in zip(old['image_paths'], row['image_paths'], row['image_sha256']):
            if not old_catalog[previous] == expected == new_catalog[current]:
                raise ValueError(f'Image changed: {uid}: {current}')
        ref = old['icl_reference']
        if ref['trajectory_id'] not in new_by:
            raise ValueError(f'Original ICL reference missing: {uid}')
        for path, index in zip(ref['image_paths'], ref['frame_indices']):
            if old_catalog[path] != new_by[ref['trajectory_id']]['image_sha256'][index]:
                raise ValueError(f'Original ICL reference image changed: {uid}')
    return old_by, new_by


def relabel_pairs(pairs, new_by, targets):
    result = []
    for original in pairs:
        a, b = original['a'], original['b']
        if a[0] not in new_by or b[0] not in new_by:
            continue
        if original['mode'] == 'cross':
            if (new_by[a[0]]['instruction'] != new_by[b[0]]['instruction'] or
                new_by[a[0]]['reference_sha256'] != new_by[b[0]]['reference_sha256']):
                raise ValueError(f'Cross pair no longer shares task and label scale: {original["pair_id"]}')
        pair = copy.deepcopy(original)
        pair['gold_relation'] = relation(targets[a[0]]['progress'][a[1]], targets[b[0]]['progress'][b[1]], .1)
        # No adjudicated human-conflict field is supplied for this release.
        pair['involves_known_conflict'] = False
        result.append(pair)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--original-benchmark', type=Path, required=True)
    parser.add_argument('--original-plan', type=Path, required=True)
    parser.add_argument('--benchmark-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    for path in (args.benchmark_output, args.output):
        if path.exists():
            raise FileExistsError(f'Use a new output: {path}')
    old_info, old_inputs = load_benchmark(args.original_benchmark)
    if old_info['tie_threshold'] != .1:
        raise ValueError('This rescore preserves the original 0.1 threshold')
    old_root = Path(old_info['data_root'])
    root = args.data_root.resolve()
    if file_hash(old_root/'checksums.sha256') != old_info['source_checksums_sha256']:
        raise ValueError('Original image checksum catalog changed')
    old_catalog, catalog = checksum_catalog(old_root), checksum_catalog(root)
    for name in ('dataset.json','manifests/trajectories.jsonl','metadata/source_labels.jsonl','splits/excluded_towel.json'):
        verified_file(root, name, catalog)
    release = json.loads((root/'dataset.json').read_text())
    rows = read_jsonl(root/'manifests/trajectories.jsonl')
    if len(rows) != release['trajectories']:
        raise ValueError('Release trajectory count mismatch')
    # Explicitly retain the user's no-towel policy even if a future export includes them.
    rows = [r for r in rows if r['instruction'] != 'Fold the towel in half']
    old_by, new_by = validate_reuse(old_inputs, rows, old_catalog, catalog)
    image_paths = sorted({p for row in rows for p in row['image_paths']})
    print(f'Image catalogs match. Verifying {len(image_paths)} actual PNG files with {args.workers} readers.', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda p: verified_file(root,p,catalog), image_paths), 1):
            if i % 1000 == 0:
                print(f'Verified images {i}/{len(image_paths)}', flush=True)
    source = unique_by_id(read_jsonl(root/'metadata/source_labels.jsonl'))
    inputs, targets = [], []
    for row in sorted(rows, key=lambda r:r['trajectory_id']):
        uid = row['trajectory_id']
        verified_file(root, row['label_path'], catalog)
        if catalog[row['label_path']] != row['label_sha256']:
            raise ValueError(f'Annotation manifest checksum mismatch: {uid}')
        label = json.loads((root/row['label_path']).read_text())
        if label['trajectory_id'] != uid or label['reference_sha256'] != row['reference_sha256']:
            raise ValueError(f'Annotation identity mismatch: {uid}')
        if label['prompt_version'] != release['prompt_version'] or row['prompt_version'] != release['prompt_version']:
            raise ValueError(f'Unexpected prompt version: {uid}')
        frames = label['frame_labels']
        if [f['frame_index'] for f in frames] != list(range(32)):
            raise ValueError(f'Frame index mismatch: {uid}')
        progress = [f['progress'] for f in frames]
        validate_progress(progress)
        model_input = copy.deepcopy(old_by[uid])
        model_input.update(image_paths=row['image_paths'], reference_key=row['reference_sha256'])
        ref = model_input['icl_reference']
        ref['image_paths'] = [new_by[ref['trajectory_id']]['image_paths'][i] for i in ref['frame_indices']]
        inputs.append(model_input)
        targets.append(dict(trajectory_id=uid, progress=progress, label_sha256=row['label_sha256'],
                            prompt_version=label['prompt_version'],
                            source_quality_label=source[uid]['original_quality_label'],
                            terminal_outcome=label['terminal_outcome']['label'],
                            human_model_conflict=False, human_conflict_status='not_provided_by_release',
                            success_basis='new_release_model_terminal_outcome'))
    targets_by = unique_by_id(targets)
    original_pairs = read_jsonl(args.original_benchmark/'pairs.jsonl')
    pairs = relabel_pairs(original_pairs, new_by, targets_by)
    excluded_groups = json.loads((root/'splits/excluded_towel.json').read_text())
    excluded_ids = {uid for group in excluded_groups for uid in group['trajectory_ids']}
    excluded_ids.update(r['trajectory_id'] for r in read_jsonl(root/'manifests/trajectories.jsonl')
                        if r['instruction'] == 'Fold the towel in half')
    assert not excluded_ids & set(new_by)
    args.benchmark_output.mkdir(parents=True)
    for name, values in [('inputs',inputs),('targets',targets),('pairs',pairs)]:
        write_jsonl(args.benchmark_output/f'{name}.jsonl',values)
    info = dict(schema_version=old_info['schema_version'], source_release=release['release'],
                source_manifest_sha256=file_hash(root/'manifests/trajectories.jsonl'),
                source_checksums_sha256=file_hash(root/'checksums.sha256'),
                original_benchmark_id=old_info['benchmark_id'],
                seed=old_info['seed'], tie_threshold=.1, tie_rule=old_info['tie_rule'], pair_scope='reference',
                pair_sampling='retain_original_pair_ids_and_endpoints_filtered_to_new_release',
                inherited_sampling_strata='original reference hashes; validated against new same-task/same-scale constraints',
                reference_frames=old_info['reference_frames'], reference_progress=old_info['reference_progress'],
                reference_policy='preserve_original_model_ICL_inputs; new labeler reference does not replace model reference',
                exclude_known_conflicts=False, human_conflict_metadata='not_provided_by_new_release',
                excluded=[dict(trajectory_id=uid,reason='excluded_towel_all_sources') for uid in sorted(excluded_ids)],
                counts=dict(trajectories=len(inputs),frames=32*len(inputs),tasks=len({r['instruction'] for r in inputs}),
                            icl_references=len({r['icl_reference']['trajectory_id'] for r in inputs}),
                            reference_scales=len({r['reference_key'] for r in inputs}),pairs=len(pairs),
                            modes=dict(Counter(p['mode'] for p in pairs)),
                            gold_relations=dict(Counter(p['gold_relation'] for p in pairs))),
                files={name:file_hash(args.benchmark_output/name) for name in ('inputs.jsonl','targets.jsonl','pairs.jsonl')})
    info['benchmark_id'] = digest(info)
    info['data_root'] = str(root)
    write_json(args.benchmark_output/'benchmark.json', info)
    load_benchmark(args.benchmark_output)
    plan = json.loads(args.original_plan.read_text())
    if plan['benchmark_id'] != old_info['benchmark_id']:
        raise ValueError('Prediction plan uses a different original benchmark')
    args.output.mkdir(parents=True)
    results = []
    for job in plan['jobs']:
        prediction_path = Path(job['output'])
        old_predictions = validate_predictions(old_info,old_inputs,read_jsonl(prediction_path))
        run = json.loads(prediction_path.with_suffix('.run.json').read_text())
        if run['benchmark_id'] != old_info['benchmark_id'] or run['model_id'] != job['name']:
            raise ValueError('Original run metadata mismatch')
        if run['adapter'] == 'icl' and (run['reference_frames'] != info['reference_frames'] or
                                      run['reference_progress'] != info['reference_progress']):
            raise ValueError('Original model ICL protocol mismatch')
        destination = args.output/job['name']
        destination.mkdir()
        converted = [dict(old_predictions[row['trajectory_id']],benchmark_id=info['benchmark_id']) for row in inputs]
        validate_predictions(info,inputs,converted)
        write_jsonl(destination/'predictions.jsonl',converted)
        write_json(destination/'reuse.json',dict(original_predictions=str(prediction_path),
                   original_predictions_sha256=file_hash(prediction_path),original_run=run,
                   verification='All new image bytes match original checksum catalog; tasks, order, ICL inputs unchanged',
                   score_only=True,original_benchmark_id=old_info['benchmark_id'],benchmark_id=info['benchmark_id']))
        report = score(args.benchmark_output,destination/'predictions.jsonl',destination/'scores')
        item = dict(model_id=job['name'])
        for mode in ('within','cross','overall'):
            values = report['overall'] if mode=='overall' else report['by_mode'][mode]
            for key,value in values.items():
                item[f'{mode}_{key}']=value
        results.append(item)
        print(f"{job['name']}: within={item['within_accuracy']:.2%} cross={item['cross_accuracy']:.2%}",flush=True)
    results.sort(key=lambda r:(0,int(r['model_id'].split('step')[1])) if r['model_id'].startswith('icl_4b') else
                 (1,int(r['model_id'].split('step')[1])) if r['model_id'].startswith('icl_27b') else
                 (2,0 if 'full32' in r['model_id'] else 1))
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),benchmark_id=info['benchmark_id'],
                 source_release=release['release'],tie_threshold=.1,counts=info['counts'],
                 excluded_towel=len(excluded_ids),images_verified=len(image_paths),
                 original_benchmark_id=old_info['benchmark_id'],predictions_reused=True,
                 pair_sampling=info['pair_sampling'],results=results)
    write_json(args.output/'summary.json',summary)
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    lines=['# Equal-keyframes v1：固定帧对复评','',
           '阈值 0.1；502 条轨迹；排除全部 68 条毛巾任务。',
           '沿用旧帧对并过滤本版缺少的轨迹：同轨迹 16064 对，跨轨迹 10752 对。',
           '使用新标注 progress（等权 P）；原模型输入经逐图 SHA256 核验一致，保留原 ICL 参考并复用预测。','',
           '| 模型 | 同轨迹 | 跨轨迹 | 总体 |','|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['model_id']} | {r['within_accuracy']:.2%} | {r['cross_accuracy']:.2%} | {r['overall_accuracy']:.2%} |")
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
