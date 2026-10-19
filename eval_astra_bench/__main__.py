"""python -m eval_astra_bench {prepare,predict,score,validate}."""
import argparse
import fcntl
import json
from pathlib import Path

from .core import (DEFAULT_DATA, PREDICTION_SCHEMA, file_hash, load_benchmark, prediction_row,
                   prepare, read_jsonl, score, validate_predictions, write_json, canonical)


def run_predictions(args):
    info, rows = load_benchmark(args.benchmark)
    root = (args.data_root or Path(info['data_root'])).resolve()
    checkpoint = args.checkpoint.resolve()
    if file_hash(root/'manifests/trajectories.jsonl') != info['source_manifest_sha256']:
        raise ValueError('Data root is a different release: source manifest hash mismatch')
    if file_hash(root/'checksums.sha256') != info['source_checksums_sha256']:
        raise ValueError('Data root is a different release: checksum manifest mismatch')
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or args.official_frames not in (8, 16, 32):
        raise ValueError('Batch size/limit must be positive; official frames must be 8, 16 or 32')
    # Only small metadata is hashed here; weight paths, sizes and mtimes are also recorded.
    metadata_files = ('icl_config.yaml', 'icl_state.json', 'config.yaml', 'config.json', 'processor/tokenizer_config.json')
    weights = sorted(checkpoint.glob('*.safetensors')) + sorted(checkpoint.glob('pytorch_model_fsdp_0/*'))
    identity = dict(checkpoint=str(checkpoint),
                    metadata_sha256={p:file_hash(checkpoint/p) for p in metadata_files if (checkpoint/p).is_file()},
                    weights=[dict(path=str(p.relative_to(checkpoint)), size=p.stat().st_size,
                                  mtime_ns=p.stat().st_mtime_ns) for p in weights if p.is_file()])
    run = dict(schema_version=PREDICTION_SCHEMA, benchmark_id=info['benchmark_id'], adapter=args.adapter,
               model_id=args.model_id, checkpoint=identity, data_root=str(root), batch_size=args.batch_size,
               device_map=args.device_map, base_model=str(args.base_model.resolve()) if args.base_model else None,
               official_frames=args.official_frames if args.adapter == 'official' else None,
               reference_frames=info['reference_frames'] if args.adapter == 'icl' else None,
               reference_progress=info['reference_progress'] if args.adapter == 'icl' else None)
    if args.dry_run:
        print(json.dumps(dict(run=run, trajectories=len(rows), queries=32*len(rows),
                              model_loaded=False), ensure_ascii=False, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sidecar = args.output.with_suffix('.run.json')
        existing = []
        if args.output.exists() or sidecar.exists():
            if not args.resume:
                raise FileExistsError('Use a new output or --resume')
            if not sidecar.exists() or json.loads(sidecar.read_text()) != run:
                raise ValueError('Resume metadata differs; use a new output for this protocol/checkpoint')
            existing = read_jsonl(args.output) if args.output.exists() else []
        done = validate_predictions(info, rows, existing, allow_partial=True)
        pending = [r for r in rows if r['trajectory_id'] not in done]
        if args.limit:
            pending = pending[:args.limit]
        if not pending:
            print(f'Already complete: {len(done)}/{len(rows)} trajectories')
            return
        # Validate image availability for each row lazily in the adapter.
        from .adapters import ICLAdapter, OfficialAdapter
        cls = ICLAdapter if args.adapter == 'icl' else OfficialAdapter
        adapter = cls(checkpoint=checkpoint, data_root=root, batch_size=args.batch_size,
                      device_map=args.device_map, base_model=run['base_model'], official_frames=args.official_frames)
        write_json(sidecar, run)
        write_json(args.output.with_suffix('.load.json'), adapter.report)
        import time
        import torch
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        trajectory_batch_size = args.batch_size if args.adapter == 'official' else 1
        with args.output.open('a') as stream:
            for start in range(0, len(pending), trajectory_batch_size):
                batch = pending[start:start+trajectory_batch_size]
                if args.adapter == 'official':
                    results = adapter.predict_many(batch)
                else:
                    results = [(adapter.predict(batch[0]), None)]
                for offset, (row, (progress, details)) in enumerate(zip(batch, results)):
                    result = prediction_row(info, row, progress)
                    if details is not None:
                        result['adapter_details'] = details
                    stream.write(canonical(result) + '\n')
                    stream.flush()
                    print(f"Predicted {len(done)+start+offset+1}/{len(rows)}: {row['trajectory_id']}", flush=True)
        elapsed = time.monotonic()-started
        write_json(args.output.with_suffix('.runtime.json'), dict(
            benchmark_id=info['benchmark_id'], newly_predicted_trajectories=len(pending),
            completed_trajectories=len(done)+len(pending), prediction_seconds=elapsed,
            trajectories_per_second=len(pending)/elapsed,
            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            requested_batch_size=args.batch_size,
            batch_unit='trajectories' if args.adapter == 'official' else 'query_frames'))
        all_rows = read_jsonl(args.output)
        validate_predictions(info, rows, all_rows, allow_partial=True)
        print(f'Saved {len(all_rows)}/{len(rows)} trajectories: {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Freeze trajectories, ICL references, targets and random pairs once')
    p.add_argument('--data-root', type=Path, default=Path(DEFAULT_DATA))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261018)
    p.add_argument('--threshold', type=float, default=.1)
    p.add_argument('--within-per-trajectory', type=int, default=32)
    p.add_argument('--cross-per-group', type=int, default=256)
    p.add_argument('--pair-scope', choices=['reference','task'], default='reference')
    p.add_argument('--exclude-known-conflicts', action='store_true')
    p = sub.add_parser('predict', help='Stage 1: generate model-independent per-frame JSONL predictions')
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--adapter', choices=['icl','official'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--base-model', type=Path)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device-map', choices=['single','auto'], default='single')
    p.add_argument('--official-frames', type=int, choices=[8,16,32], default=32,
                   help='Whole-trajectory frames; 32=direct outputs, 8/16=uniform sampling + linear interpolation')
    p.add_argument('--limit', type=int, help='Smoke run: only this many pending trajectories; cannot score incomplete output')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Print run configuration without loading weights')
    p = sub.add_parser('score', help='Stage 2: compare strict three-way relations on frozen pairs')
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('validate', help='Validate external model predictions against the shared schema')
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.data_root, args.output, seed=args.seed, threshold=args.threshold,
                         within_per_trajectory=args.within_per_trajectory, cross_per_group=args.cross_per_group,
                         pair_scope=args.pair_scope, exclude_known_conflicts=args.exclude_known_conflicts)
        print(json.dumps({'benchmark_id':result['benchmark_id'], **result['counts']}, indent=2))
    elif args.command == 'predict':
        run_predictions(args)
    elif args.command == 'score':
        result = score(args.benchmark, args.predictions, args.output)
        print(json.dumps({'overall':result['overall'], 'by_mode':result['by_mode']}, indent=2))
    else:
        info, rows = load_benchmark(args.benchmark)
        result = validate_predictions(info, rows, read_jsonl(args.predictions))
        print(f'Valid: {len(result)} trajectories, {len(result)*32} frames')


if __name__ == '__main__':
    main()
