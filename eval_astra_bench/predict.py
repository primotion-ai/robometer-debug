"""Standalone prediction with a private CPU pool and bounded image prefetch.

The legacy CLI remains unchanged for the in-flight, source-pinned queue.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

from .core import (DEFAULT_DATA, PREDICTION_SCHEMA, file_hash, load_benchmark, prediction_row,
                   read_jsonl, validate_predictions, write_json, canonical)
from .prefetch import ImagePrefetcher, configure_cpu


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
                              model_loaded=False, io_settings=args.io_settings), ensure_ascii=False, indent=2))
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
        from .loaded_inference import predict_icl, predict_official
        torch.set_num_threads(args.cpu_cores)
        with args.output.with_suffix('.io.jsonl').open('a') as io_log:
            io_log.write(canonical(args.io_settings) + '\n')
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        trajectory_batch_size = args.batch_size if args.adapter == 'official' else 1
        with ImagePrefetcher(root, adapter=args.adapter, workers=args.io_workers,
                             prefetch_batches=args.prefetch_batches, official_frames=args.official_frames) as loader, \
                args.output.open('a') as stream:
            start = 0
            for loaded_batch in loader.batches(pending, trajectory_batch_size):
                batch = [loaded.row for loaded in loaded_batch]
                if args.adapter == 'official':
                    results = predict_official(adapter, loaded_batch)
                else:
                    results = [(predict_icl(adapter, loaded_batch[0]), None)]
                for offset, (row, (progress, details)) in enumerate(zip(batch, results)):
                    result = prediction_row(info, row, progress)
                    if details is not None:
                        result['adapter_details'] = details
                    stream.write(canonical(result) + '\n')
                    stream.flush()
                    print(f"Predicted {len(done)+start+offset+1}/{len(rows)}: {row['trajectory_id']}", flush=True)
                start += len(batch)
        elapsed = time.monotonic()-started
        write_json(args.output.with_suffix('.runtime.json'), dict(
            benchmark_id=info['benchmark_id'], newly_predicted_trajectories=len(pending),
            completed_trajectories=len(done)+len(pending), prediction_seconds=elapsed,
            trajectories_per_second=len(pending)/elapsed,
            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            requested_batch_size=args.batch_size, io_settings=args.io_settings,
            batch_unit='trajectories' if args.adapter == 'official' else 'query_frames'))
        all_rows = read_jsonl(args.output)
        validate_predictions(info, rows, all_rows, allow_partial=True)
        print(f'Saved {len(all_rows)}/{len(rows)} trajectories: {args.output}', flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--adapter', choices=['icl', 'official'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device-map', choices=['single', 'auto'], default='single')
    parser.add_argument('--official-frames', type=int, choices=[8, 16, 32], default=32)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cpu-cores', type=int, default=8)
    parser.add_argument('--cpu-slot', type=int, help='Disjoint CPU core group; defaults to first numeric CUDA_VISIBLE_DEVICES ID, otherwise 0')
    parser.add_argument('--io-workers', type=int, default=8)
    parser.add_argument('--prefetch-batches', type=int, default=2, help='Future batches to preload; 0 disables read-ahead')
    args = parser.parse_args()
    if args.io_workers < 1 or args.prefetch_batches < 0:
        parser.error('Reader workers must be positive and prefetch batches nonnegative')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    slot = args.cpu_slot if args.cpu_slot is not None else (int(visible) if visible.isdigit() else 0)
    try:
        cpus = configure_cpu(args.cpu_cores, slot)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.io_settings = dict(cpu_cores=args.cpu_cores, cpu_ids=cpus, cpu_slot=slot,
                            io_workers=args.io_workers, prefetch_batches=args.prefetch_batches,
                            prefetch_unit='trajectories' if args.adapter == 'icl' else 'trajectory_batches')
    print('Per-process CPU / data loading: ' + canonical(args.io_settings), flush=True)
    run_predictions(args)


if __name__ == '__main__':
    main()
