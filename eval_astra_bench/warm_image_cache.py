"""Warm the host's shared file cache without changing inference or source images.

Run alongside an existing prediction queue. Eight bounded readers visit upcoming
trajectories first, then the earlier ones for subsequent checkpoints. This is a
best-effort cache warmup, not a persistent cache or a substitute for prefetching.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .core import file_hash, load_benchmark, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--queue-status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    info, rows = load_benchmark(args.benchmark)
    root = Path(info['data_root']).resolve()
    if file_hash(root/'checksums.sha256') != info['source_checksums_sha256']:
        raise ValueError('Source release changed')
    status = json.loads(args.queue_status.read_text())
    if status['benchmark_id'] != info['benchmark_id']:
        raise ValueError('Queue uses a different benchmark')
    completed = [j.get('trajectories', 0) for j in status['jobs'].values() if j['state'] == 'running']
    start = min(completed, default=0)
    order = rows[start:] + rows[:start]
    seen, errors = set(), []
    total_bytes = 0
    started = time.monotonic()

    def read(relative):
        try:
            size = 0
            with (root/relative).open('rb') as stream:
                while chunk := stream.read(1024*1024):
                    size += len(chunk)
            return relative, size, None
        except OSError as exc:
            return relative, 0, str(exc)

    def report(state, trajectories):
        elapsed = time.monotonic()-started
        data = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(),
                    benchmark_id=info['benchmark_id'], data_root=str(root), workers=args.workers,
                    start_row=start, visited_trajectories=trajectories, read_files=len(seen),
                    bytes_read=total_bytes, elapsed_seconds=round(elapsed, 2), errors=errors)
        write_json(args.output, data)
        print(json.dumps(data, ensure_ascii=False), flush=True)

    report('running', 0)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(order, 1):
            paths = row['image_paths'] + row['icl_reference']['image_paths']
            pending = list(dict.fromkeys(p for p in paths if p not in seen))
            for relative, size, error in pool.map(read, pending):
                if error:
                    errors.append(dict(path=relative, error=error))
                else:
                    seen.add(relative)
                    total_bytes += size
            if index % 10 == 0:
                report('running', index)
    report('complete' if not errors else 'completed_with_errors', len(order))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
