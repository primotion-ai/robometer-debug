"""Run the existing benchmark entry point for official 4B and every 003 checkpoint."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--models', help='Optional comma-separated model names to run')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    models = [('official4b', '/mnt/base_model/Robometer-4B', '/mnt/base_model/Qwen3-VL-4B-Instruct')]
    models += [(f'qwen27b_step{s}', str(ROOT / f'logs/rbm_qwen27b_5nodes_003/checkpoint-{s}'),
                '/mnt/base_model/Qwen3.8-27B') for s in ([1000] if args.smoke else [1000, 2000, 3000, 4000, 5000])]
    for model in models:
        if not args.models or model[0] in args.models.split(','):
            jobs.put(model)

    def worker(gpu):
        while True:
            try:
                name, checkpoint, base = jobs.get_nowait()
            except queue.Empty:
                return
            dest = output / name
            dest.mkdir(exist_ok=True)
            if (dest / 'COMPLETE.json').exists():
                print(f'SKIP completed {name}', flush=True)
                continue
            cmd = [sys.executable, 'robometer/evals/run_baseline_eval.py', '--config-name', 'rbm_comparison',
                   f'model_path={checkpoint}', f'model_config.base_model_path={base}', f'output_dir={dest}']
            if args.smoke:
                dataset = 'aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking'
                cmd += [f'custom_eval.{kind}=[{dataset}]' for kind in
                        ['reward_alignment', 'policy_ranking', 'quality_preference', 'confusion_matrix']]
                cmd += ['custom_eval.reward_alignment_max_trajectories=2', 'custom_eval.policy_ranking_max_tasks=2',
                        'custom_eval.num_examples_per_quality_pr=2', 'custom_eval.max_comparisons=2',
                        'custom_eval.confusion_matrix_n_trajectories_per_source=2']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       ROBOMETER_PROCESSED_DATASETS_PATH='/mnt/dataset/robometer-1m', OMP_NUM_THREADS='8',
                       MKL_NUM_THREADS='8', MPLBACKEND='Agg', PYTHONUNBUFFERED='1', PYTHONHASHSEED='42',
                       TOKENIZERS_PARALLELISM='false')
            started = time.time()
            (dest / 'invocation.json').write_text(json.dumps({'command': cmd, 'gpu': gpu, 'started': started}, indent=2))
            print(f'START {name} on GPU {gpu}', flush=True)
            with (dest / 'run.log').open('w') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            report = {'exit_code': result.returncode, 'seconds': time.time() - started}
            success = result.returncode == 0 and (dest / 'all_metrics.json').exists()
            (dest / ('COMPLETE.json' if success else 'FAILED.json')).write_text(json.dumps(report, indent=2))
            print(f'{"COMPLETE" if success else "FAILED"} {name}: {report}', flush=True)
            if not success:
                raise RuntimeError(f'{name} failed; see {dest / "run.log"}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus.split(','))) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus.split(',')]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
