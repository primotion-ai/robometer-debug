"""Use mounted uv environment; compatible with one platform command per node."""
import argparse
import os
from pathlib import Path
import shlex
import sys
from datetime import datetime, timezone

from scripts.launch_b200 import disable_network_proxies, resolve_topology
from .config import load_config

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['single', 'multi', 'debug'])
    parser.add_argument('model', choices=['4b', '27b', '397b'])
    parser.add_argument('--dry-run', action='store_true')
    args, overrides = parser.parse_known_args()
    if any('=' not in s or s.startswith('--') for s in overrides):
        parser.error('Overrides must be dotted key=value')
    disable_network_proxies(os.environ)
    env = os.environ.copy()
    if args.mode == 'multi' and args.model == '27b':
        # Default to four nodes; explicit platform topology remains authoritative.
        env.setdefault('NNODES', '4')
    nodes, gpus, rank = resolve_topology(args.mode, env)
    if args.mode == 'multi' and (not env.get('MASTER_ADDR') or not env.get('RUN_NAME')):
        parser.error('Multi-node requires MASTER_ADDR and the same RUN_NAME on all nodes')
    name = env.get('RUN_NAME', f'icl_{args.model}_{args.mode}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in name):
        parser.error('RUN_NAME must be a simple directory name')
    runtime = [f'train.output_dir=logs/{name}']
    if args.mode == 'debug':
        runtime += ['train.max_steps=3', 'train.batch_size=1', 'data.workers=2']
    runtime += overrides
    if args.mode == 'debug':
        runtime += ['logging.wandb=false', 'train.save_steps=0', 'train.eval_steps=0']
        env['WANDB_MODE'] = 'disabled'
        env.pop('WANDB_API_KEY', None)
    cfg = load_config(args.model, runtime)
    for key, value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                       'OMP_NUM_THREADS':'8','USE_TF':'0','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}.items():
        env.setdefault(key, value)
    env['WANDB_DISABLE_CODE'] = 'true'
    env['WANDB_IGNORE_GLOBS'] = '.secrets/*,**/.secrets/*'
    if not cfg.logging.wandb:
        env['WANDB_MODE'] = 'disabled'
        env.pop('WANDB_API_KEY', None)
    else:
        env['WANDB_MODE'] = 'online'
    pure = cfg.model.precision == 'pure_fp16'
    distributed = 'ddp.yaml' if args.model == '4b' else ('b200_397b_fp16_fsdp.yaml' if pure else 'b200_fsdp.yaml')
    offload = env.get('FSDP_OFFLOAD_PARAMS', 'true' if pure and nodes == 1 else 'false').lower()
    if offload not in ('true','false'):
        parser.error('FSDP_OFFLOAD_PARAMS must be true or false')
    if offload == 'true' and cfg.train.gradient_accumulation_steps != 1:
        parser.error('FSDP CPU offload requires gradient_accumulation_steps=1')
    command = [sys.executable, '-m', 'accelerate.commands.launch',
               '--config_file', str(ROOT/'robometer/configs/distributed'/distributed),
               '--num_processes', str(nodes*gpus), '--num_machines', str(nodes), '--machine_rank', str(rank),
               '--main_process_ip', env.get('MASTER_ADDR','127.0.0.1'),
               '--main_process_port', env.get('MASTER_PORT','29501'),
               '--num_cpu_threads_per_process', env['OMP_NUM_THREADS']]
    if args.model != '4b':
        layer = 'Qwen3_5Moe' if pure else 'Qwen3_5'
        command += ['--fsdp_transformer_layer_cls_to_wrap', f'{layer}DecoderLayer,{layer}VisionBlock',
                    '--fsdp_offload_params', offload]
    command += ['--module', 'robometer.icl_progress.train', '--model', args.model, *runtime]
    print(f'ICL {args.mode}: {nodes}x{gpus}; rank={rank}; proxies disabled; W&B={cfg.logging.wandb}', flush=True)
    print(f'Per-GPU batch={cfg.train.batch_size}, initial candidates={cfg.train.batch_size+1}, '
          f'global batch={nodes*gpus*cfg.train.batch_size*cfg.train.gradient_accumulation_steps}; '
          f'keyframes={cfg.data.min_keyframes}..{cfg.data.max_keyframes}', flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
