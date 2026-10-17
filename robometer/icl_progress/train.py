"""Standalone Accelerate training: one soft-target ICL loss, no RBM trainer."""
import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from scripts.launch_b200 import disable_network_proxies
from .config import load_config
from .data import TrajectoryStore, ICLBatchDataset
from .collator import ICLCollator
from .model import load_model


@dataclass
class Cursor:
    updates: int = 0
    consumed_batches: int = 0
    skipped_updates: int = 0

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, values):
        for key, value in values.items():
            setattr(self, key, value)


def log_event(accelerator, output, values, step=None):
    if not accelerator.is_main_process:
        return
    print('[ICL] ' + json.dumps(values, ensure_ascii=False), flush=True)
    with (output / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(values, ensure_ascii=False) + '\n')
    if step is not None and accelerator.trackers:
        accelerator.log(values, step=step)


@torch.no_grad()
def evaluate(model, loader, accelerator):
    model.eval()
    totals = torch.zeros(4, device=accelerator.device)
    for batch in loader:
        out = model(**batch)
        probs = out['logits'].float().softmax(-1)
        k = batch['anchor_mask'].sum(-1)
        anchors = torch.arange(probs.shape[1], device=probs.device)[None, :] / (k[:, None] - 1)
        predicted = (probs * anchors).sum(-1)
        target = (batch['labels'] * anchors).sum(-1)
        hits = batch['labels'].gather(1, probs.argmax(-1, keepdim=True)).squeeze(-1) > 0
        n = len(probs)
        totals += torch.stack([out['loss'].float() * n, (predicted - target).abs().sum(), hits.float().sum(),
                               torch.tensor(float(n), device=probs.device)])
    totals = accelerator.reduce(totals, reduction='sum').cpu().tolist()
    model.train()
    return {'eval/loss': totals[0] / totals[3], 'eval/progress_mae': totals[1] / totals[3],
            'eval/adjacent_anchor_accuracy': totals[2] / totals[3], 'eval/queries': int(totals[3])}


def audit_fp16(model, optimizer, accelerator):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    for name, p in model.named_parameters():
        if p.is_floating_point() and p.dtype != torch.float16:
            raise RuntimeError(f'397b parameter is not FP16: {name}: {p.dtype}')
    if accelerator.scaler is not None or accelerator.mixed_precision != 'no':
        raise RuntimeError('397b must not use AMP GradScaler or FP32 master parameters')
    blocks = {id(m) for m in model.modules() if type(m).__name__ in
              {'Qwen3_5MoeDecoderLayer', 'Qwen3_5MoeVisionBlock'}}
    wrapped = {id(m._checkpoint_wrapped_module) for m in model.modules() if isinstance(m, CheckpointWrapper)}
    if not blocks or not blocks.issubset(wrapped):
        raise RuntimeError('397b requires activation checkpoints on every language and vision block')
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state and state[key].dtype != torch.float16:
                raise RuntimeError(f'397b optimizer {key} is not FP16')
    return len(blocks)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', choices=['4b', '27b', '397b'], required=True)
    args, overrides = parser.parse_known_args()
    if any('=' not in s for s in overrides):
        parser.error('Overrides must be dotted key=value')
    disable_network_proxies(os.environ)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    cfg = load_config(args.model, overrides)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    torch.set_num_interop_threads(2)
    pure = cfg.model.precision == 'pure_fp16'
    # FSDP's plugin captures current_device in its meta materialization closure
    # before Accelerator finishes distributed state initialization.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    accelerator = Accelerator(
        mixed_precision='no' if pure else 'bf16', gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False),
        log_with='wandb' if cfg.logging.wandb else None, step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False),
                         InitProcessGroupKwargs(timeout=timedelta(minutes=30))])
    if pure and not (accelerator.state.fsdp_plugin and accelerator.state.fsdp_plugin.activation_checkpointing):
        raise ValueError('397b requires the ICL multi/single launcher with FSDP activation checkpointing')
    set_seed(cfg.train.seed)
    output = Path(cfg.train.output_dir)
    if output.exists() and any(output.iterdir()) and not cfg.train.resume:
        raise ValueError(f'Output directory already exists: {output}; choose a new RUN_NAME or resume')
    accelerator.wait_for_everyone()
    output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output / ('resume_config.yaml' if cfg.train.resume else 'config.yaml'))
    if cfg.logging.wandb:
        if accelerator.is_main_process and not os.environ.get('WANDB_API_KEY'):
            key = Path(os.environ.get('WANDB_API_KEY_FILE', '.secrets/wandb_api_key'))
            os.environ['WANDB_API_KEY'] = key.read_text().strip()
        accelerator.init_trackers(cfg.logging.project, config=OmegaConf.to_container(cfg, resolve=True),
                                  init_kwargs={'wandb': {'name': output.name}})
    log_event(accelerator, output, {'phase': 'data_loading', 'model': args.model})
    store = TrajectoryStore(cfg.data)
    if accelerator.is_main_process:
        store.save_summary(output / 'dataset_summary.json')
    log_event(accelerator, output, {'phase': 'model_loading', 'train_trajectories': sum(map(len, store.indices['train'].values())),
                                   'validation_trajectories': sum(map(len, store.indices['validation'].values()))})
    model, processor = load_model(cfg.model)
    log_event(accelerator, output, {'phase': 'model_loaded',
               'parameters': sum(p.numel() for p in model.parameters()),
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'precision': cfg.model.precision, 'freeze_vision': cfg.model.freeze_vision})
    # Meta materialization on nonzero FSDP ranks replaces Parameter objects.
    # Build AdamW only AFTER wrapping so every rank optimizes live parameters.
    model = accelerator.prepare_model(model)
    epsilon = 1e-4 if pure else 1e-8
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay,
                                 eps=epsilon, foreach=False if pure else None, fused=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(cfg.train.max_steps * cfg.train.warmup_ratio),
                                               cfg.train.max_steps)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    live = {id(p) for p in model.parameters() if p.requires_grad}
    bound = {id(p) for group in optimizer.param_groups for p in group['params']}
    if live != bound or any(p.is_meta for group in optimizer.param_groups for p in group['params']):
        raise RuntimeError('Optimizer parameters must exactly match live, materialized trainable parameters')
    cursor = Cursor()
    accelerator.register_for_checkpointing(cursor)
    if cfg.train.resume:
        checkpoint = Path(cfg.train.resume)
        meta = json.loads((checkpoint / 'icl_state.json').read_text())
        for name, expected in [('world_size', accelerator.num_processes), ('batch_size', cfg.train.batch_size),
                               ('accumulation', cfg.train.gradient_accumulation_steps)]:
            if meta[name] != expected:
                raise ValueError(f'Resume requires unchanged {name}')
        if meta['model'] != args.model:
            raise ValueError('Cannot resume a different backbone')
        saved = OmegaConf.load(checkpoint / 'icl_config.yaml')
        for section in ('model', 'data'):
            if OmegaConf.to_container(saved[section]) != OmegaConf.to_container(cfg[section]):
                raise ValueError(f'Resume requires unchanged {section} configuration')
        if saved.train.seed != cfg.train.seed or saved.train.max_steps != cfg.train.max_steps:
            raise ValueError('Resume requires unchanged seed/max_steps to preserve sampling and LR schedule')
        accelerator.load_state(str(checkpoint))
    if pure:
        count = audit_fp16(model, optimizer, accelerator)
        log_event(accelerator, output, {'phase': 'precision_audit', 'checkpointed_blocks': count, 'dtype': 'float16'})
    if cursor.updates >= cfg.train.max_steps:
        log_event(accelerator, output, {'phase': 'already_complete', **cursor.state_dict()})
        accelerator.end_training()
        return
    collator = ICLCollator(processor, cfg.data.root, cfg.data.max_keyframes, cfg.data.max_pixels)
    world = accelerator.num_processes
    length = (cfg.train.max_steps * cfg.train.gradient_accumulation_steps + cfg.train.max_skipped_updates *
              cfg.train.gradient_accumulation_steps + 1) * world
    train_data = ICLBatchDataset(store, cfg.data, cfg.train.batch_size, length, cfg.train.seed,
                                 offset=cursor.consumed_batches * world)
    options = dict(batch_size=1, collate_fn=collator, num_workers=cfg.data.workers,
                   pin_memory=True, persistent_workers=cfg.data.workers > 0)
    loader = accelerator.prepare(DataLoader(train_data, shuffle=False, **options))
    eval_loader = None
    if cfg.train.eval_steps > 0:
        eval_data = ICLBatchDataset(store, cfg.data, cfg.train.batch_size, cfg.train.eval_batches * world,
                                   cfg.train.seed + 100000, split='validation')
        eval_loader = accelerator.prepare(DataLoader(eval_data, shuffle=False, **options))
    if accelerator.is_main_process:
        processor.save_pretrained(output / 'processor')
    log_event(accelerator, output, {'phase': 'training', 'world_size': world, 'batch_per_gpu': cfg.train.batch_size,
                                   'global_batch': world * cfg.train.batch_size * cfg.train.gradient_accumulation_steps,
                                   'max_keyframes': cfg.data.max_keyframes, 'loss': 'anchor_soft_cross_entropy'})

    def save():
        path = output / f'checkpoint-{cursor.updates}'
        accelerator.wait_for_everyone()
        accelerator.save_state(str(path))
        if accelerator.is_main_process:
            OmegaConf.save(cfg, path / 'icl_config.yaml')
            processor.save_pretrained(path / 'processor')
            (path / 'icl_state.json').write_text(json.dumps({**cursor.state_dict(), 'world_size': world,
                'batch_size': cfg.train.batch_size, 'accumulation': cfg.train.gradient_accumulation_steps,
                'model': args.model}, indent=2))
        accelerator.wait_for_everyone()
        log_event(accelerator, output, {'phase': 'checkpoint', 'step': cursor.updates, 'path': str(path)})

    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_sum = torch.zeros((), device=accelerator.device)
    bad_window = torch.zeros((), dtype=torch.int32, device=accelerator.device)
    step_start = time.monotonic()
    last_saved = -1
    for batch in loader:
        if cursor.updates >= cfg.train.max_steps:
            break
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs['loss']
            loss_sum += loss.detach().float()
            bad_window |= (~torch.isfinite(loss.detach())).int()
            accelerator.backward(loss)
            cursor.consumed_batches += 1
            if not accelerator.sync_gradients:
                continue
            norm = accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            bad_window |= (~torch.isfinite(norm)).int()
            bad = accelerator.reduce(bad_window, reduction='sum').item() > 0
            if bad:
                cursor.skipped_updates += 1
                log_event(accelerator, output, {'phase': 'skipped_nonfinite', **cursor.state_dict()})
                optimizer.zero_grad(set_to_none=True)
                loss_sum.zero_(); bad_window.zero_()
                if cursor.skipped_updates > cfg.train.max_skipped_updates:
                    raise RuntimeError('Exceeded nonfinite update limit')
                step_start = time.monotonic()
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            cursor.updates += 1
        mean_loss = accelerator.reduce(loss_sum, reduction='mean').item() / cfg.train.gradient_accumulation_steps
        if cursor.updates == 1:
            state_numel = sum(state['exp_avg'].numel() for state in optimizer.state.values() if 'exp_avg' in state)
            param_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
            counts = accelerator.gather(torch.tensor([[param_numel, state_numel]], device=accelerator.device)).cpu().tolist()
            if any(params == 0 or moments != params for params, moments in counts):
                raise RuntimeError(f'AdamW state coverage does not match trainable parameter shards: {counts}')
            log_event(accelerator, output, {'phase': 'optimizer_audit', 'parameter_and_moment_numel_by_rank': counts})
            if pure:
                audit_fp16(model, optimizer, accelerator)
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=accelerator.device)
        peaks = accelerator.gather(peak.reshape(1)).cpu().tolist()
        log_event(accelerator, output, {'phase': 'update', 'step': cursor.updates, 'train/loss': mean_loss,
                    'train/grad_norm': float(norm), 'train/lr': scheduler.get_last_lr()[0],
                    'train/seconds': time.monotonic() - step_start, 'peak_allocated_gib_by_rank': peaks,
                    'skipped_updates': cursor.skipped_updates}, step=cursor.updates)
        loss_sum.zero_(); bad_window.zero_()
        if eval_loader is not None and cursor.updates % cfg.train.eval_steps == 0:
            log_event(accelerator, output, {'phase': 'eval', 'step': cursor.updates,
                       **evaluate(model, eval_loader, accelerator)}, step=cursor.updates)
        if cfg.train.save_steps > 0 and cursor.updates % cfg.train.save_steps == 0:
            save(); last_saved = cursor.updates
        step_start = time.monotonic()
    if cursor.updates < cfg.train.max_steps:
        raise RuntimeError('Data stream ended before the requested successful updates')
    if cfg.train.save_steps > 0 and last_saved != cursor.updates:
        save()
    log_event(accelerator, output, {'phase': 'complete', **cursor.state_dict()})
    accelerator.end_training()


if __name__ == '__main__':
    main()
