from pathlib import Path

from omegaconf import OmegaConf


def load_config(model, overrides=()):
    root = Path(__file__).parent / 'configs'
    cfg = OmegaConf.merge(OmegaConf.load(root / 'base.yaml'), OmegaConf.load(root / f'{model}.yaml'))
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg.model.name = model
    if not 2 <= cfg.data.min_keyframes <= cfg.data.max_keyframes <= 11:
        raise ValueError('Require 2 <= min_keyframes <= max_keyframes <= 11')
    if not 0 < cfg.data.validation_fraction < 1:
        raise ValueError('validation_fraction must be strictly between 0 and 1')
    if cfg.train.batch_size < 1 or cfg.train.gradient_accumulation_steps < 1 or cfg.train.max_steps < 1:
        raise ValueError('Batch size, accumulation and max_steps must be positive')
    if cfg.data.workers < 0 or cfg.data.max_candidate_rounds < 1:
        raise ValueError('workers must be nonnegative and max_candidate_rounds positive')
    if cfg.data.max_trajectories_per_source is not None and cfg.data.max_trajectories_per_source < 1:
        raise ValueError('max_trajectories_per_source must be positive or null')
    if cfg.train.eval_steps < 0 or cfg.train.save_steps < 0 or cfg.train.eval_batches < 1:
        raise ValueError('eval/save_steps must be nonnegative and eval_batches positive')
    if not 0 <= cfg.train.warmup_ratio <= 1 or cfg.train.learning_rate <= 0:
        raise ValueError('Require warmup_ratio in [0,1] and positive learning_rate')
    if cfg.data.candidate_extra != 1:
        raise ValueError('Each batch must start with B+1 candidates')
    if cfg.model.precision not in ('bf16', 'pure_fp16'):
        raise ValueError('precision must be bf16 or pure_fp16')
    if cfg.model.precision == 'pure_fp16' and model != '397b':
        raise ValueError('pure_fp16 is isolated to 397b')
    if cfg.model.precision == 'pure_fp16' and not cfg.model.gradient_checkpointing:
        raise ValueError('397b pure FP16 requires full FSDP activation checkpointing')
    return cfg
