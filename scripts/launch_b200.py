#!/usr/bin/env python3
"""Shared, offline-first launcher for the single-node, multi-node and debug scripts."""

import argparse
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def disable_network_proxies(env):
    """Use direct connections in preflight and every spawned training worker."""
    proxy_keys = {
        "http_proxy", "https_proxy", "all_proxy", "ftp_proxy",
        "wandb_http_proxy", "wandb_https_proxy", "wandb_x_proxies",
    }
    for key in list(env):
        if key.lower() in proxy_keys:
            del env[key]
    # Both Python HTTP clients and the W&B Go service must bypass proxies.
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"


def resolve_topology(mode, env, *, visible_gpu_count=None):
    """Resolve node-level platform variables; auto mode queries CUDA visibility.

    In multi mode WORLD_SIZE, when supplied, means total GPU workers, as in the
    platform contract. Do not infer nodes from MPI NP or an ambiguous WORLD_SIZE.
    """
    def integer(name, default=None):
        value = env.get(name, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be an integer; got {value!r}") from None

    nodes, rank = 1, 0
    gpus = integer("GPUS_PER_NODE", "8")
    if mode == "multi":
        if "LOCAL_RANK" in env:
            raise ValueError("Launch this entrypoint once per NODE, not once per GPU; the script starts GPU workers itself")
        if "OMPI_COMM_WORLD_RANK" in env:
            raise ValueError("This entrypoint expects one platform command per node, not an mpirun GPU worker; select the platform's PyTorch/custom per-node entrypoint")
        nodes = integer("NNODES", "5")
        rank_key = next((key for key in ("NODE_RANK", "GROUP_RANK", "RANK", "SLURM_PROCID") if key in env), None)
        if rank_key is None:
            raise ValueError("Multi-node launch requires NODE_RANK, GROUP_RANK or platform RANK in [0, NNODES)")
        rank = integer(rank_key)
        if "NPROC_PER_NODE" in env:
            requested = env["NPROC_PER_NODE"].strip().lower()
            if requested in {"auto", "gpu"}:
                if "GPUS_PER_NODE" in env:
                    # An explicit GPU count can narrow an automatic allocation.
                    # The normal preflight still checks available CUDA devices.
                    platform_gpus = gpus
                else:
                    if visible_gpu_count is None:
                        import torch

                        visible_gpu_count = torch.cuda.device_count()
                    platform_gpus = visible_gpu_count
                print(f"NPROC_PER_NODE={requested} resolved to {platform_gpus} GPU workers per node", flush=True)
            else:
                platform_gpus = integer("NPROC_PER_NODE")
            if "GPUS_PER_NODE" in env and gpus != platform_gpus:
                raise ValueError(f"GPUS_PER_NODE={gpus} conflicts with NPROC_PER_NODE={platform_gpus}; set both to the allocated GPUs per node, or omit GPUS_PER_NODE")
            gpus = platform_gpus
        if "WORLD_SIZE" in env:
            world = integer("WORLD_SIZE")
            if world != nodes * gpus:
                raise ValueError(f"WORLD_SIZE={world} conflicts with NNODES={nodes} x GPUs per node={gpus} = {nodes * gpus}; configure the platform's total GPU worker count consistently")
    if nodes < 1 or not 0 <= rank < nodes:
        raise ValueError("Require NNODES >= 1 and 0 <= NODE_RANK < NNODES")
    if gpus < 2:
        raise ValueError(f"These FSDP launchers require at least 2 GPUs per node; resolved {gpus}. Check container GPU visibility and NPROC_PER_NODE/GPUS_PER_NODE; auto does not fall back to CPU workers")
    return nodes, gpus, rank


def compose_config(name, overrides):
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from robometer.configs.experiment_configs import ExperimentConfig

    ConfigStore.instance().store(name="base_config", node=ExperimentConfig)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "robometer/configs")):
        return compose(config_name=name, overrides=overrides)


def check_weights(model_path):
    """Check all indexed shards, including truncated files, without reading tensor data."""
    path = Path(model_path)
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    elif (path / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    else:
        raise ValueError(f"Model download incomplete: no safetensors index or single weight file in {path}")
    missing = [s for s in shards if not (path / s).is_file()]
    if missing:
        raise ValueError(f"Model download incomplete: {len(missing)}/{len(shards)} shards missing; first: {missing[0]}")
    for shard in shards:
        file = path / shard
        with file.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated shard: {file}")
            header_size = struct.unpack("<Q", raw)[0]
            if not 0 < header_size < 100_000_000:
                raise ValueError(f"Invalid safetensors header: {file}")
            header = json.loads(stream.read(header_size))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if file.stat().st_size != 8 + header_size + end:
            raise ValueError(f"Truncated or inconsistent shard: {file}")
    print(f"Weights complete: {len(shards)} safetensors shards", flush=True)


def check_data(cfg, root):
    from datasets import Dataset
    from robometer.data.datasets.base import resolve_dataset_keys
    from robometer.data.datasets.helpers import resolve_processed_path

    requested = list(resolve_dataset_keys(cfg.data.train_datasets, "train"))
    if cfg.training.do_eval:
        requested.extend(resolve_dataset_keys(cfg.data.eval_datasets, "eval"))
    for kind in cfg.custom_eval.eval_types:
        requested.extend(resolve_dataset_keys(getattr(cfg.custom_eval, kind), "eval"))
    missing = []
    for name in dict.fromkeys(requested):
        folder = Path(root) / name.replace("/", "_").replace(":", "_")
        if not (folder / "dataset_info.json").is_file() or not (folder / "processed_dataset").is_dir():
            missing.append(str(folder))
            continue
        if not (folder / "index_mappings.json").is_file():
            missing.append(str(folder / "index_mappings.json"))
        dataset = Dataset.load_from_disk(str(folder / "processed_dataset"))
        if not len(dataset):
            raise ValueError(f"Empty processed dataset: {folder}")
        field = "embeddings_path" if cfg.data.load_embeddings else "frames"
        # One actual trajectory per subset catches relocation errors early;
        # this is not an exhaustive integrity scan of the entire million rows.
        frame_file = dataset[0].get(field)
        if isinstance(frame_file, str) and not Path(resolve_processed_path(frame_file, root)).is_file():
            missing.append(resolve_processed_path(frame_file, root))
    if missing:
        raise ValueError(f"Processed datasets missing ({len(missing)}):\n" + "\n".join(missing))
    print(f"Dataset caches available: {len(set(requested))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["single", "multi", "debug"])
    parser.add_argument("model", choices=["4b", "27b", "397b"])
    parser.add_argument("--dry-run", action="store_true", help="Print command/config; do not load keys or run preflight/training")
    parser.add_argument("--check-only", action="store_true", help="Check environment, model shards, processor and data; no training")
    args, overrides = parser.parse_known_args()
    if any("=" not in item or item.startswith("--") for item in overrides):
        parser.error("Extra arguments must be Hydra key=value overrides")

    if args.model == "4b" and args.mode == "multi":
        parser.error("The 4b recipe currently supports single-node and debug launch only")

    # All single/multi/debug shell entrypoints share this launcher. Sanitize
    # before importing network clients; workers inherit the same clean env.
    disable_network_proxies(os.environ)
    print("[robometer launcher] Network proxies disabled; using direct connections", flush=True)
    env = os.environ.copy()
    check_numerics = env.get("ROBOMETER_CHECK_NUMERICS") == "1"
    trace_progress = env.get("ROBOMETER_TRACE_PROGRESS") == "1"
    diagnostic = check_numerics or trace_progress
    if diagnostic and args.model != "397b":
        parser.error("397b diagnostics require the 397b pure FP16 recipe")
    try:
        nodes, gpus, rank = resolve_topology(args.mode, env)
    except ValueError as error:
        parser.error(str(error))
    master = env.get("MASTER_ADDR", "")
    if args.mode == "multi" and not master and env.get("SLURM_JOB_NODELIST"):
        master = subprocess.check_output(["scontrol", "show", "hostnames", env["SLURM_JOB_NODELIST"]], text=True).splitlines()[0]
    if args.mode == "multi" and not master:
        parser.error("Set MASTER_ADDR to the rank-0 node address reachable from every node")
    master = master or "127.0.0.1"
    port = env.get("MASTER_PORT", "29500")
    try:
        if not 1 <= int(port) <= 65535:
            raise ValueError
    except ValueError:
        parser.error("MASTER_PORT must be an integer in [1, 65535]")
    run_name = env.get("RUN_NAME")
    if not run_name:
        if args.mode == "multi":
            if not env.get("SLURM_JOB_ID"):
                parser.error("Set the same RUN_NAME on all nodes")
            suffix = env["SLURM_JOB_ID"]
        else:
            suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"qwen{args.model}_{args.mode}_{suffix}"
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for c in run_name):
        parser.error("RUN_NAME must contain only letters, digits, dots, underscores or hyphens")

    env.setdefault("ROBOMETER_PROCESSED_DATASETS_PATH", "/mnt/dataset/robometer-1m")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env["WANDB_DISABLE_CODE"] = "true"
    env["WANDB_IGNORE_GLOBS"] = ".secrets/*,**/.secrets/*"

    runtime = [f"training.exp_name={run_name}"]
    if diagnostic:
        # Preserve the real batch/images/topology while locating a first-step NaN.
        runtime.append(f"training.max_steps={3 if trace_progress else 1}")
    if args.mode == "debug":
        runtime += [
            f"training.max_steps={env.get('DEBUG_STEPS', '3')}",
            "training.per_device_train_batch_size=1",
            "training.gradient_accumulation_steps=1", "data.max_frames=2",
            "data.resized_height=128", "data.resized_width=128",
            "data.train_datasets=[aliangdw/metaworld:metaworld_train]",
            "data.dataloader_num_workers=0", "data.dataloader_persistent_workers=false",
        ]
    # Users can tune debug workload size, but cannot accidentally enable uploads or saves.
    runtime += overrides
    if args.mode == "debug" or diagnostic:
        env["WANDB_MODE"] = "disabled"
        env.pop("WANDB_API_KEY", None)
        runtime += [
            "debug=false", "logging.log_to=[]", "logging.wandb_mode=disabled",
            "logging.save_model=false", "logging.save_processor=false",
            "logging.save_best.save_every=null", "logging.save_best.metric_names=[]",
            "logging.save_best.greater_is_better=[]", "logging.save_best.upload_to_hub=false",
            "training.save_strategy=no", "training.do_eval=false",
            "training.evaluation_strategy=no", "custom_eval.eval_types=[]",
        ]
    else:
        env["WANDB_MODE"] = "online"
        runtime += ["logging.log_to=[wandb]", "logging.wandb_mode=online"]
        if env.get("WANDB_ENTITY"):
            runtime.append(f"logging.wandb_entity={env['WANDB_ENTITY']}")
        if env.get("WANDB_PROJECT"):
            runtime.append(f"logging.wandb_project={env['WANDB_PROJECT']}")

    cfg = compose_config(f"qwen{args.model}", runtime)
    if diagnostic and not cfg.training.pure_fp16:
        parser.error("397b diagnostics require the pure_fp16 recipe")
    fsdp_config_name = "ddp.yaml" if args.model == "4b" else "b200_fsdp.yaml"
    if cfg.training.pure_fp16:
        if args.model != "397b":
            parser.error("pure_fp16 is supported only by the isolated 397b recipe")
        from robometer.utils.pure_fp16 import validate_pure_fp16_config

        try:
            validate_pure_fp16_config(cfg)
        except ValueError as error:
            parser.error(str(error))
        fsdp_config_name = "b200_397b_fp16_fsdp.yaml"
    # CPU offload makes full 397B finetuning possible to attempt on a single node.
    offload = env.get("FSDP_OFFLOAD_PARAMS", "true" if args.model == "397b" and nodes == 1 else "false").lower()
    if offload not in {"true", "false"}:
        parser.error("FSDP_OFFLOAD_PARAMS must be true or false")
    if offload == "true" and cfg.training.gradient_accumulation_steps != 1:
        parser.error("CPU offload requires gradient_accumulation_steps=1 in this FSDP1 launcher")
    if args.model == "4b" and offload == "true":
        parser.error("4b DDP does not use FSDP CPU offload")
    layer = "Qwen3_5Moe" if args.model == "397b" else "Qwen3_5"
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--config_file", str(ROOT / "robometer/configs/distributed" / fsdp_config_name),
        "--num_machines", str(nodes), "--num_processes", str(nodes * gpus),
        "--machine_rank", str(rank), "--main_process_ip", master, "--main_process_port", port,
        "--num_cpu_threads_per_process", env["OMP_NUM_THREADS"],
        *([] if args.model == "4b" else ["--fsdp_transformer_layer_cls_to_wrap",
             f"{layer}DecoderLayer,{layer}VisionBlock", "--fsdp_offload_params", offload]),
        "scripts/profile_rank_entry.py" if args.mode == "debug" and env.get("ROBOMETER_PROFILE_NATIVE") == "1" else "train.py",
        "--config-name", f"qwen{args.model}", *runtime,
    ]
    print(f"{args.mode}: {nodes} nodes x {gpus} GPUs; node rank {rank}; CPU offload={offload}", flush=True)
    print(f"Python: {sys.executable}; rendezvous: {master}:{port}; total GPU workers: {nodes * gpus}", flush=True)
    print(f"Effective batch: {nodes * gpus * cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps} preference pairs/update", flush=True)
    if cfg.training.pure_fp16:
        print("397B pure FP16: FP16 parameters/gradients/AdamW moments; AMP off; FSDP block checkpointing on; AdamW epsilon=1e-4", flush=True)
    if cfg.training.skip_anomalous_updates:
        print(f"Anomalous batches: skip globally if loss > {cfg.training.skip_loss_threshold} or "
              "outputs/loss/gradient norm are non-finite; schedule/checkpoints count successful updates only", flush=True)
    if check_numerics:
        print("397B numeric diagnostics: per-module finite checks; W&B/evaluation/checkpoint saving disabled", flush=True)
    if trace_progress:
        print("397B progress diagnostics: rank heartbeat every 30s; Python stacks every 120s; W&B/evaluation/checkpoint saving disabled", flush=True)
    print(
        f"Training schedule: {cfg.training.max_steps} steps; eval every {cfg.training.eval_steps}; "
        f"custom eval every {cfg.training.custom_eval_steps}; "
        f"checkpoint strategy={cfg.training.save_strategy}, every {cfg.training.save_steps}; "
        f"retain {cfg.training.save_total_limit} checkpoints",
        flush=True,
    )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    import torch
    from transformers import AutoConfig, AutoProcessor

    print(f"PyTorch: {torch.__version__}; PyTorch CUDA runtime: {torch.version.cuda}; visible GPUs: {torch.cuda.device_count()}", flush=True)
    if torch.cuda.device_count() < gpus:
        raise ValueError(f"Requested {gpus} GPUs but only {torch.cuda.device_count()} are visible; check the platform GPU allocation")
    actual = AutoConfig.from_pretrained(cfg.model.base_model_id, local_files_only=True)
    expected = {"4b": "qwen3_vl", "27b": "qwen3_5", "397b": "qwen3_5_moe"}[args.model]
    if actual.model_type != expected:
        raise ValueError(f"Expected {expected}; found {actual.model_type} in {cfg.model.base_model_id}")
    if cfg.model.linear_attention_backend == "fla":
        import importlib

        kernels = importlib.import_module(f"transformers.models.{expected}.modeling_{expected}")
        if not kernels.is_fast_path_available:
            raise RuntimeError("FLA kernels unavailable. Run: uv sync --locked --extra b200 --extra b200-kernels")
        print("Linear attention: FLA + causal-conv1d available", flush=True)
    check_weights(cfg.model.base_model_id)
    AutoProcessor.from_pretrained(cfg.model.base_model_id, local_files_only=True)
    check_data(cfg, env["ROBOMETER_PROCESSED_DATASETS_PATH"])
    if args.check_only:
        print("Preflight passed; no training or W&B upload performed.")
        return
    if args.mode != "debug" and not diagnostic and not env.get("WANDB_API_KEY"):
        key_file = Path(env.get("WANDB_API_KEY_FILE", str(ROOT / ".secrets/wandb_api_key")))
        env["WANDB_API_KEY"] = key_file.read_text().strip()
        if not env["WANDB_API_KEY"]:
            raise ValueError(f"Empty W&B key file: {key_file}")
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
