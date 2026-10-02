"""Opt-in, local-only CPU/CUDA traces for diagnosing distributed training stalls."""

from contextlib import contextmanager
import faulthandler
import gc
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

import torch


@contextmanager
def profile_python_gc(output_dir):
    """Measure GC pauses and capture GIL-independent stacks without a CUDA profiler."""
    if os.environ.get("ROBOMETER_PROFILE_GC") != "1":
        yield
        return
    rank = os.environ.get("RANK", "0")
    folder = Path(output_dir) / "profile"
    folder.mkdir(parents=True, exist_ok=True)
    starts = {}
    owner_pid = os.getpid()
    with (folder / f"gc_rank_{rank}.jsonl").open("w") as events, (
        folder / f"threads_rank_{rank}.txt"
    ).open("w") as stacks:
        def record(phase, info):
            if os.getpid() != owner_pid:  # DataLoader fork inherits callbacks.
                return
            generation = info["generation"]
            if phase == "start":
                usage = resource.getrusage(resource.RUSAGE_SELF)
                starts[generation] = (time.perf_counter(), time.time(), time.thread_time(), usage.ru_minflt)
            elif generation in starts:
                started, timestamp, cpu_started, faults = starts.pop(generation)
                elapsed = time.perf_counter() - started
                if elapsed >= 0.01 or generation == 2:
                    events.write(json.dumps({
                        "timestamp": timestamp, "duration_s": elapsed,
                        "thread_cpu_s": time.thread_time() - cpu_started,
                        "minor_faults": resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults,
                        "thread": threading.get_ident(), **info,
                    }) + "\n")
                    events.flush()

        gc.callbacks.append(record)
        dump_threads = os.environ.get("ROBOMETER_PROFILE_THREADS") == "1"
        if dump_threads:
            faulthandler.dump_traceback_later(5, repeat=True, file=stacks)
        try:
            yield
        finally:
            if dump_threads:
                faulthandler.cancel_dump_traceback_later()
            gc.callbacks.remove(record)


@contextmanager
def diagnose_rank_wait(trainer):
    """Opt-in short A/B/A run on a repeated batch; never a training recipe."""
    if os.environ.get("ROBOMETER_DIAGNOSE_RANK_WAIT") != "1":
        yield
        return
    from transformers import TrainerCallback

    if trainer.args.report_to or trainer.args.max_steps > 20 or trainer.args.max_steps < 1:
        raise ValueError("Rank diagnostic requires uploads disabled and 1–20 steps")
    if str(trainer.args.save_strategy).lower() not in {"no", "savestrategy.no"}:
        raise ValueError("Rank diagnostic requires save_strategy=no")
    original_fetch = trainer.get_batch_samples
    initially_enabled = gc.isenabled()
    cached = None

    def copy_containers(value):
        # Preserve tensor storage while protecting the cached batch from dict/list edits.
        if isinstance(value, dict):
            return {k: copy_containers(v) for k, v in value.items()}
        if isinstance(value, list):
            return [copy_containers(v) for v in value]
        if isinstance(value, tuple):
            return tuple(copy_containers(v) for v in value)
        return value

    def repeat_batch(*args, **kwargs):
        nonlocal cached
        if cached is None:
            cached = original_fetch(*args, **kwargs)
        else:
            trainer._batch_fetch_seconds = 0.0
        return copy_containers(cached)

    class GcComparison(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.step = state.global_step + 1
            self.automatic = self.step <= 2 or self.step >= 6
            (gc.enable if self.automatic else gc.disable)()
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            record = {"step": self.step, "automatic_gc": self.automatic,
                      "optimizer_step_s": time.perf_counter() - self.started,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            path = Path(args.output_dir) / f"gc_comparison_rank_{args.process_index}.jsonl"
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"Rank wait diagnostic: {record}", flush=True)

    callback = GcComparison()
    trainer.get_batch_samples = repeat_batch
    trainer.add_callback(callback)
    try:
        yield
    finally:
        trainer.remove_callback(callback)
        trainer.get_batch_samples = original_fetch
        (gc.enable if initially_enabled else gc.disable)()


@contextmanager
def profile_training_step(output_dir, step, rank):
    """Capture one requested optimizer step, without changing normal training."""
    if os.environ.get("ROBOMETER_PROFILE_STEP") != str(step):
        yield
        return

    folder = Path(output_dir) / "profile"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f"step_{step}_rank_{rank}"
    main_thread = threading.get_ident()
    stop = threading.Event()
    started = time.perf_counter()

    def sample_stacks():
        with prefix.with_suffix(".stacks.jsonl").open("w") as stream:
            while not stop.wait(1.0):
                frame = sys._current_frames().get(main_thread)
                if frame is not None:
                    stream.write(json.dumps({
                        "elapsed_s": time.perf_counter() - started,
                        "stack": traceback.format_stack(frame),
                    }) + "\n")
                    stream.flush()

    sampler = threading.Thread(target=sample_stacks, daemon=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    sampler.start()
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False) as profiler:
            yield
    finally:
        stop.set()
        sampler.join(timeout=2)

    profiler.export_chrome_trace(str(prefix) + ".trace.json.gz")
    averages = profiler.key_averages()
    prefix.with_suffix(".operators.txt").write_text(
        averages.table(sort_by="self_device_time_total", row_limit=60)
        + "\n\nCPU operators:\n"
        + averages.table(sort_by="self_cpu_time_total", row_limit=60)
    )
    print(f"Training profile saved: {prefix}", flush=True)
