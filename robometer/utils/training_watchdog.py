"""Debug-only rank progress and Python stacks, without CUDA synchronization."""

import faulthandler
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


class TrainingWatchdog:
    def __init__(self, output_dir, rank, device, interval=30, stack_interval=120):
        self.directory = Path(output_dir)
        self.rank, self.device = rank, device
        self.interval, self.stack_interval = interval, stack_interval
        self.state = {"phase": "train_begin", "step": 0, "event": "start", "module": ""}
        self.changed_at = time.monotonic()
        self.stopped = threading.Event()
        self.handles = []

    def note(self, **updates):
        self.state = {**self.state, **updates}
        self.changed_at = time.monotonic()

    def _snapshot(self):
        record = {"time": datetime.now(timezone.utc).isoformat(), "rank": self.rank,
                  **self.state, "seconds_since_event": round(time.monotonic() - self.changed_at, 1)}
        if self.device.type == "cuda":
            record.update(allocated_gib=round(torch.cuda.memory_allocated(self.device) / 2**30, 2),
                          reserved_gib=round(torch.cuda.memory_reserved(self.device) / 2**30, 2),
                          peak_allocated_gib=round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2))
        line = json.dumps(record)
        with (self.directory / f"progress_rank_{self.rank}.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(f"[ROBOMETER_PROGRESS] {line}", flush=True)

    def _heartbeat(self):
        while not self.stopped.wait(self.interval):
            self._snapshot()

    def start(self, model):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stack_file = (self.directory / f"stacks_rank_{self.rank}.log").open("w")
        faulthandler.dump_traceback_later(self.stack_interval, repeat=True, file=self.stack_file)
        blocks = {"Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"}

        def before(name):
            def hook(module, inputs):
                self.note(event="forward_enter", module=name)
            return hook

        def after(name, is_block):
            def hook(module, inputs, output):
                self.note(event="forward_exit", module=name)
                if is_block and isinstance(output, torch.Tensor) and output.requires_grad:
                    def backward_enter(gradient):
                        self.note(phase="backward", event="block_output_gradient", module=name)
                        return gradient
                    output.register_hook(backward_enter)
            return hook

        for name, module in model.named_modules():
            is_block = type(module).__name__ in blocks
            if is_block or not any(module.children()):
                self.handles.append(module.register_forward_pre_hook(before(name)))
                self.handles.append(module.register_forward_hook(after(name, is_block)))
        self._snapshot()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="robometer-watchdog")
        self.thread.start()

    def close(self):
        self.stopped.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=2)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if hasattr(self, "stack_file") and not self.stack_file.closed:
            faulthandler.cancel_dump_traceback_later()
            self.stack_file.close()
