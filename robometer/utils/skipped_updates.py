"""397B-only retries: global rejection, successful-update clocks, resume cursor."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import skip_first_batches
from loguru import logger

from robometer.utils.numerics import _tensors


class SkippedUpdates:
    def __init__(self, trainer):
        self.trainer = trainer
        self.attempts = 0
        self.skipped = 0
        self.epoch = 0
        self.batches_in_epoch = 0
        self.output_errors = []
        self.batch_metadata = {}

    def any_rank(self, bad):
        flag = torch.tensor(int(bad), device=self.trainer.args.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def rejected_outputs(self, output):
        values = {key: getattr(output, key, None)
                  for key in ("pref_logits", "progress_logits", "success_logits")}
        errors = [name for name, t in _tensors(values, "outputs")
                  if t.is_floating_point() and not torch.isfinite(t.detach()).all()]
        self.output_errors.extend(errors)
        if not self.any_rank(bool(errors)):
            return None
        # Complete FSDP backward on every rank, then discard ALL gradients. Do
        # not sanitize logits and train on them, or abandon an unfinished graph.
        terms = [t.float().reshape(-1)[:1].sum() * 0 for _, t in _tensors(values, "outputs")
                 if t.requires_grad]
        if not terms:
            raise RuntimeError("Rejected forward has no autograd graph to finish")
        return sum(terms)

    def reject(self, reason, loss, norm=None):
        self.skipped += 1
        trainer = self.trainer
        detail = dict(rank=trainer.accelerator.process_index, loss=str(loss),
                      grad_norm=None if norm is None else str(norm),
                      nonfinite_outputs=self.output_errors, batch=self.batch_metadata)
        details = [None] * dist.get_world_size() if dist.is_initialized() else [detail]
        if dist.is_initialized():
            dist.all_gather_object(details, detail)
        if trainer.is_world_process_zero():
            record = dict(time=datetime.now(timezone.utc).isoformat(), reason=reason,
                          attempt=self.attempts, successful_updates=trainer.state.global_step,
                          skipped_updates=self.skipped, epoch=self.epoch,
                          batches_in_epoch=self.batches_in_epoch,
                          loss_threshold=trainer.config.training.skip_loss_threshold,
                          learning_rate=trainer._get_learning_rate(), ranks=details)
            path = Path(trainer.args.output_dir) / "skipped_updates.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            logger.warning(f"Skipped update: attempt={self.attempts}, successful_updates="
                           f"{trainer.state.global_step}, reason={reason}, total_skipped={self.skipped}; {path}")

    def save(self, directory):
        if self.trainer.is_world_process_zero():
            (Path(directory) / "skipped_update_state.json").write_text(json.dumps(
                dict(attempts=self.attempts, skipped=self.skipped, epoch=self.epoch,
                     batches_in_epoch=self.batches_in_epoch), indent=2))

    def run(self, *, model, train_dataloader, trial, ignore_keys_for_eval, start_time,
            resume_from_checkpoint, epoch, steps_trained_in_current_epoch, **unused):
        trainer = self.trainer
        if trainer.args.gradient_accumulation_steps != 1:
            raise ValueError("397B skipped-update loop currently requires gradient_accumulation_steps=1")
        self.epoch = epoch
        self.batches_in_epoch = steps_trained_in_current_epoch
        self.attempts = trainer.state.global_step
        if resume_from_checkpoint:
            path = Path(resume_from_checkpoint) / "skipped_update_state.json"
            if path.is_file():
                for key, value in json.loads(path.read_text()).items():
                    setattr(self, key, value)
            if trainer.args.ignore_data_skip:
                self.batches_in_epoch = 0
        restore_rng = bool(resume_from_checkpoint)
        while trainer.state.global_step < trainer.state.max_steps:
            if hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(self.epoch)
            length = len(train_dataloader)
            if not length:
                raise RuntimeError("Empty training dataloader")
            loader = skip_first_batches(train_dataloader, self.batches_in_epoch) if self.batches_in_epoch else train_dataloader
            iterator = iter(loader)
            while True:
                batches, num_items = trainer.get_batch_samples(iterator, 1, trainer.args.device)
                if not batches:
                    break
                if restore_rng:
                    trainer._load_rng_state(resume_from_checkpoint)
                    restore_rng = False
                inputs = batches[0]
                self.attempts += 1
                self.batches_in_epoch += 1
                self.output_errors = []
                self.batch_metadata = {}
                for name in ("preference_inputs", "progress_inputs"):
                    batch = inputs.get(name, {})
                    if isinstance(batch, dict):
                        for key in ("data_source", "trajectory_A_data_source", "trajectory_A_data_gen_strategy"):
                            value = batch.get(key)
                            if isinstance(value, str) or (isinstance(value, (tuple, list))
                                                        and all(isinstance(x, (str, int, float, type(None))) for x in value)):
                                self.batch_metadata[f"{name}.{key}"] = value
                trainer.current_gradient_accumulation_steps = 1
                trainer.accelerator.gradient_state._set_sync_gradients(True)
                trainer.control = trainer.callback_handler.on_step_begin(trainer.args, trainer.state, trainer.control)
                loss_tensor = trainer.training_step(model, inputs, num_items)
                loss = float(loss_tensor)
                bad_loss = (bool(self.output_errors) or not math.isfinite(loss)
                            or loss > trainer.config.training.skip_loss_threshold)
                if self.any_rank(bad_loss):
                    self.reject("loss_above_threshold_or_nonfinite_output_or_loss", loss)
                    model.zero_grad(set_to_none=True)
                    del loss_tensor, inputs, batches
                    continue
                # Use FSDP's global norm and clipping; do not call the old
                # fail-fast guard because rejection now retries the next batch.
                norm = trainer.accelerator.clip_grad_norm_(
                    model.parameters(), trainer.args.max_grad_norm if trainer.args.max_grad_norm > 0 else float("inf"))
                norm_value = float(norm)
                if self.any_rank(not math.isfinite(norm_value)):
                    self.reject("nonfinite_gradient_norm", loss, norm_value)
                    model.zero_grad(set_to_none=True)
                    del loss_tensor, inputs, batches
                    continue
                trainer._tr_loss += loss_tensor
                trainer.current_flos += float(trainer.floating_point_ops(inputs))
                trainer._track_num_input_tokens(inputs)
                trainer.control = trainer.callback_handler.on_pre_optimizer_step(trainer.args, trainer.state, trainer.control)
                trainer.optimizer.step()
                trainer.control = trainer.callback_handler.on_optimizer_step(trainer.args, trainer.state, trainer.control)
                learning_rate = trainer._get_learning_rate()
                trainer.lr_scheduler.step()
                model.zero_grad(set_to_none=True)
                trainer.state.global_step += 1
                trainer.state.epoch = self.epoch + self.batches_in_epoch / length
                trainer.control = trainer.callback_handler.on_step_end(trainer.args, trainer.state, trainer.control)
                trainer._maybe_log_save_evaluate(trainer._tr_loss, norm, model, trial, self.epoch,
                                                ignore_keys_for_eval, start_time, learning_rate=learning_rate)
                del loss_tensor, inputs, batches
                if trainer.control.should_training_stop or trainer.state.global_step >= trainer.state.max_steps:
                    trainer.control.should_training_stop = True
                    trainer.control = trainer.callback_handler.on_epoch_end(trainer.args, trainer.state, trainer.control)
                    return
            trainer.control = trainer.callback_handler.on_epoch_end(trainer.args, trainer.state, trainer.control)
            self.epoch += 1
            self.batches_in_epoch = 0
            trainer.control = trainer.callback_handler.on_epoch_begin(trainer.args, trainer.state, trainer.control)
