"""The diagnostic must repeat inputs safely and restore normal GC on failures."""
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from robometer.utils.training_profile import diagnose_rank_wait


class RankDiagnosticTests(unittest.TestCase):
    def test_repeat_batch_policy_and_exception_cleanup(self):
        before = gc.isenabled()
        with tempfile.TemporaryDirectory() as folder:
            tensor = torch.ones(2)
            calls = []
            callbacks = []
            def fetch():
                calls.append(1)
                return [{"tokens": tensor, "metadata": ["original"]}], None
            trainer = SimpleNamespace(
                args=SimpleNamespace(report_to=[], max_steps=7, save_strategy="no",
                                     output_dir=folder, process_index=0),
                get_batch_samples=fetch, add_callback=callbacks.append,
                remove_callback=callbacks.remove,
            )
            with patch.dict(os.environ, {"ROBOMETER_DIAGNOSE_RANK_WAIT": "1"}), \
                 patch("torch.cuda.synchronize"), patch("torch.cuda.max_memory_allocated", return_value=0):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    with diagnose_rank_wait(trainer):
                        first = trainer.get_batch_samples()
                        first[0][0]["metadata"].append("changed")
                        second = trainer.get_batch_samples()
                        self.assertEqual(second[0][0]["metadata"], ["original"])
                        self.assertIs(second[0][0]["tokens"], tensor)
                        self.assertEqual(len(calls), 1)
                        callback = callbacks[0]
                        for step in range(1, 8):
                            state = SimpleNamespace(global_step=step-1)
                            callback.on_step_begin(trainer.args, state, None)
                            self.assertEqual(gc.isenabled(), step <= 2 or step >= 6)
                            callback.on_step_end(trainer.args, state, None)
                        raise RuntimeError("test failure")
            self.assertIs(trainer.get_batch_samples, fetch)
            self.assertEqual(callbacks, [])
            self.assertEqual(gc.isenabled(), before)
            records = [json.loads(x) for x in (Path(folder)/"gc_comparison_rank_0.jsonl").read_text().splitlines()]
            self.assertEqual([x["automatic_gc"] for x in records], [True, True, False, False, False, True, True])


if __name__ == "__main__":
    unittest.main()
