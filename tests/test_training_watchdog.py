import json
import tempfile
import unittest
from pathlib import Path

import torch

from robometer.utils.training_watchdog import TrainingWatchdog


class Qwen3_5MoeDecoderLayer(torch.nn.Linear):
    pass


class WatchdogTests(unittest.TestCase):
    def test_tracks_backward_without_changing_gradients_and_closes(self):
        model = Qwen3_5MoeDecoderLayer(2, 1, bias=False)
        with tempfile.TemporaryDirectory() as directory:
            watcher = TrainingWatchdog(directory, 7, torch.device("cpu"))
            watcher.start(model)
            try:
                watcher.note(phase="forward", step=1)
                model(torch.ones(1, 2)).sum().backward()
                self.assertEqual(watcher.state["phase"], "backward")
                torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))
                watcher._snapshot()
            finally:
                watcher.close()
                watcher.close()
            records = [json.loads(line) for line in
                       (Path(directory) / "progress_rank_7.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["phase"], "backward")
            self.assertEqual(records[-1]["step"], 1)
            self.assertFalse(watcher.thread.is_alive())
            self.assertTrue(watcher.stack_file.closed)
            self.assertFalse(model._forward_hooks or model._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
