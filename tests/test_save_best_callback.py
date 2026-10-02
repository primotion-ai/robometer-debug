"""Exercise checkpoint decisions without importing model or GPU dependencies."""

import ast
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def load_callback():
    source_path = Path(__file__).resolve().parents[1] / "robometer/utils/save.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    callback = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SaveBestCallback")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), callback],
        type_ignores=[],
    )
    namespace = {
        "TrainerCallback": object,
        "datetime": datetime,
        "np": SimpleNamespace(mean=lambda values: sum(values) / len(values)),
        "logger": Mock(),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "gc": Mock(),
        "os": os,
        "shutil": shutil,
        "re": re,
        "is_rank_0": lambda: True,
        "HfApi": Mock(),
    }
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["SaveBestCallback"], namespace


SaveBestCallback, CALLBACK_GLOBALS = load_callback()


class SaveBestCallbackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.args = SimpleNamespace(output_dir=self.folder.name)
        self.control = object()
        CALLBACK_GLOBALS["logger"].reset_mock()
        CALLBACK_GLOBALS["HfApi"].reset_mock()

    def make_callback(self, **kwargs):
        callback = SaveBestCallback(**kwargs)
        callback.setup_trainer_reference(SimpleNamespace(is_world_process_zero=lambda: True))

        def save_files(args, ckpt_dir, metrics, step):
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            (Path(ckpt_dir) / "step.txt").write_text(str(step))

        callback._save_checkpoint_files = Mock(side_effect=save_files)
        callback._cleanup_memory = Mock()
        callback._upload_checkpoint_to_hub = Mock(return_value=("https://example.test/checkpoint", "commit"))
        return callback

    def evaluate(self, callback, step, metrics):
        result = callback.on_evaluate(self.args, SimpleNamespace(global_step=step), self.control, metrics)
        self.assertIs(result, self.control)

    def test_none_keeps_default_monitor_and_selects_better_checkpoint(self):
        callback = self.make_callback()
        self.assertEqual(callback.metric_names, ["custom_eval/p_rank_spearman_mw"])
        self.assertEqual(callback.greater_is_better, [True])
        metric = callback.metric_names[0]
        for step, score in ((500, 0.6), (1000, 0.5), (1500, 0.8)):
            self.evaluate(callback, step, {metric: score})
        self.assertEqual([call.args[3] for call in callback._save_checkpoint_files.call_args_list], [500, 1500])
        self.assertEqual(callback._best_val, 0.8)
        self.assertEqual(len(callback._saved), 1)
        self.assertEqual([path.name for path in Path(self.folder.name).iterdir()],
                         ["ckpt-p_rank_spearman_mw=0.8000_step=1500"])

    def test_empty_monitors_without_periodic_save_are_noop(self):
        for directions in (None, []):
            with self.subTest(greater_is_better=directions):
                callback = self.make_callback(metric_names=[], greater_is_better=directions, upload_to_hub=True)
                self.evaluate(callback, 1000, {"eval_p_rank/spearman_rbm_1m_ood": 0.7})
                self.assertEqual(callback.metric_names, [])
                self.assertEqual(callback.greater_is_better, [])
                callback._save_checkpoint_files.assert_not_called()
                callback._upload_checkpoint_to_hub.assert_not_called()
                callback._cleanup_memory.assert_not_called()
                self.assertEqual(callback._saved, [])
                self.assertIsNone(callback._best_val)
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])
        CALLBACK_GLOBALS["logger"].warning.assert_not_called()

    def test_empty_monitors_keep_periodic_latest_and_skip_duplicate_steps(self):
        callback = self.make_callback(metric_names=[], greater_is_better=[], save_every=1000)
        for step in (0, 500, 1000, 1000, 1500, 2000):
            self.evaluate(callback, step, {})
        self.assertEqual([call.args[3] for call in callback._save_checkpoint_files.call_args_list], [1000, 2000])
        self.assertEqual(callback._last_save_step, 2000)
        self.assertEqual(callback._last_best_save_step, -1)
        self.assertEqual(callback._saved, [])
        self.assertIsNone(callback._best_val)
        # Periodic latest keeps its existing replacement policy, without a fake score.
        self.assertEqual([path.name for path in Path(self.folder.name).iterdir()], ["ckpt-latest-step=2000"])
        CALLBACK_GLOBALS["logger"].warning.assert_not_called()

    def test_empty_monitors_keep_periodic_upload_names_without_fake_scores(self):
        callback = self.make_callback(metric_names=[], save_every=1000, upload_to_hub=True)
        self.evaluate(callback, 1000, {})
        self.evaluate(callback, 2000, {})
        uploads = callback._upload_checkpoint_to_hub.call_args_list
        self.assertEqual([call.kwargs["tag_name"] for call in uploads], ["latest-step-1000", "latest-step-2000"])
        self.assertEqual(uploads[-1].kwargs["commit_message"], "Latest checkpoint at step 2000")
        CALLBACK_GLOBALS["HfApi"].return_value.delete_tag.assert_called_once_with(
            repo_id=uploads[-1].kwargs["hub_model_id"], repo_type="model", tag="latest-step-1000"
        )

    def test_periodic_save_remains_rank_zero_only(self):
        callback = self.make_callback(metric_names=[], save_every=1000)
        callback.setup_trainer_reference(SimpleNamespace(is_world_process_zero=lambda: False))
        self.evaluate(callback, 1000, {})
        callback._save_checkpoint_files.assert_not_called()
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])

    def test_explicit_direction_count_must_match_monitors(self):
        for names, directions in (([], [True]), (["metric"], [])):
            with self.subTest(names=names, directions=directions), self.assertRaises(ValueError):
                SaveBestCallback(metric_names=names, greater_is_better=directions)


if __name__ == "__main__":
    unittest.main()
