"""Multi-node logging ownership, with all W&B operations mocked (no uploads)."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from robometer.utils.distributed import get_rank, is_rank_0
from robometer.utils.logger import Logger


class DistributedLoggingTests(unittest.TestCase):
    def test_global_rank_before_group_initialization(self):
        with patch("torch.distributed.is_initialized", return_value=False):
            for rank in range(40):
                with self.subTest(rank=rank), patch.dict(
                    os.environ, {"RANK": str(rank), "LOCAL_RANK": str(rank % 8)}, clear=True
                ):
                    self.assertEqual(get_rank(), rank)
                    self.assertEqual(is_rank_0(), rank == 0)

    def test_initialized_group_is_authoritative(self):
        # Even stale parent environment variables must not override the group.
        with patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}, clear=True), patch(
            "torch.distributed.is_available", return_value=True
        ), patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_rank", return_value=32
        ):
            self.assertEqual(get_rank(), 32)
            self.assertFalse(is_rank_0())

    def test_standalone_process(self):
        with patch.dict(os.environ, {}, clear=True), patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(get_rank(), 0)
            self.assertTrue(is_rank_0())

    def test_five_nodes_create_and_log_one_wandb_run(self):
        run = MagicMock()
        run.id, run.name, run.project, run.entity = "test-id", "test-run", "test-project", "test-entity"
        run.url, run.notes = "https://example.invalid/test-run", ""
        with tempfile.TemporaryDirectory() as folder, patch(
            "torch.distributed.is_initialized", return_value=False
        ), patch("robometer.utils.logger.setup_loguru_logging"), patch(
            "robometer.utils.logger.wandb.run", None
        ), patch("robometer.utils.logger.wandb.init", return_value=run) as init:
            for rank in range(40):
                with patch.dict(os.environ, {"RANK": str(rank), "LOCAL_RANK": str(rank % 8)}, clear=True):
                    logger = Logger(log_to=["wandb"], output_dir=folder, is_main_process=is_rank_0())
                    logger.init_wandb(project="test-project", entity=None, name="test-run", config={})
                    logger.log_scalars({"train/loss": 1.25}, step=1)
                    logger.write_wandb_info(folder, "test-run")
                    self.assertEqual(logger.enabled("wandb"), rank == 0)
            init.assert_called_once()
            run.log.assert_called_once_with({"train/loss": 1.25}, step=1)
            self.assertEqual(json.loads((Path(folder) / "wandb_info.json").read_text())["wandb_id"], "test-id")


if __name__ == "__main__":
    unittest.main()
