import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robometer.data.datasets.helpers import load_frames_from_npz, resolve_processed_path


class ProcessedPathTests(unittest.TestCase):
    def test_multi_image_resolution_reaches_qwen_processor(self):
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from robometer.data.collators.rbm_heads import RBMBatchCollator

        collator = RBMBatchCollator.__new__(RBMBatchCollator)
        collator.use_multi_image = True
        collator.use_per_frame_progress_token = True
        collator.resized_height = 128
        collator.resized_width = 128
        frames, extras = collator._prepare_frames_for_conversation([Image.new("RGB", (256, 256))])
        content = []
        collator._add_vision_content_to_list(content, frames, extras)
        images, _ = process_vision_info([{"role": "user", "content": content}], image_patch_size=16)
        self.assertEqual(images[0].size, (128, 128))

    def test_relocated_archive_and_existing_layouts(self):
        with tempfile.TemporaryDirectory(prefix="processed_datasets-parent-") as tmp:
            root = Path(tmp) / "robometer-1m"
            frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
            target = root / "subset/frames/example.npz"
            target.parent.mkdir(parents=True)
            np.savez(target, frames=frames)
            with patch.dict(os.environ, {"ROBOMETER_PROCESSED_DATASETS_PATH": str(root)}):
                for filename in ("./processed_datasets/subset/frames/example.npz",
                                 "subset/frames/example.npz", str(target)):
                    np.testing.assert_array_equal(load_frames_from_npz(filename), frames)
                self.assertEqual(resolve_processed_path("/elsewhere/missing.npz"), "/elsewhere/missing.npz")
            # Original repository-root layout remains usable as well.
            with patch.dict(os.environ, {"ROBOMETER_PROCESSED_DATASETS_PATH": tmp}):
                self.assertEqual(resolve_processed_path("robometer-1m/subset/frames/example.npz"), str(target))


if __name__ == "__main__":
    unittest.main()
