"""Confusion-matrix VLM evaluation must not need a separate text encoder."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import torch

from robometer.configs.experiment_configs import DataConfig
# Import through the dataset entrypoint, matching production initialization order.
from robometer.data.datasets.custom_eval import ConfusionMatrixSampler


class ConfusionMatrixOfflineTests(unittest.TestCase):
    def make_sampler(self, folder, load_embeddings):
        frames = Path(folder) / "frames.npz"
        embeddings = Path(folder) / "embeddings.pt"
        np.savez(frames, frames=np.zeros((3, 16, 16, 3), dtype=np.uint8))
        torch.save({"video_embeddings": torch.ones(3, 4), "text_embedding": torch.zeros(2)}, embeddings)
        dataset = Dataset.from_list([
            dict(id=str(i), task=task, data_source="test", is_robot=True,
                 quality_label="successful", partial_success=1.0,
                 frames=str(frames), embeddings_path=str(embeddings))
            for i, task in enumerate(["task A", "task B"])
        ])
        indices = {key: {} for key in (
            "optimal_by_task", "suboptimal_by_task", "quality_indices", "partial_success_indices",
            "paired_human_robot_by_task", "tasks_with_multiple_quality_labels",
        )}
        indices.update(robot_trajectories=[0, 1], human_trajectories=[],
                       task_indices={"task A": [0], "task B": [1]}, source_indices={"test": [0, 1]})
        return ConfusionMatrixSampler(
            config=DataConfig(load_embeddings=load_embeddings, max_frames=2),
            dataset=dataset, combined_indices=indices, verbose=False,
        )

    def test_vlm_pairs_use_raw_text_and_local_frames_without_minilm(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "sentence_transformers.SentenceTransformer", side_effect=AssertionError("Unexpected MiniLM load")
        ) as encoder:
            sampler = self.make_sampler(folder, load_embeddings=False)
            samples = [sampler[i].trajectory for i in range(len(sampler))]
            self.assertEqual(len(samples), 4)
            self.assertEqual({(s.metadata["video_task"], s.task) for s in samples},
                             {(a, b) for a in ("task A", "task B") for b in ("task A", "task B")})
            for sample in samples:
                self.assertEqual(sample.frames.shape, (2, 16, 16, 3))
                self.assertIsNone(sample.text_embedding)
            encoder.assert_not_called()

    def test_embedding_models_still_override_instruction_embedding(self):
        vectors = {"task A": np.array([1., 0.]), "task B": np.array([0., 1.])}
        with tempfile.TemporaryDirectory() as folder, patch("sentence_transformers.SentenceTransformer") as encoder:
            encoder.return_value.encode.side_effect = vectors.__getitem__
            sampler = self.make_sampler(folder, load_embeddings=True)
            encoder.assert_called_once_with("sentence-transformers/all-MiniLM-L12-v2")
            for i in range(len(sampler)):
                sample = sampler[i].trajectory
                torch.testing.assert_close(sample.text_embedding, torch.tensor(vectors[sample.task]))
                self.assertEqual(tuple(sample.video_embeddings.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
