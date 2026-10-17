"""Offline contract tests: no paid model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa

from astra_label.data import load_trajectory
from astra_label.pipeline import make_anchors, run, validate_labels


def fixture(root, ids=("ref", "target")):
    folder = root / "subset"
    (folder / "processed_dataset").mkdir(parents=True)
    (folder / "frames").mkdir()
    records = []
    for ident in ids:
        np.savez(folder / "frames" / f"{ident}.npz", frames=np.zeros((5, 12, 16, 3), dtype=np.uint8))
        records.append({"id": ident, "task": "put the cup on the plate",
                        "frames": f"/old/machine/processed_datasets/subset/frames/{ident}.npz"})
    table = pa.Table.from_pylist(records)
    with pa.OSFile(str(folder / "processed_dataset" / "data.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    (folder / "processed_dataset" / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data.arrow"}]}))


def label(index, progress):
    return {"frame_index": index, "progress": progress, "matched_reference_keyframe": None,
            "phase": "approach", "confidence": 0.8, "evidence": "Cup position is visible.", "needs_review": False}


class PipelineTests(unittest.TestCase):
    def test_load_by_id_not_row_and_relocate_old_asset_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, ids=("17", "0"))
            trajectory = load_trajectory(root, "subset", "0")
            self.assertEqual(trajectory.row, 1)
            self.assertEqual(trajectory.frames.shape, (5, 12, 16, 3))
            with self.assertRaises(ValueError):
                load_trajectory(root, "../subset", "0")

    def test_duplicate_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d), ids=("same", "same"))
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                load_trajectory(Path(d), "subset", "same")

    def test_anchor_weights_depend_on_stages_not_frame_distance(self):
        self.assertEqual(make_anchors([1, 9], 10), [
            {"frame_index": 0, "progress": 0.0},
            {"frame_index": 1, "progress": 0.5},
            {"frame_index": 9, "progress": 1.0}])
        self.assertEqual(make_anchors([0, 9], 10)[1]["progress"], 1.0)
        for bad in ([0], [5, 2], [2, 2], [-1, 3], [1, 10], [True, 4]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                make_anchors(bad, 10)

    def test_validation_preserves_regression_and_incomplete_endpoint(self):
        response = {"same_task": True, "task_evidence": "same cup task", "labels": [
            label(0, 0.2), label(1, 0.7), label(2, 0.4)]}
        original = copy.deepcopy(response)
        result = validate_labels(response, [0, 1, 2], [0, 4])
        self.assertEqual([x["progress"] for x in result], [0.2, 0.7, 0.4])
        self.assertEqual(response, original)

    def test_missing_duplicate_invalid_and_wrong_task_are_rejected(self):
        good = {"same_task": True, "task_evidence": "same task", "labels": [label(0, 0.2), label(1, 0.8)]}
        cases = []
        for value in (-0.1, 1.1, float("nan"), True):
            bad = copy.deepcopy(good)
            bad["labels"][0]["progress"] = value
            cases.append(bad)
        bad = copy.deepcopy(good)
        bad["same_task"] = False
        cases.append(bad)
        bad = copy.deepcopy(good)
        bad["labels"][1]["frame_index"] = 0
        cases.append(bad)
        bad = copy.deepcopy(good)
        bad["labels"].pop()
        cases.append(bad)
        bad = copy.deepcopy(good)
        bad["labels"][0]["matched_reference_keyframe"] = 99
        cases.append(bad)
        bad = copy.deepcopy(good)
        del bad["labels"][0]["matched_reference_keyframe"]
        cases.append(bad)
        bad = copy.deepcopy(good)
        bad["task_evidence"] = ""
        cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_labels(bad, [0, 1], [0, 4])

    def test_prepare_only_never_calls_model_and_rejects_output_reuse_for_other_data(self):
        class NoModel:
            def ask(self, *args):
                raise AssertionError("prepare-only must not call the model")

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            config = {"data_root": str(root), "reference": {"category": "subset", "id": "ref", "keyframes": [4]},
                      "target": {"category": "subset", "id": "target"}}
            output = root / "output"
            manifest = run(config, output, prepare_only=True, backend=NoModel())
            self.assertEqual(manifest["planned_calls"], 2)
            self.assertEqual(len(list((output / "images").glob("target_*.png"))), 5)
            self.assertFalse((output / "labels.json").exists())
            config["instruction"] = "changed instruction"
            with self.assertRaisesRegex(ValueError, "different config/data"):
                run(config, output, prepare_only=True, backend=NoModel())

    def test_complete_pipeline_exports_model_values_without_interpolation(self):
        # Only replace the remote model boundary; use real Arrow/NPZ, prompts and exports.
        class Model:
            def ask(self, name, prompt, images, schema):
                if name == "reference":
                    return {"reference_valid": True, "reason": "visible reference", "anchors": [
                        {"frame_index": 0, "progress": 0.0, "criterion": "initial state", "transition": "start"},
                        {"frame_index": 4, "progress": 1.0, "criterion": "cup on plate", "transition": "move cup"}]}
                ids = json.loads(prompt.split("LABEL_FRAME_INDICES_JSON: ")[1].splitlines()[0])
                values = [0.1, 0.6, 0.3, 0.7, 0.8]
                return {"same_task": True, "task_evidence": "cup and plate", "labels": [label(i, values[i]) for i in ids]}

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            config = {"data_root": str(root), "reference": {"category": "subset", "id": "ref", "keyframes": [4]},
                      "target": {"category": "subset", "id": "target"}, "batch_size": 2}
            output = root / "output"
            run(config, output, backend=Model())
            result = json.loads((output / "labels.json").read_text())
            self.assertEqual([x["progress"] for x in result["labels"]], [0.1, 0.6, 0.3, 0.7, 0.8])
            self.assertTrue((output / "progress.png").is_file())
            self.assertEqual(len((output / "labels.csv").read_text().splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
