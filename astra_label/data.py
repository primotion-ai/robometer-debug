"""Read the same processed Arrow/NPZ layout as vtools, without its web server."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


@dataclass
class Trajectory:
    category: str
    id: str
    row: int
    task: str
    frames: np.ndarray
    path: Path


def load_trajectory(root: Path, category: str, trajectory_id: str) -> Trajectory:
    root = Path(root).resolve()
    if not isinstance(category, str) or Path(category).name != category:
        raise ValueError("category must be a direct subset directory name")
    folder = (root / category).resolve()
    if folder.parent != root:
        raise ValueError("category escapes data root")
    processed = folder / "processed_dataset"
    state = json.loads((processed / "state.json").read_text())
    matches = []
    offset = 0
    for entry in state["_data_files"]:
        path = (processed / entry["filename"]).resolve()
        if path.parent != processed:
            raise ValueError("Arrow file escapes processed_dataset")
        with pa.memory_map(str(path), "r") as source:
            try:
                table = pa.ipc.open_stream(source).read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                table = pa.ipc.open_file(source).read_all()
            indices = pc.indices_nonzero(pc.equal(pc.cast(table.column("id"), pa.string()), str(trajectory_id)))
            for index in indices.to_pylist():
                # Do not deserialize language embeddings or training labels.
                item = table.select(["id", "task", "frames"]).slice(index, 1).to_pylist()[0]
                matches.append((offset + index, item))
            offset += table.num_rows
    if len(matches) != 1:
        raise ValueError(f"id {trajectory_id!r}: expected one match in {category}, found {len(matches)} (missing/ambiguous)")
    row, item = matches[0]
    if not isinstance(item["frames"], str) or not item["frames"]:
        raise ValueError("record has no frames NPZ path")
    path = (folder / "frames" / Path(item["frames"]).name).resolve()
    if not path.is_relative_to(folder):
        raise ValueError("frames asset escapes subset")
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"]
    if frames.ndim != 4 or not len(frames) or frames.shape[-1] not in (1, 3, 4) or frames.dtype != np.uint8:
        raise ValueError(f"expected nonempty uint8 THWC frames, got {frames.shape}, {frames.dtype}")
    return Trajectory(category, str(item["id"]), row, item["task"] or "", frames, path)
