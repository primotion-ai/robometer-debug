"""Prepare reference/query images, ask Astra, validate and export unchanged scores."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .backend import CodexBackend, write_json
from .data import load_trajectory
from .connection import Connection
from .prompts import LABEL_SCHEMA, REFERENCE_SCHEMA, label_prompt, reference_prompt


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def unit_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0,1]")


def make_anchors(keyframes, frame_count, progress=None):
    if not isinstance(keyframes, list) or not keyframes:
        raise ValueError("reference.keyframes must be a nonempty list")
    for value in keyframes:
        integer(value, "reference keyframe")
    if keyframes != sorted(set(keyframes)) or keyframes[-1] >= frame_count:
        raise ValueError("keyframes must be strictly increasing, unique and within the reference")
    indices = sorted(set([0, *keyframes]))
    if len(indices) < 2:
        raise ValueError("reference needs an initial frame and a later completion keyframe")
    if progress is None:
        values = [i / (len(indices) - 1) for i in range(len(indices))]
    else:
        # Custom values align with the complete anchor list INCLUDING auto-added frame 0.
        if not isinstance(progress, list) or len(progress) != len(indices):
            raise ValueError("reference.progress must align with [0 + unique keyframes]")
        for value in progress:
            unit_number(value, "reference.progress")
        if progress[0] != 0 or progress[-1] != 1 or any(a >= b for a, b in zip(progress, progress[1:])):
            raise ValueError("reference.progress must strictly increase from 0 to 1")
        values = progress
    return [{"frame_index": index, "progress": value} for index, value in zip(indices, values)]


def validate_reference(response, anchors):
    if not isinstance(response, dict) or response.get("reference_valid") is not True:
        raise ValueError(f"Model rejected reference: {response}")
    actual = response.get("anchors")
    if not isinstance(actual, list) or len(actual) != len(anchors):
        raise ValueError("model rubric omitted/added reference anchors")
    for item, expected in zip(actual, anchors):
        if not isinstance(item, dict):
            raise ValueError("invalid rubric anchor")
        unit_number(item.get("progress"), "rubric.progress")
        integer(item.get("frame_index"), "rubric.frame_index")
        if item["frame_index"] != expected["frame_index"] or not math.isclose(
                item["progress"], expected["progress"], abs_tol=1e-6):
            raise ValueError("model did not echo reference anchors; scores will not be repaired")
        if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("criterion", "transition")):
            raise ValueError("reference rubric lacks visual criteria")


def validate_labels(response, indices, reference_indices):
    if not isinstance(response, dict) or response.get("same_task") is not True:
        raise ValueError(f"Model reported incompatible task or malformed response: {response}")
    if set(response) != set(LABEL_SCHEMA["required"]):
        raise ValueError("model response fields do not match the label schema")
    if not isinstance(response.get("task_evidence"), str) or not response["task_evidence"].strip():
        raise ValueError("model response lacks task compatibility evidence")
    labels = response.get("labels")
    if not isinstance(labels, list) or len(labels) != len(indices):
        raise ValueError("model omitted/added target frames")
    for label, index in zip(labels, indices):
        if not isinstance(label, dict):
            raise ValueError("invalid frame label")
        if set(label) != set(LABEL_SCHEMA["properties"]["labels"]["items"]["required"]):
            raise ValueError("frame label fields do not match the label schema")
        integer(label.get("frame_index"), "frame_index")
        if label["frame_index"] != index:
            raise ValueError("model returned duplicate, missing, out-of-order or unexpected frame indices")
        for key in ("progress", "confidence"):
            unit_number(label.get(key), key)
        match = label.get("matched_reference_keyframe")
        if match is not None and (type(match) is not int or match not in reference_indices):
            raise ValueError("matched_reference_keyframe is not a supplied reference anchor")
        if type(label.get("needs_review")) is not bool:
            raise ValueError("needs_review must be boolean")
        if not all(isinstance(label.get(key), str) and label[key].strip() for key in ("phase", "evidence")):
            raise ValueError("frame label lacks phase/visual evidence")
    # No sorting, clipping, interpolation, anchor snapping, smoothing or endpoint normalization.
    return labels


def save_frame(frames, index, kind, folder, max_side):
    path = folder / f"{kind.lower()}_{index:06d}.png"
    pixels = frames[index]
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    image = Image.fromarray(pixels).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max(240, image.width), image.height + 28), "white")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((6, 8), f"{kind} | frame_index={index}", fill="black")
    canvas.save(path)
    return path


def overview_indices(count, limit):
    if limit == 0:
        return []
    return sorted(set(np.linspace(0, count - 1, min(count, limit), dtype=int).tolist()))


def export(output, manifest, rubric, labels):
    result = {"model": manifest["model"], "instruction": manifest["instruction"],
              "reference": manifest["reference"], "target": manifest["target"],
              "rubric": rubric, "score_source": "model_direct_no_postprocessing", "labels": labels,
              "keyframe_matches": [{"target_frame_index": item["frame_index"],
                                     "reference_frame_index": item["matched_reference_keyframe"],
                                     "progress": item["progress"], "confidence": item["confidence"]}
                                    for item in labels if item["matched_reference_keyframe"] is not None]}
    if "connection" in manifest:
        result["connection"] = manifest["connection"]
    write_json(output / "labels.json", result)
    with (output / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    # Plotting joins observed samples for viewing; no new labels are generated.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot([v["frame_index"] for v in labels], [v["progress"] for v in labels], label="Astra direct progress")
    for anchor in manifest["anchors"]:
        ax.axhline(anchor["progress"], alpha=0.2, color="gray", linestyle="--")
    matches = [v for v in labels if v["matched_reference_keyframe"] is not None]
    if matches:
        ax.scatter([v["frame_index"] for v in matches], [v["progress"] for v in matches],
                   color="darkorange", label="Model-matched reference keyframes", zorder=3)
    review = [v for v in labels if v["needs_review"]]
    if review:
        ax.scatter([v["frame_index"] for v in review], [v["progress"] for v in review],
                   marker="x", color="red", label="Needs review", zorder=4)
    ax.set(xlabel="Stored target frame index (zero-based)", ylabel="Progress", ylim=(-0.03, 1.03))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / "progress.png", dpi=160)
    plt.close(fig)
    return result


def run(config, output, prepare_only=False, backend=None):
    allowed = {"data_root", "reference", "target", "instruction", "batch_size", "context_frames",
               "overview_frames", "reference_neighbors", "max_image_side", "model", "reasoning", "timeout", "codex_bin",
               "auth_mode", "api_config", "api_base_url", "api_use_proxy"}
    if set(config) - allowed:
        raise ValueError(f"unknown config fields: {sorted(set(config) - allowed)}")
    output = Path(output).resolve()
    connection = Connection.from_config(config)
    root = Path(config.get("data_root", os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH",
                                                     "/mnt/workspace/dataset/robometer-1m")))
    reference_spec, target_spec = config["reference"], config["target"]
    reference = load_trajectory(root, reference_spec["category"], reference_spec["id"])
    target = load_trajectory(root, target_spec["category"], target_spec["id"])
    instruction = config.get("instruction") or reference.task
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction is required when the reference task is empty")
    batch_size = integer(config.get("batch_size", 16), "batch_size", 1)
    context = integer(config.get("context_frames", 2), "context_frames")
    overview = integer(config.get("overview_frames", 8), "overview_frames")
    neighbors = integer(config.get("reference_neighbors", 1), "reference_neighbors")
    max_side = integer(config.get("max_image_side", 768), "max_image_side", 32)
    timeout = integer(config.get("timeout", 900), "timeout", 1)
    model = config.get("model", "gpt-6-astra")
    reasoning = config.get("reasoning", "high")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    if reasoning not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("invalid reasoning effort")
    anchors = make_anchors(reference_spec["keyframes"], len(reference.frames), reference_spec.get("progress"))
    ref_indices = sorted({j for a in anchors for j in range(max(0, a["frame_index"] - neighbors),
                                                          min(len(reference.frames), a["frame_index"] + neighbors + 1))})
    def identity(trajectory):
        return {"category": trajectory.category, "id": trajectory.id, "row": trajectory.row,
                "task": trajectory.task, "frame_count": len(trajectory.frames), "frames_path": str(trajectory.path),
                "frames_sha256": hashlib.sha256(trajectory.frames.tobytes()).hexdigest()}
    manifest = {"config": config, "instruction": instruction, "model": model, "reasoning": reasoning,
                "reference": identity(reference), "target": identity(target), "anchors": anchors,
                "planned_calls": 1 + math.ceil(len(target.frames) / batch_size),
                "index_convention": "zero_based_processed_npz", "score_source": "model_direct_no_postprocessing"}
    output.mkdir(parents=True, exist_ok=True)
    if connection.mode == "api":
        manifest["connection"] = connection.public()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output directory belongs to a different config/data run; choose a new --output directory")
    write_json(manifest_path, manifest)
    image_folder = output / "images"
    image_folder.mkdir(exist_ok=True)
    ref_paths = {i: save_frame(reference.frames, i, "REFERENCE", image_folder, max_side) for i in ref_indices}
    target_paths = {i: save_frame(target.frames, i, "TARGET", image_folder, max_side) for i in range(len(target.frames))}
    reference_manifest = [{"role": "reference", "frame_index": i, "file": path.name} for i, path in ref_paths.items()]
    ref_prompt = reference_prompt(instruction, reference.task, anchors, reference_manifest)
    (output / "reference_prompt.txt").write_text(ref_prompt)
    write_json(output / "reference_schema.json", REFERENCE_SCHEMA)
    write_json(output / "label_schema.json", LABEL_SCHEMA)
    print(f"Reference {len(reference.frames)} frames; target {len(target.frames)} frames; "
          f"anchors {anchors}; planned model calls {manifest['planned_calls']}", flush=True)
    if prepare_only:
        return manifest
    backend = backend or CodexBackend(output, model, reasoning, timeout, config.get("codex_bin", "codex"), connection)
    rubric = backend.ask("reference", ref_prompt, list(ref_paths.values()), REFERENCE_SCHEMA)
    validate_reference(rubric, anchors)
    write_json(output / "rubric.json", rubric)
    labels = []
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    global_indices = overview_indices(len(target.frames), overview)
    for start in range(0, len(target.frames), batch_size):
        indices = list(range(start, min(start + batch_size, len(target.frames))))
        visible = sorted(set(global_indices + list(range(max(0, start - context),
                                                        min(len(target.frames), indices[-1] + 1 + context)))))
        images = list(ref_paths.values()) + [target_paths[i] for i in visible]
        image_manifest = reference_manifest + [{"role": "target_query" if i in indices else "target_context_only",
                                               "frame_index": i, "file": target_paths[i].name} for i in visible]
        prompt = label_prompt(instruction, reference.task, target.task, rubric, indices, image_manifest,
                              labels[-context:] if context else [])
        name = f"target_{start:06d}_{indices[-1]:06d}"
        response = backend.ask(name, prompt, images, LABEL_SCHEMA)
        # Save before validating so semantic rejection is still inspectable.
        write_json(chunks / f"{name}.json", response)
        batch = validate_labels(response, indices, [a["frame_index"] for a in anchors])
        labels.extend(batch)
        print(f"Validated {len(labels)}/{len(target.frames)} frame labels", flush=True)
    return export(output, manifest, rubric, labels)
