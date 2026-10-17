"""Read-only browser for Robometer's processed trajectory datasets."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from functools import lru_cache
from io import BytesIO
import json
import os
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
import numpy as np
import pyarrow as pa
from PIL import Image

ROOT = Path(os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH", "/mnt/workspace/dataset/robometer-1m")).resolve()
HERE = Path(__file__).resolve().parent
app = FastAPI(title="Robometer Dataset Explorer")
cache_lock = RLock()


def read_json(path: Path):
    return json.loads(path.read_text())


def subset_dir(name: str) -> Path:
    path = (ROOT / name).resolve()
    if path.parent != ROOT or not (path / "processed_dataset" / "state.json").is_file():
        raise HTTPException(404, "子集不存在或不是已解包的数据集")
    return path


@lru_cache(maxsize=2)
def table(name: str):
    folder = subset_dir(name) / "processed_dataset"
    state = read_json(folder / "state.json")
    tables = []
    for entry in state["_data_files"]:
        path = (folder / entry["filename"]).resolve()
        if path.parent != folder:
            raise HTTPException(400, "无效的 Arrow 文件路径")
        source = pa.memory_map(str(path), "r")
        try:
            tables.append(pa.ipc.open_stream(source).read_all())
        except pa.ArrowInvalid:
            source.seek(0)
            tables.append(pa.ipc.open_file(source).read_all())
    return pa.concat_tables(tables)


def record(name: str, row: int):
    data = table(name)
    if row < 0 or row >= data.num_rows:
        raise HTTPException(404, "轨迹行号超出范围")
    return data.slice(row, 1).to_pylist()[0]


def asset(name: str, row: int, kind: str) -> Path:
    folder = subset_dir(name)
    entry = record(name, row)
    field, subfolder = ("frames", "frames") if kind == "frames" else ("embeddings_path", "embeddings")
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise HTTPException(404, f"本条轨迹没有 {field} 文件路径")
    # Archives have been flattened; historical absolute/processed_datasets prefixes
    # must not be used to read arbitrary files on the host.
    path = (folder / subfolder / Path(value).name).resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise HTTPException(404, f"找不到 {subfolder}/{Path(value).name}")
    return path


@lru_cache(maxsize=2)
def frames(name: str, row: int):
    with np.load(asset(name, row, "frames"), allow_pickle=False) as archive:
        data = archive["frames"]
    if data.ndim != 4 or data.shape[-1] not in (1, 3, 4) or len(data) == 0:
        raise HTTPException(422, f"不支持的图像数组形状：{data.shape}")
    return data


@lru_cache(maxsize=2)
def mappings(name: str):
    path = subset_dir(name) / "index_mappings.json"
    return read_json(path) if path.exists() else {}


@app.get("/")
def home():
    return FileResponse(HERE / "index.html")


@app.get("/api/subsets")
def subsets():
    if not ROOT.is_dir():
        raise HTTPException(404, f"数据根目录不存在：{ROOT}")
    result = []
    for path in sorted(ROOT.iterdir()):
        if not (path / "processed_dataset" / "state.json").is_file():
            continue
        info = path / "dataset_info.json"
        meta = read_json(info) if info.exists() else {}
        result.append({"name": path.name, "count": meta.get("total_trajectories")})
    return {"root": str(ROOT), "subsets": result}


@app.get("/api/trajectories")
def trajectories(subset: str, q: str = "", offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
    with cache_lock:
        data = table(subset)
        ids = data.column("id").to_pylist()
    matches = [i for i, value in enumerate(ids) if q.casefold() in str(value).casefold()]
    rows = matches[offset:offset + limit]
    return {"total": len(ids), "matched": len(matches), "offset": offset,
            "items": [{"row": i, "id": ids[i]} for i in rows]}


@lru_cache(maxsize=1)
def training_categories():
    spec = importlib.util.spec_from_file_location("vtools_categories", HERE.parent / "robometer/data/dataset_category.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_context(name, item):
    data = table(name)
    # Use the same task grouping as training, limited here to this subset.
    indices = mappings(name).get("task_indices", {}).get(item.get("task"), [])
    if not indices:
        indices = [i for i, task in enumerate(data.column("task").to_pylist()) if task == item.get("task")]
    cols = [c for c in ("id", "quality_label", "partial_success") if c in data.column_names]
    peers = data.select(cols).take(pa.array(indices, type=pa.int64())).to_pylist()
    counts = dict(Counter(p.get("quality_label") or "unknown" for p in peers))
    source = item.get("data_source") or ""
    preference_only = training_categories().is_preference_only_ds(source)
    # Mirrors rbm_heads.should_compute_progress for an unaugmented trajectory.
    if preference_only:
        enabled, reason = False, "该数据源属于 preference_only，训练 collator 屏蔽进度损失。"
    elif item.get("partial_success") is not None and "roboreward" in source:
        enabled, reason = True, "RoboReward 的部分成功标注启用进度监督。"
    elif item.get("quality_label") == "successful":
        enabled, reason = True, "成功轨迹可参与进度监督；标签根据采样帧索引和成功阈值动态生成。"
    else:
        enabled, reason = False, "常规失败/次优轨迹的进度损失被屏蔽；仍可参与偏好比较。其他标签需按具体策略判断。"
    return {"quality_counts": counts, "task_total": len(indices), "peers": [dict(p, row=i) for i, p in zip(indices[:100], peers[:100])],
            "peers_truncated": len(indices) > 100, "progress_enabled": enabled, "progress_reason": reason,
            "preference_only": preference_only,
            "scope": "当前子集、相同 task；实际训练还会合并已配置子集、过滤数据并随机增强。"}


@app.get("/api/record")
def trajectory(subset: str, row: int):
    with cache_lock:
        item = record(subset, row)
        data = frames(subset, row)
        groups = {}
        for key, values in mappings(subset).items():
            if isinstance(values, list) and row in values:
                groups[key] = True
            elif isinstance(values, dict):
                selected = [k for k, indices in values.items() if isinstance(indices, list) and row in indices]
                if selected:
                    groups[key] = selected
        folder = subset_dir(subset)
        meta = folder / "dataset_info.json"
        return {"row": row, "record": item, "dataset_info": read_json(meta) if meta.exists() else {},
                "features": read_json(folder / "processed_dataset" / "dataset_info.json").get("features", {}),
                "dataset_state": read_json(folder / "processed_dataset" / "state.json"),
                "training": training_context(subset, item), "index_membership": groups, "frame_count": len(data), "frame_shape": list(data.shape),
                "frame_dtype": str(data.dtype), "resolved_frames": str(asset(subset, row, "frames"))}


@app.get("/api/frame")
def frame(subset: str, row: int, index: int = Query(0, ge=0)):
    with cache_lock:
        data = frames(subset, row)
        if index >= len(data):
            raise HTTPException(404, "帧号超出范围")
        pixels = data[index]
        if pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/embeddings")
def embeddings(subset: str, row: int):
    import torch
    path = asset(subset, row, "embeddings")
    value = torch.load(path, map_location="cpu", weights_only=True)

    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return {"shape": list(obj.shape), "dtype": str(obj.dtype), "values": obj.tolist()}
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    return {"path": str(path), "content": convert(value)}


@app.get("/api/download")
def download(subset: str, row: int, kind: str):
    if kind not in ("frames", "embeddings"):
        raise HTTPException(400, "仅支持 frames 或 embeddings")
    path = asset(subset, row, kind)
    return FileResponse(path, filename=path.name)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.data_root.resolve()
    uvicorn.run(app, host=args.host, port=args.port)
