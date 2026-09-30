"""原子写入和滚动保留；只操作当前实验明确命名的产物。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path: str | Path, value) -> None:
    """先完整写入同目录临时文件，fsync 后替换；失败时保留已有有效文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".incomplete-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path: str | Path, value) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".incomplete-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_bytes(path: str | Path, value: bytes) -> None:
    """恢复 checkpoint 内嵌的 best 原始字节，保持原文件 SHA256 不变。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".incomplete-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prune_files(directory: Path, pattern: str, keep: int) -> None:
    """只删自己生成的周期文件，best/latest 不匹配周期文件模式。"""
    files = sorted(directory.glob(pattern))
    for path in files[:-keep]:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"拒绝清理非普通文件: {path}")
        path.unlink()


def save_checkpoint(directory: Path, payload: dict, keep: int) -> Path:
    """周期快照保存完整恢复状态；latest.pt 硬链接到新快照，避免重复存储。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{payload['cursor']['step']:09d}.pt"
    atomic_torch(path, payload)
    latest = directory / "latest.pt"
    temporary = directory / ".latest-link"
    temporary.unlink(missing_ok=True)
    os.link(path, temporary)
    os.replace(temporary, latest)
    # 新周期档和 latest 都成功发布之后再清理旧周期档。
    prune_files(directory, "step_*.pt", keep)
    return path


def save_evaluation(directory: Path, payload: dict, keep: int, *, prefix: str) -> Path:
    """每类结果最多保留 keep 份；轻量标量历史另存，不参与清理。"""
    directory.mkdir(parents=True, exist_ok=True)
    index = int(payload["step"])
    path = directory / f"{prefix}_{index:09d}.json"
    # 外部多次评估同一 checkpoint 需要独立编号，以免覆盖另一组 split。
    if prefix == "evaluation":
        previous = sorted(directory.glob("evaluation_*.json"))
        index = int(previous[-1].stem.split("_")[-1]) + 1 if previous else 1
        path = directory / f"evaluation_{index:09d}.json"
    atomic_json(path, payload)
    prune_files(directory, f"{prefix}_*.json", keep)
    return path
