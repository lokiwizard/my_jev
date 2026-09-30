"""固定 backbone + 完整 optimizer/RNG/数据游标恢复，支持 epoch 内 optimizer 边界。"""

from __future__ import annotations

import importlib.metadata
import io
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_bytes, file_hash, object_hash


def configure_runtime(runtime: dict):
    """在第一次数值计算前固定 CUDA 后端算法；恢复配置也绑定这个开关。"""
    deterministic = runtime["deterministic"]
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("checkpoint 与当前可见 CUDA 设备数量不一致")
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def runtime_versions() -> dict:
    result = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate", "my-jev")
    }
    result.update(python=sys.version.split()[0], numpy=np.__version__)
    return result


def save_source_snapshot(directory: Path) -> None:
    """新实验保留小型源码归档，代码以后改动时仍能找回与恢复档匹配的实现。"""
    root = Path(__file__).resolve().parents[2]
    files = list((root / "src/my_jev").rglob("*.py"))
    files += list((root / "scripts").glob("*.py"))
    files += list((root / "scripts").glob("*.sh"))
    files += list((root / "configs").glob("*.yaml"))
    files += [
        root / "src/my_jev/_upstream/LICENSE",
        root / "THIRD_PARTY_NOTICES.md",
        root / "README.md",
    ]
    files += [
        root / name for name in ("train.py", "eval.py", "infer.py", "pyproject.toml", "uv.lock")
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    atomic_bytes(directory / "source_snapshot.zip", buffer.getvalue())


def source_fingerprint() -> str:
    """恢复身份只绑定实现文件；README/测试说明更新不改变训练含义。"""
    root = Path(__file__).resolve().parents[2]
    files = {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in (root / "src/my_jev").rglob("*.py")
    }
    for name in ("train.py", "eval.py", "pyproject.toml", "uv.lock"):
        if (root / name).exists():
            files[name] = file_hash(root / name)
    return object_hash(files)


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    """独立本地 RNG 产生每个 epoch 的顺序，不消耗模型/dropout 的 RNG。"""
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def weights_payload(model, config: dict, step: int, validation: dict) -> dict:
    """best.pt 只保留推理权重及身份，不额外复制 AdamW 状态。"""
    return {
        "format_version": 1,
        "kind": "weights_only",
        "config": config,
        "step": step,
        "validation": validation,
        "parameters": model.trainable_state(),
    }


def read_checkpoint(path: str | Path) -> dict:
    """仅加载本项目生成的可信 checkpoint；含 RNG 的恢复档需要 weights_only=False。"""
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("format_version") != 1 or value.get("kind") not in ("training", "weights_only"):
        raise ValueError("不是本项目支持的 checkpoint")
    return value
