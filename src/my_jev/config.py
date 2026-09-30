"""严格配置校验：拒绝拼错的字段，保存命令行覆盖后的有效配置。"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import yaml

SCHEMA = {
    "experiment": {"name": str, "seed": int, "output_root": str},
    "model": {
        "id": str,
        "revision": str,
        "dtype": str,
        "max_length": int,
        "lora_rank": int,
        "lora_alpha": int,
        "lora_dropout": float,
        "gradient_checkpointing": bool,
        "sort_candidates": bool,
    },
    "data": {
        "repo": str,
        "revision": str,
        "config": str,
        "directory": str,
        "train_limit": int,
        "validation_limit": int,
        "calibration_limit": int,
        "test_limit": int,
        "ood_limit": int,
    },
    "train": {
        "epochs": int,
        "max_steps": int,
        "learning_rate": float,
        "head_learning_rate": float,
        "weight_decay": float,
        "batch_size": int,
        "accumulation": int,
        "brier_weight": float,
        "grad_clip": float,
        "candidate_batch_size": int,
        "checkpoint_every": int,
        "eval_every": int,
        "checkpoint_keep": int,
        "eval_keep": int,
    },
    "runtime": {"device": str, "deterministic": bool},
}


def validate_config(config: dict) -> dict:
    """所有数值必须有限；0 只在明确表示 unlimited/disabled 的字段中合法。"""
    if not isinstance(config, dict) or set(config) != set(SCHEMA):
        raise ValueError(f"配置必须包含且仅包含 {sorted(SCHEMA)}")
    for section, fields in SCHEMA.items():
        values = config[section]
        if not isinstance(values, dict) or set(values) != set(fields):
            raise ValueError(f"{section} 字段应为 {sorted(fields)}，实际为 {values}")
        for key, expected in fields.items():
            value = values[key]
            allowed = (int, float) if expected is float else (expected,)
            if type(value) not in allowed:
                raise ValueError(f"{section}.{key} 类型错误: {value!r}")
            if expected is str and not value.strip():
                raise ValueError(f"{section}.{key} 不能为空")
            if expected in (int, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{section}.{key} 必须有限且非负")
    positive = {
        "model": ["max_length", "lora_alpha"],
        "train": [
            "epochs",
            "learning_rate",
            "head_learning_rate",
            "batch_size",
            "accumulation",
            "grad_clip",
            "candidate_batch_size",
            "checkpoint_keep",
            "eval_keep",
        ],
    }
    for section, keys in positive.items():
        for key in keys:
            if config[section][key] <= 0:
                raise ValueError(f"{section}.{key} 必须大于 0")
    if config["model"]["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("model.dtype 只支持 bfloat16 / float32")
    if not 0 <= config["model"]["lora_dropout"] < 1:
        raise ValueError("lora_dropout 必须在 [0, 1) 内")
    for section in ("model", "data"):
        if not re.fullmatch(r"[0-9a-f]{40}", config[section]["revision"]):
            raise ValueError(f"{section}.revision 必须是完整 40 位 commit SHA")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", config["experiment"]["name"]):
        raise ValueError("experiment.name 只能包含字母、数字、下划线和连字符")
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """--set section.key=value 使用 YAML 标量解析，覆盖前检查路径存在。"""
    with Path(path).open(encoding="utf-8") as handle:
        # JSON 的 5e-05 是合法浮点数，PyYAML 1.1 会误读成字符串，需原生解析。
        config = json.load(handle) if Path(path).suffix == ".json" else yaml.safe_load(handle)
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"覆盖参数缺少 =: {expression}")
        key, raw = expression.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2 or parts[0] not in SCHEMA or parts[1] not in SCHEMA[parts[0]]:
            raise ValueError(f"未知配置字段: {key}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = yaml.safe_load(raw)
        config[parts[0]][parts[1]] = value
    return validate_config(config)


def check_resume_config(current: dict, saved: dict) -> None:
    """只允许延长 epochs；其余实验条件改变时应新开实验。优化器使用恒定 LR。"""
    left, right = copy.deepcopy(current), copy.deepcopy(saved)
    if left["train"]["epochs"] < right["train"]["epochs"]:
        raise ValueError("续训不能缩短 train.epochs")
    left["train"].pop("epochs")
    right["train"].pop("epochs")
    if left != right:
        raise ValueError("续训配置与 checkpoint 不一致；仅允许增大 train.epochs")
