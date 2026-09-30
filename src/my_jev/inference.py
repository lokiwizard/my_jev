"""加载实验权重与匹配的温度；推理不需要训练数据文件。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ._upstream.api import compile_request, format_response
from ._upstream.metrics import softmax
from .artifacts import file_hash
from .checkpoint import read_checkpoint
from .config import load_config
from .model import build_model


def load_run(
    run: str | Path,
    checkpoint: str = "best",
    *,
    uncalibrated: bool = False,
    device: str | None = None,
    model_factory=build_model,
):
    """只允许 best/latest；T 必须绑定所加载权重的哈希，不能跨权重套用。"""
    run = Path(run)
    if checkpoint not in ("best", "latest"):
        raise ValueError("checkpoint 只能是 best 或 latest")
    path = run / "checkpoints" / f"{checkpoint}.pt"
    saved = read_checkpoint(path)
    config = load_config(run / "config.json")
    if config["model"] != saved["config"]["model"]:
        raise ValueError("实验配置与 checkpoint 的模型定义不一致")
    temperature = 1.0
    if not uncalibrated:
        calibration = json.loads((run / "temperature.json").read_text())
        if calibration["checkpoint_sha256"] != file_hash(path):
            raise ValueError("温度不属于当前权重；latest 请显式使用 --uncalibrated")
        temperature = float(calibration["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("温度必须有限且为正")
    if device:
        config["runtime"]["device"] = device
    model = model_factory(config)
    model.load_trainable_state(saved["parameters"])
    model.eval()
    return model, config, temperature, saved, path


@torch.inference_mode()
def predict(
    model,
    request: dict,
    *,
    temperature: float = 1.0,
    candidate_batch_size: int = 4,
    prefix_cache: bool = False,
) -> dict:
    """Choice=argmax，Noul=P(true)，Score=Σ(index × P(index))，均不生成文本。"""
    if set(request) != {"state", "questions"}:
        raise ValueError("请求必须且只包含 state 和 questions")
    records = compile_request(request["state"], request["questions"])
    model.eval()
    if prefix_cache:
        logits, cache_metrics = model.score_cached(records, candidate_batch_size)
    else:
        logits, cache_metrics = model(records, candidate_batch_size), None
    values = [row.detach().float().cpu().tolist() for row in logits]
    if any(not all(math.isfinite(value) for value in row) for row in values):
        raise FloatingPointError("推理 logits 非有限")
    result = format_response(records, [softmax(row, temperature) for row in values])
    result["scoring"] = {
        "temperature": temperature,
        "prefix_cache": prefix_cache,
        "logits": dict(zip((row["id"] for row in records), values)),
    }
    if cache_metrics is not None:
        result["scoring"]["cache_metrics"] = cache_metrics
    return result
