"""eval/no_grad 下保存逐记录 logits；指标按决策平均，软标签不伪造硬类别。"""

from __future__ import annotations

from collections import defaultdict

import torch
from tqdm.auto import tqdm

from ._upstream.metrics import evaluate_probabilities, fit_temperature, softmax
from .data import model_record


def summarize(predictions: list[dict], temperature: float = 1.0) -> dict:
    """总体与每个 kind/source 共用完整记录统计；Score 用期望等级的 MAE。"""
    groups = defaultdict(list)
    for prediction in predictions:
        groups["all"].append(prediction)
        groups[f"kind:{prediction['kind']}"].append(prediction)
        groups[f"source:{prediction['source']}"].append(prediction)
    summaries = {}
    for name, values in groups.items():
        probabilities = [softmax(row["logits"], temperature) for row in values]
        metrics = evaluate_probabilities([row["target"] for row in values], probabilities)
        if name == "kind:score":
            errors = [
                abs(
                    sum(i * p for i, p in enumerate(predicted))
                    - sum(i * y for i, y in enumerate(row["target"]))
                )
                for row, predicted in zip(values, probabilities)
            ]
            metrics["ordinal_mae"] = sum(errors) / len(errors)
        if name == "kind:noul":
            hard = [
                (row, p) for row, p in zip(values, probabilities) if sorted(row["target"]) == [0, 1]
            ]
            tp = sum(row["target"][1] == 1 and p[1] >= 0.5 for row, p in hard)
            fp = sum(row["target"][1] == 0 and p[1] >= 0.5 for row, p in hard)
            fn = sum(row["target"][1] == 1 and p[1] < 0.5 for row, p in hard)
            metrics["hard_f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        summaries[name] = metrics
    return {"overall": summaries.pop("all"), "groups": summaries}


@torch.inference_mode()
def evaluate_model(
    model, rows: list, candidate_batch_size: int, *, description: str, batch_size: int = 1
) -> list:
    """进度条按决策记录计数；验证后由调用者决定是否恢复 train 模式。"""
    model.eval()
    if batch_size < 1:
        raise ValueError("评估 batch_size 必须为正数")
    predictions = []
    with tqdm(total=len(rows), desc=description, unit="decision") as progress:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            values = model([model_record(row) for row in batch], candidate_batch_size)
            if len(values) != len(batch):
                raise ValueError("评估 logits 与决策数量不符")
            for row, logits in zip(batch, values):
                logits = logits.float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"评估 logits 非有限: {row['id']}")
                predictions.append(
                    {
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "source": row["source"],
                        "kind": row["kind"],
                        "options": row["options"],
                        "target": row["target"],
                        "logits": logits.cpu().tolist(),
                    }
                )
            progress.update(len(batch))
    return predictions


def calibrate(predictions: list) -> float:
    """只能传 calibration 的预测，由调用者负责 split 隔离。"""
    return fit_temperature(
        [row["logits"] for row in predictions], [row["target"] for row in predictions]
    )
