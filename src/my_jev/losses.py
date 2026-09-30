"""一条决策的 soft cross-entropy + Brier；候选数不同也保持每条决策等权。"""

import torch


def decision_loss(logits: torch.Tensor, target: torch.Tensor, brier_weight: float):
    """logits/target: [K]；返回 total、CE、Brier，全部以 FP32 计算。"""
    if logits.ndim != 1 or target.shape != logits.shape:
        raise ValueError(f"logits/target shape 不符: {logits.shape} / {target.shape}")
    logits, target = logits.float(), target.to(logits.device).float()
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise FloatingPointError("logits 或 target 非有限")
    if (target < 0).any() or not torch.isclose(target.sum(), target.new_tensor(1.0)):
        raise ValueError("target 必须是有效概率分布")
    ce = -(target * logits.log_softmax(-1)).sum()
    brier = ((logits.softmax(-1) - target) ** 2).sum()
    return ce + brier_weight * brier, ce, brier
