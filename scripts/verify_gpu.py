"""受控 GPU 验收：4 次更新，先中断再恢复；不启动正式实验。"""

import argparse
import json
from pathlib import Path

import torch

from my_jev.artifacts import atomic_json, file_hash
from my_jev.checkpoint import read_checkpoint, runtime_versions, seed_everything
from my_jev.config import load_config
from my_jev.data import model_record, read_split
from my_jev.inference import load_run, predict
from my_jev.model import build_model
from my_jev.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".verification-tmp/gpu"))
    parser.add_argument("--config", default="configs/qwen35_2b.yaml")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    config["experiment"].update(name="gpu_acceptance", output_root=str(arguments.output))
    # 按所验收的真实 B×G 取满 4 次更新；不悄悄降成逐记录训练。
    training_count = 4 * config["train"]["batch_size"] * config["train"]["accumulation"]
    config["data"].update(
        train_limit=training_count,
        validation_limit=4,
        calibration_limit=4,
        test_limit=4,
        ood_limit=4,
    )
    config["train"].update(epochs=1, checkpoint_every=1, eval_every=1)
    seed_everything(config["experiment"]["seed"])
    model = build_model(config).train()
    initial = model.trainable_state()
    train_rows = read_split(
        config["data"]["directory"],
        "train",
        limit=training_count,
        seed=config["experiment"]["seed"],
    )
    # 验收真实 LoRA、head 梯度。不会把这次探测 optimizer 更新用于训练。
    from my_jev.losses import decision_loss

    row = train_rows[0]
    logits = model([model_record(row)], config["train"]["candidate_batch_size"])[0]
    target = torch.tensor(row["target"], dtype=torch.float32, device=logits.device)
    decision_loss(logits, target, config["train"]["brier_weight"])[0].backward()
    gradient_norms = {
        name: p.grad.float().norm().item()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    assert all(torch.isfinite(torch.tensor(list(gradient_norms.values()))))
    assert any("lora_" in name and norm > 0 for name, norm in gradient_norms.items())
    assert gradient_norms["head.weight"] > 0
    # 全量长度预检给出候选 token 总量最大的训练决策，额外检查实际显存压力。
    length_audit = json.loads(Path("data/preflight.json").read_text())["splits"]["train"]
    with (Path(config["data"]["directory"]) / "train.jsonl").open() as handle:
        stress_row = next(
            value
            for line in handle
            if (value := json.loads(line))["id"] == length_audit["largest_decision_id"]
        )
    model.zero_grad(set_to_none=True)
    logits = model([model_record(stress_row)], config["train"]["candidate_batch_size"])[0]
    target = torch.tensor(stress_row["target"], dtype=torch.float32, device=logits.device)
    stress_loss = decision_loss(logits, target, config["train"]["brier_weight"])[0]
    stress_loss.backward()
    assert torch.isfinite(stress_loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    stress = {
        "id": stress_row["id"],
        "candidates": len(stress_row["options"]),
        "tokens": length_audit["largest_decision_tokens"],
        "loss": stress_loss.item(),
    }
    del stress_loss
    del model, logits, target
    torch.cuda.empty_cache()
    continuous_run = train(config)
    continuous = read_checkpoint(continuous_run / "checkpoints/latest.pt")
    torch.cuda.empty_cache()
    run = train(config, stop_after_steps=2)
    interrupted = read_checkpoint(run / "checkpoints/latest.pt")
    assert interrupted["cursor"]["step"] == 2
    run = train(config, resume=run / "checkpoints/latest.pt")
    restored = read_checkpoint(run / "checkpoints/latest.pt")
    assert restored["cursor"]["step"] == 4
    assert restored["cursor"]["epoch"] == 1
    resume_difference = max(
        (continuous["parameters"][name] - restored["parameters"][name]).float().abs().max().item()
        for name in initial
    )
    # 同机同配置有数值误差时显式给出边界；CPU 对照另行要求逐位相等。
    assert resume_difference <= 1e-5, f"GPU 恢复参数偏差过大: {resume_difference}"
    changes = {
        name: (restored["parameters"][name] - initial[name]).float().abs().max().item()
        for name in initial
    }
    assert any("lora_" in name and difference > 0 for name, difference in changes.items())
    assert changes["head.weight"] > 0
    assert len(list((run / "checkpoints").glob("step_*.pt"))) == 3
    assert len(list((run / "validation").glob("step_*.json"))) == 3
    model, _, temperature, _, _ = load_run(run)
    request = json.loads(Path("examples/request.json").read_text())
    candidate_batch = config["train"]["candidate_batch_size"]
    result = predict(model, request, temperature=temperature, candidate_batch_size=candidate_batch)
    assert {value["type"] for value in result["answers"].values()} == {"choice", "noul", "score"}
    # 缓存是实验性优化，记录数值差异，不把不同计算路径宣称为逐位一致。
    with torch.inference_mode():
        uncached = model([model_record(value) for value in train_rows[:2]], candidate_batch)
        cached, cache_stats = model.score_cached(
            [model_record(value) for value in train_rows[:2]], candidate_batch
        )
    cache_difference = max((a - b).abs().max().item() for a, b in zip(uncached, cached))
    assert all(torch.isfinite(value).all() for value in cached)
    del model, uncached, cached
    torch.cuda.empty_cache()
    reloaded, _, _, _, _ = load_run(run)
    repeated = predict(
        reloaded, request, temperature=temperature, candidate_batch_size=candidate_batch
    )
    assert result == repeated, "同一路径重新加载权重后的结果应一致"
    report = {
        "gpu": torch.cuda.get_device_name(),
        "versions": runtime_versions(),
        "torch_cuda": torch.version.cuda,
        "run": str(run),
        "updates": 4,
        "training_decisions": training_count,
        "batch_size": config["train"]["batch_size"],
        "accumulation": config["train"]["accumulation"],
        "candidate_batch_size": config["train"]["candidate_batch_size"],
        "gradient_checkpointing": config["model"]["gradient_checkpointing"],
        "interrupted_at": 2,
        "resumed_to": 4,
        "gpu_resume_max_parameter_difference": resume_difference,
        "trainable_parameters": sum(value.numel() for value in initial.values()),
        "head_max_update": changes["head.weight"],
        "max_lora_update": max(value for name, value in changes.items() if "lora_" in name),
        "gradient_norms": gradient_norms,
        "largest_training_decision": stress,
        "checkpoint_sha256": file_hash(run / "checkpoints/best.pt"),
        "temperature": temperature,
        "reload_equal": True,
        "prefix_cache_max_logit_difference": cache_difference,
        "cache_stats": cache_stats,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "formal_training_started": False,
    }
    atomic_json(arguments.output / "report.json", report)
    atomic_json(arguments.output / "example_response.json", result)
    print(json.dumps({"run": str(run), "report": str(arguments.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
