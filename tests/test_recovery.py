import copy
import json
import os

import pytest
import torch
from conftest import TinyDecisionModel, row

from my_jev.artifacts import atomic_json, file_hash, save_checkpoint, save_evaluation
from my_jev.checkpoint import read_checkpoint
from my_jev.data import audit_dataset, verify_manifest
from my_jev.inference import load_run
from my_jev.losses import decision_loss
from my_jev.trainer import train


def test_rolling_retention_and_atomic_failure(tmp_path):
    for step in range(1, 8):
        save_checkpoint(tmp_path, {"cursor": {"step": step}}, 3)
        save_evaluation(tmp_path / "evaluation", {"step": step}, 3, prefix="evaluation")
    assert [x.name for x in sorted(tmp_path.glob("step_*.pt"))] == [
        "step_000000005.pt",
        "step_000000006.pt",
        "step_000000007.pt",
    ]
    assert os.stat(tmp_path / "latest.pt").st_ino == os.stat(tmp_path / "step_000000007.pt").st_ino
    assert len(list((tmp_path / "evaluation").glob("*.json"))) == 3
    path = tmp_path / "valid.json"
    atomic_json(path, {"valid": True})
    before = file_hash(path)
    with pytest.raises(ValueError):
        atomic_json(path, {"invalid": float("nan")})
    assert file_hash(path) == before
    assert not list(tmp_path.glob(".incomplete-*"))


def test_real_training_loop_resume_equivalence(tiny_config):
    continuous = train(tiny_config, model_factory=TinyDecisionModel)
    final = read_checkpoint(continuous / "checkpoints/latest.pt")
    interrupted = train(tiny_config, stop_after_steps=2, model_factory=TinyDecisionModel)
    # 模拟 best 发布后、下一恢复档发布前中断：必须从档内 best 字节还原。
    (interrupted / "checkpoints/best.pt").write_bytes(b"simulated interrupted best publication")
    resumed = train(
        tiny_config, resume=interrupted / "checkpoints/latest.pt", model_factory=TinyDecisionModel
    )
    restored = read_checkpoint(resumed / "checkpoints/latest.pt")
    assert final["cursor"] == restored["cursor"]
    for name in final["parameters"]:
        assert torch.equal(final["parameters"][name], restored["parameters"][name]), name
    for index, state in final["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, restored["optimizer"]["state"][index][key])
    assert torch.equal(final["rng"]["torch"], restored["rng"]["torch"])
    assert json.loads((resumed / "status.json").read_text())["test_evaluated"] is False
    assert len(list((resumed / "checkpoints").glob("step_*.pt"))) == 3
    assert len(list((resumed / "validation").glob("step_*.json"))) == 3
    events = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [e["rows"] for e in events if e["event"] == "train"] == [3, 3, 1, 3, 3, 1]
    _, _, temperature, _, _ = load_run(resumed, model_factory=TinyDecisionModel)
    assert temperature > 0
    with pytest.raises(ValueError, match="温度不属于"):
        load_run(resumed, "latest", model_factory=TinyDecisionModel)


def test_training_rejects_changed_data(tiny_config):
    interrupted = train(tiny_config, stop_after_steps=1, model_factory=TinyDecisionModel)
    from pathlib import Path

    path = Path(tiny_config["data"]["directory"]) / "train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        train(
            tiny_config,
            resume=interrupted / "checkpoints/latest.pt",
            model_factory=TinyDecisionModel,
        )


def test_resume_older_periodic_snapshot(tiny_config):
    directory = train(tiny_config, model_factory=TinyDecisionModel)
    original = read_checkpoint(directory / "checkpoints/latest.pt")
    train(
        tiny_config,
        resume=directory / "checkpoints/step_000000004.pt",
        model_factory=TinyDecisionModel,
    )
    recovered = read_checkpoint(directory / "checkpoints/latest.pt")
    assert original["cursor"] == recovered["cursor"]
    assert all(
        torch.equal(value, recovered["parameters"][name])
        for name, value in original["parameters"].items()
    )
    assert list(directory.glob("recovery_history_*.jsonl"))


def test_group_leakage_and_manifest(tiny_config):
    from pathlib import Path

    directory = Path(tiny_config["data"]["directory"])
    verify_manifest(directory)
    path = directory / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["group_id"] = "group-train-0"
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(ValueError, match="group 跨 split"):
        audit_dataset(directory)


def test_small_fixture_overfit(tiny_config):
    torch.manual_seed(42)
    model = TinyDecisionModel(tiny_config)
    model.backbone.dropout.p = 0
    record = row(kind="noul")
    target = torch.tensor([0.0, 1.0])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.03)
    initial = decision_loss(model([record])[0], target, 0.1)[0].item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = decision_loss(model([record])[0], target, 0.1)[0]
        loss.backward()
        optimizer.step()
    final = decision_loss(model([record])[0], target, 0.1)[0].item()
    assert final < initial * 0.1


def test_parameter_loading_rejects_invalid_shape(tiny_config):
    model = TinyDecisionModel(tiny_config)
    state = copy.deepcopy(model.trainable_state())
    state["head.bias"] = torch.zeros(2)
    with pytest.raises(ValueError, match="shape"):
        model.load_trainable_state(state)


def test_true_batch_gradients_and_partial_weighting(tiny_config):
    """B=3 的前向与逐记录累积按决策等权；最后不完整微批不能单独等权。"""
    from my_jev.trainer import backward_decisions

    torch.manual_seed(42)
    original = TinyDecisionModel(tiny_config)
    original.backbone.dropout.p = 0
    batched = copy.deepcopy(original)
    rows = [row(index=i, kind=kind) for i, kind in enumerate(("choice", "noul", "score", "noul"))]
    settings = tiny_config["train"]
    for record in rows:
        backward_decisions(original, [record], settings, len(rows))
    backward_decisions(batched, rows[:3], settings, len(rows))
    backward_decisions(batched, rows[3:], settings, len(rows))
    for (name, left), (_, right) in zip(original.named_parameters(), batched.named_parameters()):
        if left.requires_grad:
            assert torch.allclose(left.grad, right.grad, atol=1e-6), name


def test_true_batch_resume_with_incomplete_microbatch(tiny_config):
    tiny_config["train"].update(batch_size=2, accumulation=2)
    tiny_config["model"]["sort_candidates"] = True
    continuous = train(tiny_config, model_factory=TinyDecisionModel)
    expected = read_checkpoint(continuous / "checkpoints/latest.pt")
    interrupted = train(tiny_config, stop_after_steps=1, model_factory=TinyDecisionModel)
    resumed = train(
        tiny_config, resume=interrupted / "checkpoints/latest.pt", model_factory=TinyDecisionModel
    )
    actual = read_checkpoint(resumed / "checkpoints/latest.pt")
    assert expected["cursor"] == actual["cursor"]
    assert all(
        torch.equal(value, actual["parameters"][name])
        for name, value in expected["parameters"].items()
    )
    events = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [value["rows"] for value in events if value["event"] == "train"] == [4, 3, 4, 3]
