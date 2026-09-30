import copy
import math

import pytest
import torch
from conftest import TinyDecisionModel, row

from my_jev._upstream.api import candidate_prompts, compile_request, format_response
from my_jev._upstream.metrics import evaluate_probabilities, fit_temperature, softmax
from my_jev.artifacts import atomic_json
from my_jev.config import check_resume_config, load_config, validate_config
from my_jev.data import model_record, validate_row
from my_jev.losses import decision_loss


def test_analytic_loss_and_gradient():
    logits = torch.zeros(2, requires_grad=True)
    target = torch.tensor([0.8, 0.2])
    total, ce, brier = decision_loss(logits, target, 0.1)
    assert ce.item() == pytest.approx(math.log(2))
    assert brier.item() == pytest.approx(0.18)
    total.backward()
    assert torch.allclose(logits.grad, torch.tensor([-0.33, 0.33]), atol=1e-6)


@pytest.mark.parametrize("target", [[1.0], [1.2, -0.2], [0.2, 0.2], [float("nan"), 0.0]])
def test_loss_rejects_invalid_targets(target):
    with pytest.raises((ValueError, FloatingPointError)):
        decision_loss(torch.zeros(2), torch.tensor(target), 0.1)


def test_candidate_permutation_and_padding(tiny_config):
    model = TinyDecisionModel(tiny_config).eval()
    record = row()
    record["options"] = ["short", "a substantially longer candidate", "middle"]
    order = [2, 0, 1]
    original = model([record], 2)[0]
    permuted = model([{**record, "options": [record["options"][i] for i in order]}], 1)[0]
    assert torch.allclose(permuted, original[order], atol=1e-6)
    assert "SECRET_LABEL" not in "".join(candidate_prompts(model_record(record)))
    assert set(model_record(record)) == {"state", "question", "kind", "options"}
    logits = model([row(kind="noul")])[0]
    assert logits.shape == (2,) and logits[0] == 0


def test_typed_score_and_noul():
    records = compile_request(
        "context",
        {
            "choice": {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None}},
            "score": {"type": "score", "instructions": "q", "criteria": ["low", "mid", "high"]},
            "noul": {"type": "noul", "instructions": "q"},
        },
    )
    answers = format_response(records, [[0.2, 0.8], [0.2, 0.3, 0.5], [0.1, 0.9]])["answers"]
    assert answers["choice"]["choice"] == "b"
    assert answers["score"]["score"] == pytest.approx(1.3)
    assert answers["noul"]["noul"] == 0.9


def test_soft_metrics_and_calibration():
    metrics = evaluate_probabilities([[0.25, 0.75]], [[0.25, 0.75]])
    assert metrics["accuracy"] is None
    assert metrics["expected_accuracy"] == 0.75
    assert metrics["brier"] == 0
    logits, targets = [[0, 4], [4, 0]], [[0.2, 0.8], [0.8, 0.2]]
    temperature = fit_temperature(logits, targets)
    assert temperature == pytest.approx(4 / math.log(4), rel=1e-4)
    assert (
        evaluate_probabilities(targets, [softmax(x, temperature) for x in logits])["nll"]
        < (evaluate_probabilities(targets, [softmax(x) for x in logits])["nll"])
    )


def test_strict_config_and_resume(tiny_config):
    changed = copy.deepcopy(tiny_config)
    changed["train"]["learning_rate"] = 0.2
    with pytest.raises(ValueError, match="仅允许"):
        check_resume_config(changed, tiny_config)
    changed = copy.deepcopy(tiny_config)
    changed["train"]["epochs"] += 1
    check_resume_config(changed, tiny_config)
    changed["train"]["epohcs"] = 3
    with pytest.raises(ValueError):
        validate_config(changed)
    with pytest.raises(ValueError, match="未知配置"):
        load_config("configs/qwen35_2b.yaml", ["train.epohcs=3"])


def test_invalid_label_schema():
    for update in ({"target": [0.2]}, {"target": [1.0, -0.5, 0.5]}, {"options": ["a", "a"]}):
        with pytest.raises(ValueError):
            validate_row({**row(), **update}, "train")


def test_saved_json_and_scientific_learning_rate(tmp_path):
    config = load_config("configs/qwen35_2b.yaml")
    path = tmp_path / "config.json"
    atomic_json(path, config)
    assert load_config(path) == config
    assert (
        load_config("configs/qwen35_2b.yaml", ["train.learning_rate=5e-5"])["train"][
            "learning_rate"
        ]
        == 5e-5
    )


def test_length_sorting_restores_decision_order_and_gradients(tiny_config):
    original = TinyDecisionModel(tiny_config)
    original.backbone.dropout.p = 0
    sorted_model = copy.deepcopy(original)
    sorted_model.sort_candidates = True
    records = [row(index=i, kind=kind) for i, kind in enumerate(("score", "noul", "choice"))]
    records[0]["options"] = ["long candidate " * 8, "short", "medium answer"]
    records[1]["state"] += " longer context" * 15
    original_scores = original(records, 2)
    sorted_scores = sorted_model(records, 2)
    for left, right in zip(original_scores, sorted_scores):
        assert torch.allclose(left, right, atol=1e-6)
    for model, scores in ((original, original_scores), (sorted_model, sorted_scores)):
        sum(
            decision_loss(value, torch.tensor(record["target"]), 0.1)[0]
            for record, value in zip(records, scores)
        ).backward()
    for (name, left), (_, right) in zip(
        original.named_parameters(), sorted_model.named_parameters()
    ):
        if left.requires_grad:
            assert torch.allclose(left.grad, right.grad, atol=1e-6), name


def test_batched_evaluation_preserves_rows_and_metrics(tiny_config):
    from my_jev.evaluation import evaluate_model, summarize

    tiny_config["model"]["sort_candidates"] = True
    model = TinyDecisionModel(tiny_config)
    records = [row(index=i, kind=kind) for i, kind in enumerate(("choice", "noul", "score"))]
    single = evaluate_model(model, records, 2, description="fixture", batch_size=1)
    batched = evaluate_model(model, records, 2, description="fixture", batch_size=2)
    assert [value["id"] for value in single] == [value["id"] for value in batched]
    for left, right in zip(single, batched):
        assert left["target"] == right["target"]
        assert left["logits"] == pytest.approx(right["logits"], abs=1e-6)
    assert summarize(single)["overall"]["nll"] == pytest.approx(
        summarize(batched)["overall"]["nll"], abs=1e-6
    )
