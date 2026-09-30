"""服务器验收夹具。小模型只验证训练工程语义，不代表 Qwen 的任务效果。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

from my_jev.artifacts import atomic_json
from my_jev.config import load_config
from my_jev.data import SPLITS, audit_dataset
from my_jev.model import DecisionModel


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"] + "\nAssistant:"

    def __call__(self, prompts, *, padding=False, **kwargs):
        encoded = [[byte + 1 for byte in prompt.encode()] for prompt in prompts]
        if kwargs.get("return_tensors") != "pt":
            return {"input_ids": encoded}
        length = max(map(len, encoded))
        return BatchEncoding(
            {
                "input_ids": torch.tensor([row + [0] * (length - len(row)) for row in encoded]),
                "attention_mask": torch.tensor(
                    [[1] * len(row) + [0] * (length - len(row)) for row in encoded]
                ),
            }
        )


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, 8)
        self.embedding.requires_grad_(False)
        self.q_proj = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input_ids, attention_mask, **kwargs):
        assert kwargs["use_cache"] is False
        embedded = self.embedding(input_ids)
        positions = torch.arange(1, embedded.shape[1] + 1).view(1, -1, 1)
        hidden = embedded.cumsum(1) / positions
        return SimpleNamespace(last_hidden_state=self.dropout(self.q_proj(hidden)))


class TinyDecisionModel(DecisionModel):
    """沿用生产 forward/权重保存，而非另写一个训练循环。"""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.tokenizer = TinyTokenizer()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(8, 1)
        self.device_name, self.max_length = "cpu", 4096
        self.sort_candidates = config["model"]["sort_candidates"]


def row(split="train", index=0, *, kind="choice"):
    options = ["no", "yes"] if kind == "noul" else ["alpha", "beta", "gamma"]
    target = [0.2, 0.8] if kind == "noul" else [0.1, 0.7, 0.2]
    return {
        "id": f"{split}-{index}",
        "group_id": f"group-{split}-{index}",
        "split": split,
        "source": "fixture",
        "kind": kind,
        "state": f"independent fixture {split} {index}",
        "question": "Choose a response",
        "options": options,
        "target": target,
        "metadata": {"correct": "SECRET_LABEL"},
    }


@pytest.fixture
def tiny_config(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/qwen35_2b.yaml")
    config["data"].update(repo="fixture/testing", revision="0" * 40, config="synthetic")
    directory = tmp_path / "data"
    directory.mkdir()
    for split in SPLITS:
        count = 7 if split == "train" else 3
        rows = [
            row(split, index, kind=["choice", "score", "noul"][index % 3]) for index in range(count)
        ]
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in rows)
        )
    atomic_json(
        directory / "manifest.json",
        {
            "source": {key: config["data"][key] for key in ("repo", "revision", "config")},
            "fixture": True,
            "audit": audit_dataset(directory),
        },
    )
    config["data"]["directory"] = str(directory)
    config["experiment"]["output_root"] = str(tmp_path / "runs")
    config["train"].update(
        epochs=2,
        accumulation=3,
        checkpoint_every=1,
        eval_every=1,
        learning_rate=0.01,
        head_learning_rate=0.01,
    )
    config["runtime"]["device"] = "cpu"
    return config
