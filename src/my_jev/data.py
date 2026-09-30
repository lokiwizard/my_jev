"""保留原始 split 和软标签；白名单字段构造模型输入，审计字段不进入 prompt。"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from tqdm.auto import tqdm

from ._upstream.api import candidate_prompts
from .artifacts import file_hash, object_hash

SPLITS = ("train", "validation", "calibration", "test", "ood")


def validate_row(row: dict, split: str) -> None:
    """target: [K]、有限非负、和为 1；Noul 必须按照 [no, yes] 排列。"""
    required = {
        "id",
        "group_id",
        "split",
        "source",
        "state",
        "question",
        "kind",
        "options",
        "target",
    }
    if not required <= row.keys():
        raise ValueError(f"记录缺少字段: {required - row.keys()}")
    if row["split"] != split or split not in SPLITS:
        raise ValueError(f"记录 {row['id']} 的 split 与文件不一致")
    for key in ("id", "group_id", "source", "question"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{key} 必须是非空文本")
    if row["kind"] not in ("choice", "noul", "score"):
        raise ValueError(f"未知 kind: {row['kind']}")
    if not isinstance(row["state"], (str, dict, list)) or not row["state"]:
        raise ValueError("state 必须是非空文本、对象或列表")
    json.dumps(row["state"], allow_nan=False)
    options, target = row["options"], row["target"]
    if not isinstance(options, list) or len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("训练 options 至少包含两个不同候选")
    if not all(isinstance(option, str) and option.strip() for option in options):
        raise ValueError("候选描述必须是非空文本")
    if not isinstance(target, list) or len(target) != len(options):
        raise ValueError(f"target 和 options 长度不一致: {row['id']}")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in target
    ):
        raise ValueError("target 必须有限且非负")
    if not math.isclose(sum(target), 1.0, abs_tol=1e-6, rel_tol=1e-6):
        raise ValueError(f"target 的和不为 1: {row['id']}")
    if row["kind"] == "noul" and options != ["no", "yes"]:
        raise ValueError("Noul options 必须是 [no, yes]")


def model_record(row: dict) -> dict:
    """主动建立白名单；target/metadata/id/source 等不会混入模型文本。"""
    return {key: row[key] for key in ("state", "question", "kind", "options")}


def read_split(directory: str | Path, split: str, *, limit: int = 0, seed: int = 42) -> list:
    rows = []
    with (Path(directory) / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                validate_row(row, split)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{split}.jsonl:{line_number}: {error}") from error
            rows.append(row)
    if not rows:
        raise ValueError(f"{split} 为空")
    # 限定的检查集用固定 shuffle 选取，避免只取数据文件开头的单一来源。
    if limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def audit_dataset(directory: str | Path) -> dict:
    """逐 ID、group、完整输入检查泄漏，返回实际数据字节与分布的清单。"""
    directory = Path(directory)
    ids, groups, inputs = set(), {}, {}
    result = {"files": {}, "counts": {}, "sources": {}, "kinds": {}}
    for split in SPLITS:
        rows = read_split(directory, split)
        sources, kinds = {}, {}
        for row in tqdm(rows, desc=f"audit {split}", unit="decision"):
            if row["id"] in ids:
                raise ValueError(f"重复 ID: {row['id']}")
            ids.add(row["id"])
            group = row["group_id"]
            if group in groups and groups[group] != split:
                raise ValueError(f"group 跨 split: {group}")
            groups[group] = split
            # 排序候选后指纹仍一致，不能通过候选置换掩盖泄漏。
            fingerprint = object_hash({**model_record(row), "options": sorted(row["options"])})
            labels = dict(zip(row["options"], row["target"]))
            if fingerprint in inputs:
                previous_split, previous_target = inputs[fingerprint]
                if previous_split != split:
                    raise ValueError(f"相同模型输入跨 split: {row['id']}")
                if previous_target != labels:
                    raise ValueError(f"相同模型输入具有冲突标签: {row['id']}")
            inputs[fingerprint] = split, labels
            sources[row["source"]] = sources.get(row["source"], 0) + 1
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        result["files"][f"{split}.jsonl"] = file_hash(directory / f"{split}.jsonl")
        result["counts"][split] = len(rows)
        result["sources"][split], result["kinds"][split] = sources, kinds
    return result


def verify_manifest(directory: str | Path, source: dict | None = None) -> dict:
    """训练启动及续训均核验原始文件，避免数据被替换后继续同一实验。"""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if source is not None:
        expected = {key: source[key] for key in ("repo", "revision", "config")}
        if manifest["source"] != expected:
            raise ValueError("数据 manifest 来源与配置不一致")
    if set(manifest["audit"]["files"]) != {f"{split}.jsonl" for split in SPLITS}:
        raise ValueError("manifest 必须覆盖全部五个 split 的文件")
    for filename, expected in manifest["audit"]["files"].items():
        if file_hash(directory / filename) != expected:
            raise ValueError(f"数据文件与 manifest 不符: {filename}")
    return manifest


def render_prompts(row: dict, tokenizer) -> list[str]:
    """只使用白名单字段并沿用 upstream chat template；不生成答案 token。"""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in candidate_prompts(model_record(row))
    ]


def check_input_lengths(rows: list, tokenizer, max_length: int, *, description: str) -> dict:
    """检查每条完整候选序列，不截断。返回实际 token 预算，便于估算运行量。"""
    maximum = total_tokens = sequences = largest_decision_tokens = 0
    longest_row_id = largest_decision_id = None
    for row in tqdm(rows, desc=description, unit="decision"):
        encoded = tokenizer(render_prompts(row, tokenizer), padding=False, truncation=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if max(lengths) > max_length:
            raise ValueError(f"{row['id']} 输入长 {max(lengths)}，超过 max_length={max_length}")
        if max(lengths) > maximum:
            maximum, longest_row_id = max(lengths), row["id"]
        if sum(lengths) > largest_decision_tokens:
            largest_decision_tokens, largest_decision_id = sum(lengths), row["id"]
        total_tokens += sum(lengths)
        sequences += len(lengths)
    return {
        "rows": len(rows),
        "candidate_sequences": sequences,
        "input_tokens": total_tokens,
        "max_length": maximum,
        "longest_row_id": longest_row_id,
        "largest_decision_id": largest_decision_id,
        "largest_decision_tokens": largest_decision_tokens,
    }
