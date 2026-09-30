"""只用 train 数据测批处理吞吐和显存；临时权重不保存，不运行正式实验。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from my_jev.artifacts import atomic_json, file_hash
from my_jev.checkpoint import runtime_versions, seed_everything, source_fingerprint
from my_jev.config import load_config
from my_jev.data import model_record, read_split, render_prompts
from my_jev.model import build_model
from my_jev.trainer import backward_decisions, make_optimizer


def parse_case(value: str) -> tuple[int, int, int, bool]:
    """B:G:C:checkpoint 分别是决策微批、累积次数、候选并行数和检查点开关。"""
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError("case 必须是 B:G:C:checkpoint，例如 8:1:32:1")
    batch, accumulation, candidates, checkpoint = map(int, parts)
    if min(batch, accumulation, candidates) < 1 or checkpoint not in (0, 1):
        raise ValueError("批次必须为正数，checkpoint 必须是 0/1")
    return batch, accumulation, candidates, bool(checkpoint)


def update(model, optimizer, rows, settings):
    """复用正式反向函数、裁剪和 optimizer；最后一个有效 batch 采用真实分母。"""
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for start in range(0, len(rows), settings["batch_size"]):
        losses = backward_decisions(
            model, rows[start : start + settings["batch_size"]], settings, len(rows)
        )
        total_loss += losses[0]
    norm = torch.nn.utils.clip_grad_norm_(
        [value for value in model.parameters() if value.requires_grad], settings["grad_clip"]
    )
    if not torch.isfinite(norm):
        raise FloatingPointError("资源测量出现非有限梯度")
    optimizer.step()
    return total_loss / len(rows)


def run_case(args) -> dict:
    config = load_config(args.config)
    batch, accumulation, candidates, checkpoint = parse_case(args.case)
    config["train"].update(
        batch_size=batch, accumulation=accumulation, candidate_batch_size=candidates
    )
    config["model"]["gradient_checkpointing"] = checkpoint
    if args.sort_candidates:
        config["model"]["sort_candidates"] = True
    device = config["runtime"]["device"]
    if not device.startswith("cuda"):
        raise ValueError("资源测量需要 CUDA")
    # 给驱动和其他进程留空间；超过预算的组合标记 OOM，不持续尝试分配整卡。
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    seed_everything(config["experiment"]["seed"])
    rows = read_split(
        config["data"]["directory"], "train", limit=args.samples, seed=config["experiment"]["seed"]
    )
    record = {
        "case": args.case,
        "batch_size": batch,
        "accumulation": accumulation,
        "effective_batch_size": batch * accumulation,
        "candidate_batch_size": candidates,
        "gradient_checkpointing": checkpoint,
        "sort_candidates": config["model"]["sort_candidates"],
        "samples": len(rows),
        "repeats": args.repeats,
        "selected_ids": [row["id"] for row in rows],
        "status": "running",
    }
    sampler = None
    monitor = args.output.with_suffix(".monitor.csv")
    phase = "load"
    try:
        model = build_model(config).train()
        optimizer = make_optimizer(model, config["train"])
        logical_tokens = padded_tokens = 0
        for start in range(0, len(rows), batch):
            prompts = [
                prompt
                for row in rows[start : start + batch]
                for prompt in render_prompts(model_record(row), model.tokenizer)
            ]
            lengths = list(
                map(len, model.tokenizer(prompts, padding=False, truncation=False)["input_ids"])
            )
            logical_tokens += sum(lengths)
            if config["model"]["sort_candidates"]:
                lengths.sort()
            padded_tokens += sum(
                max(chunk) * len(chunk)
                for start in range(0, len(lengths), candidates)
                if (chunk := lengths[start : start + candidates])
            )
        record.update(
            logical_input_tokens=logical_tokens,
            padded_input_tokens=padded_tokens,
            padding_fraction=1 - logical_tokens / padded_tokens,
        )
        phase = "warmup"
        # 热身不计入吞吐；每个候选组合从相同初始化开始，不保存更新后的参数。
        update(model, optimizer, rows[: min(4, len(rows))], config["train"])
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        phase = "measurement"
        durations = []
        with monitor.open("w") as handle:
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=200",
                ],
                stdout=handle,
                stderr=subprocess.DEVNULL,
            )
            started = time.perf_counter()
            for _ in range(args.repeats):
                for start in range(0, len(rows), batch * accumulation):
                    begin = time.perf_counter()
                    loss = update(
                        model,
                        optimizer,
                        rows[start : start + batch * accumulation],
                        config["train"],
                    )
                    torch.cuda.synchronize(device)
                    durations.append(time.perf_counter() - begin)
                    if not math.isfinite(loss):
                        raise FloatingPointError("资源测量出现非有限损失")
            seconds = time.perf_counter() - started
            sampler.terminate()
            sampler.wait(timeout=10)
            sampler = None
        observations = [
            list(map(float, line.split(","))) for line in monitor.read_text().splitlines()
        ]
        record.update(
            measured_decisions=len(rows) * args.repeats,
            measured_seconds=seconds,
            decisions_per_second=len(rows) * args.repeats / seconds,
            median_update_seconds=statistics.median(durations),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            sampled_gpu_utilization_mean=statistics.mean(row[0] for row in observations)
            if observations
            else None,
            sampled_device_memory_max_gib=max(
                (row[1] / 1024 for row in observations), default=None
            ),
        )
        phase = "stress"
        # 两种真实极端记录：最多总 token 和最长单序列；重复组成完整微批。
        audit = json.loads(Path("data/preflight.json").read_text())["splits"]["train"]
        wanted = {audit["largest_decision_id"], audit["longest_row_id"]}
        stress_rows = []
        with (Path(config["data"]["directory"]) / "train.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in wanted:
                    stress_rows.append(row)
        assert len(stress_rows) == len(wanted)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        stress = []
        for row in stress_rows:
            loss = update(
                model, optimizer, [copy.deepcopy(row) for _ in range(batch)], config["train"]
            )
            torch.cuda.synchronize(device)
            stress.append({"id": row["id"], "decisions": batch, "loss": loss})
        record.update(
            stress=stress,
            stress_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            stress_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            status="passed",
        )
    except torch.OutOfMemoryError as error:
        record.update(status="oom", failed_phase=phase, error=str(error))
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=10)
        monitor.unlink(missing_ok=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen35_2b.yaml")
    parser.add_argument(
        "--output", type=Path, default=Path(".verification-tmp/profile/results.json")
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "1:4:4:1",
            "4:1:16:1",
            "4:1:16:0",
            "8:1:32:1",
            "8:1:32:0",
            "16:1:32:1",
            "16:1:64:1",
        ],
    )
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--sort-candidates",
        action="store_true",
        help="在微批内按长度分块，避免长短序列大量 padding",
    )
    args = parser.parse_args()
    if args.samples < 4 or args.repeats < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("samples 至少 4，repeats 为正数，memory-fraction 在 (0,1] 内")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.case:
        result = run_case(args)
        atomic_json(args.output, result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "case",
                        "status",
                        "decisions_per_second",
                        "peak_allocated_gib",
                        "stress_peak_allocated_gib",
                        "failed_phase",
                    )
                }
            )
        )
        return
    for case in args.cases:
        parse_case(case)
    results = []
    for index, case in enumerate(tqdm(args.cases, desc="资源组合", unit="case")):
        output = args.output.with_name(f"case_{index}.json")
        # 每组独立进程；OOM 后不复用损坏的图或缓存，不影响后续组合。
        command = [
            sys.executable,
            __file__,
            "--config",
            args.config,
            "--case",
            case,
            "--output",
            str(output),
            "--samples",
            str(args.samples),
            "--repeats",
            str(args.repeats),
            "--memory-fraction",
            str(args.memory_fraction),
        ]
        if args.sort_candidates:
            command.append("--sort-candidates")
        subprocess.run(command, check=True)
        results.append(json.loads(output.read_text()))
        atomic_json(
            args.output,
            {
                "gpu": torch.cuda.get_device_name(),
                "versions": runtime_versions(),
                "source_sha256": source_fingerprint(),
                "config_sha256": file_hash(Path(args.config)),
                "memory_fraction": args.memory_fraction,
                "timing_scope": "warmup/model load excluded; prompt rendering, tokenization, forward, backward, clipping, AdamW and CUDA synchronization included; validation/checkpoint excluded",
                "split": "train only",
                "formal_training_started": False,
                "results": results,
            },
        )


if __name__ == "__main__":
    main()
