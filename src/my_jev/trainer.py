"""单设备训练：按决策累积梯度，按 validation NLL 选模，optimizer 边界保存游标。"""

from __future__ import annotations

import json
import logging
import math
import signal
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .artifacts import (
    atomic_bytes,
    atomic_json,
    atomic_torch,
    file_hash,
    object_hash,
    save_checkpoint,
    save_evaluation,
)
from .checkpoint import (
    epoch_order,
    read_checkpoint,
    restore_rng,
    rng_state,
    runtime_versions,
    save_source_snapshot,
    seed_everything,
    source_fingerprint,
    weights_payload,
)
from .config import check_resume_config
from .data import model_record, read_split, verify_manifest
from .evaluation import calibrate, evaluate_model, summarize
from .losses import decision_loss
from .model import build_model


def make_run_directory(config: dict) -> Path:
    """每次新实验有独立目录，已有目录永不覆盖。"""
    root = Path(config["experiment"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    name = f"{config['experiment']['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    path, suffix = root / name, 0
    while path.exists():
        suffix += 1
        path = root / f"{name}_{suffix}"
    path.mkdir()
    return path


def make_logger(directory: Path) -> logging.Logger:
    logger = logging.getLogger(f"my_jev.{directory.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(directory / "train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    logger.propagate = False
    return logger


def append_event(directory: Path, event: dict):
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def reconcile_history(directory: Path, step: int):
    """恢复较早快照时，把未来标量记录留作审计，当前历史截回 checkpoint 游标。"""
    path = directory / "metrics.jsonl"
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    retained, future = [], []
    for line in lines:
        record = json.loads(line)
        (retained if record.get("step", 0) <= step else future).append(line)
    if future:
        archive = directory / f"recovery_history_{time.time_ns()}.jsonl"
        archive.write_text("\n".join(future) + "\n")
        path.write_text("\n".join(retained) + "\n")


def make_optimizer(model, settings: dict):
    """正式训练与资源基准使用相同 AdamW 参数组。"""
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": settings["learning_rate"],
            },
            {"params": list(model.head.parameters()), "lr": settings["head_learning_rate"]},
        ],
        weight_decay=settings["weight_decay"],
    )


def backward_decisions(model, rows: list[dict], settings: dict, denominator: int) -> tuple:
    """一个真实微批次前向/反向；分母为本次更新的真实决策总数。

    每条决策先重组全部候选 logits[K]，再独立计算 CE/Brier。
    跨决策取和后反向，不能把不同 K 的候选混在一个 softmax 中。
    返回脱离计算图的 loss/CE/Brier 总和，供按决策统计。
    """
    if not rows or denominator < len(rows):
        raise ValueError("微批次不能为空，denominator 不能小于实际决策数")
    logits = model([model_record(row) for row in rows], settings["candidate_batch_size"])
    terms = torch.stack(
        [
            torch.stack(
                decision_loss(
                    scores,
                    torch.tensor(row["target"], device=scores.device, dtype=torch.float32),
                    settings["brier_weight"],
                )
            )
            for row, scores in zip(rows, logits)
        ]
    )
    if len(logits) != len(rows) or not torch.isfinite(terms).all():
        raise FloatingPointError("微批次 logits 数量不符或 loss 非有限")
    totals = terms.sum(0)
    (totals[0] / denominator).backward()
    return tuple(totals.detach().cpu().tolist())


def train(
    config: dict,
    *,
    resume: Path | None = None,
    stop_after_steps: int = 0,
    model_factory=build_model,
) -> Path:
    """停止请求在一次完整更新后处理；不把未完成累积梯度当成恢复点。

    stop_after_steps 仅用于受控验收，不改变 checkpoint 中的正式训练预算。
    真实 SIGINT/SIGTERM 也会请求在下一 optimizer 边界写最新 checkpoint。
    """
    settings = config["train"]
    seed_everything(config["experiment"]["seed"])
    manifest = verify_manifest(config["data"]["directory"], config["data"])
    selected = {}
    for split in ("train", "validation", "calibration"):
        selected[split] = read_split(
            config["data"]["directory"],
            split,
            limit=config["data"][f"{split}_limit"],
            seed=config["experiment"]["seed"],
        )
    identity = {
        "data_manifest_sha256": object_hash(manifest),
        "selected_ids_sha256": {
            split: object_hash([row["id"] for row in rows]) for split, rows in selected.items()
        },
        "source_sha256": source_fingerprint(),
        "runtime": runtime_versions(),
    }
    saved = read_checkpoint(resume) if resume else None
    if saved:
        if saved["kind"] != "training":
            raise ValueError("best.pt 只供推理/评估，续训请使用 latest.pt 或周期恢复档")
        check_resume_config(config, saved["config"])
        if saved["identity"] != identity:
            raise ValueError("续训的数据、代码实现或依赖版本与 checkpoint 不一致")
        directory = Path(resume).resolve().parent.parent
    else:
        directory = make_run_directory(config)
    # Linux advisory lock 随进程退出自动释放，防止两个进程写同一个实验。
    import fcntl

    lock_handle = (directory / ".run.lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError("同一实验已有训练进程") from error
    logger = make_logger(directory)
    atomic_json(directory / "config.json", config)
    cursor = {"epoch": 0, "offset": 0, "step": 0, "epoch_loss": 0.0, "epoch_rows": 0}
    best = None
    last_validation_step = -1
    requested_stop = {"value": False}
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda *_: requested_stop.update(value=True))
    try:
        if not saved:
            save_source_snapshot(directory)
        model = model_factory(config)
        optimizer = make_optimizer(model, settings)
        if saved:
            model.load_trainable_state(saved["parameters"])
            optimizer.load_state_dict(saved["optimizer"])
            cursor = saved["cursor"]
            best = saved["best"]
            last_validation_step = saved["last_validation_step"]
            best_path = directory / "checkpoints/best.pt"
            if best:
                # best 更新与 checkpoint 发布之间硬中断，也能还原该恢复点的选模状态。
                if not best_path.exists() or file_hash(best_path) != best["sha256"]:
                    atomic_bytes(best_path, saved["best_file_bytes"])
                if file_hash(best_path) != best["sha256"]:
                    raise ValueError("恢复档内嵌的 best 字节校验失败")
                best_payload = read_checkpoint(best_path)
                atomic_json(
                    directory / "best_validation.json",
                    {
                        "step": best["step"],
                        "metrics": best_payload["validation"],
                    },
                )
            else:
                best_path.unlink(missing_ok=True)
                (directory / "best_validation.json").unlink(missing_ok=True)
            # 回退到较早游标后，未来的大文件不属于当前训练历史。
            for folder, pattern in (
                ("checkpoints", "step_*.pt"),
                ("validation", "step_*.json"),
                ("calibration", "step_*.json"),
            ):
                for path in (directory / folder).glob(pattern):
                    if int(path.stem.split("_")[-1]) > cursor["step"]:
                        path.unlink()
            (directory / "temperature.json").unlink(missing_ok=True)
            restore_rng(saved["rng"])
            reconcile_history(directory, cursor["step"])
            logger.info(
                "恢复 step=%s epoch=%s offset=%s", cursor["step"], cursor["epoch"], cursor["offset"]
            )
        else:
            atomic_json(
                directory / "provenance.json",
                {
                    "identity": identity,
                    "dataset": manifest["source"],
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "torch_cuda": torch.version.cuda,
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "selection": "minimum validation soft cross-entropy (NLL)",
                    "training": "independent candidates, CE + Brier, constant AdamW LR",
                },
            )
        logger.info("实验目录: %s", directory.resolve())
        effective_batch = settings["batch_size"] * settings["accumulation"]
        per_epoch = math.ceil(len(selected["train"]) / effective_batch)
        total_steps = per_epoch * settings["epochs"]
        if settings["max_steps"]:
            total_steps = min(total_steps, settings["max_steps"])

        def snapshot():
            # deepcopy 游标防止保存后被训练循环继续修改；parameter 在 CPU 上独立 clone。
            payload = {
                "format_version": 1,
                "kind": "training",
                "config": config,
                "identity": identity,
                "parameters": model.trainable_state(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "cursor": dict(cursor),
                "best": best,
                # LoRA/head 很小；嵌入 best 可避免跨文件发布中断导致 latest 无法恢复。
                "best_file_bytes": (directory / "checkpoints/best.pt").read_bytes()
                if best
                else None,
                "last_validation_step": last_validation_step,
            }
            return save_checkpoint(directory / "checkpoints", payload, settings["checkpoint_keep"])

        def validate():
            nonlocal best, last_validation_step
            predictions = evaluate_model(
                model,
                selected["validation"],
                settings["candidate_batch_size"],
                description="validation",
                batch_size=settings["batch_size"],
            )
            metrics = summarize(predictions)
            nll = metrics["overall"]["nll"]
            if not math.isfinite(nll):
                raise FloatingPointError("validation NLL 非有限")
            path = save_evaluation(
                directory / "validation",
                {
                    "step": cursor["step"],
                    "split": "validation",
                    "metrics": metrics,
                    "predictions": predictions,
                },
                settings["eval_keep"],
                prefix="step",
            )
            improved = best is None or nll < best["nll"]
            if improved:
                best_path = directory / "checkpoints/best.pt"
                atomic_torch(best_path, weights_payload(model, config, cursor["step"], metrics))
                best = {"step": cursor["step"], "nll": nll, "sha256": file_hash(best_path)}
                # 小型 best 指标常驻，大型逐记录验证结果仅滚动保留三份。
                atomic_json(
                    directory / "best_validation.json",
                    {
                        "step": cursor["step"],
                        "metrics": metrics,
                    },
                )
            last_validation_step = cursor["step"]
            append_event(
                directory,
                {
                    "event": "validation",
                    "step": cursor["step"],
                    "metrics": metrics,
                    "is_best": improved,
                    "file": str(path),
                },
            )
            logger.info("validation step=%s nll=%.6f best=%s", cursor["step"], nll, improved)
            model.train()

        if saved:
            # 回退恢复点后立即更新 latest；初始化后再失败也不会跳回未来游标。
            snapshot()

        with tqdm(
            total=total_steps, initial=cursor["step"], desc="train", unit="update"
        ) as progress:
            model.train()
            while cursor["epoch"] < settings["epochs"] and cursor["step"] < total_steps:
                order = epoch_order(
                    len(selected["train"]), config["experiment"]["seed"], cursor["epoch"]
                )
                if cursor["offset"] >= len(order):
                    raise ValueError("恢复游标 offset 超过 epoch 长度")
                indices = order[cursor["offset"] : cursor["offset"] + effective_batch]
                update_started = time.perf_counter()
                if str(config["runtime"]["device"]).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(model.device_name)
                optimizer.zero_grad(set_to_none=True)
                ce_sum = brier_sum = loss_sum = 0.0
                for start in range(0, len(indices), settings["batch_size"]):
                    rows = [
                        selected["train"][index]
                        for index in indices[start : start + settings["batch_size"]]
                    ]
                    # 末尾不足一个有效 batch 时仍除以真实条数，保持决策等权。
                    loss, ce, brier = backward_decisions(model, rows, settings, len(indices))
                    loss_sum += loss
                    ce_sum += ce
                    brier_sum += brier
                trainable = [p for p in model.parameters() if p.requires_grad]
                if not any(p.grad is not None for p in trainable):
                    raise RuntimeError("所有可训练参数均没有梯度")
                norm = torch.nn.utils.clip_grad_norm_(trainable, settings["grad_clip"])
                if not torch.isfinite(norm):
                    raise FloatingPointError("梯度非有限，停止且保留上一个有效 checkpoint")
                optimizer.step()
                resource_metrics = {}
                if str(config["runtime"]["device"]).startswith("cuda"):
                    torch.cuda.synchronize(model.device_name)
                    resource_metrics = {
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device_name)
                        / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(model.device_name)
                        / 2**30,
                    }
                seconds = time.perf_counter() - update_started
                cursor["step"] += 1
                cursor["offset"] += len(indices)
                cursor["epoch_loss"] += loss_sum
                cursor["epoch_rows"] += len(indices)
                epoch_finished = cursor["offset"] == len(order)
                append_event(
                    directory,
                    {
                        "event": "train",
                        "step": cursor["step"],
                        "epoch": cursor["epoch"] + 1,
                        "rows": len(indices),
                        "batch_size": settings["batch_size"],
                        "accumulation": settings["accumulation"],
                        "update_seconds": seconds,
                        "decisions_per_second": len(indices) / seconds,
                        **resource_metrics,
                        "loss": loss_sum / len(indices),
                        "ce": ce_sum / len(indices),
                        "brier": brier_sum / len(indices),
                        "gradient_norm": norm.item(),
                        "learning_rate": settings["learning_rate"],
                        "head_learning_rate": settings["head_learning_rate"],
                    },
                )
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss_sum / len(indices):.4f}",
                    epoch=cursor["epoch"] + 1,
                    best=None if best is None else round(best["nll"], 4),
                    decision_s=f"{len(indices) / seconds:.2f}",
                    gpu_gib=round(resource_metrics.get("peak_allocated_gib", 0), 2),
                )
                if epoch_finished:
                    append_event(
                        directory,
                        {
                            "event": "epoch",
                            "step": cursor["step"],
                            "epoch": cursor["epoch"] + 1,
                            "loss": cursor["epoch_loss"] / cursor["epoch_rows"],
                        },
                    )
                    cursor.update(epoch=cursor["epoch"] + 1, offset=0, epoch_loss=0.0, epoch_rows=0)
                should_validate = epoch_finished or cursor["step"] == total_steps
                if settings["eval_every"]:
                    should_validate |= cursor["step"] % settings["eval_every"] == 0
                if should_validate:
                    validate()
                stop = requested_stop["value"] or (
                    stop_after_steps and cursor["step"] >= stop_after_steps
                )
                if (
                    should_validate
                    or epoch_finished
                    or stop
                    or (
                        settings["checkpoint_every"]
                        and cursor["step"] % settings["checkpoint_every"] == 0
                    )
                ):
                    snapshot()
                if stop:
                    atomic_json(
                        directory / "status.json", {"status": "interrupted", "step": cursor["step"]}
                    )
                    logger.info("已在 optimizer 边界暂停，latest.pt 可以续训")
                    return directory
            if best is None:
                validate()
            snapshot()
        # 最优权重选定之后才使用 calibration；test/OOD 不会进入训练入口。
        best_payload = read_checkpoint(directory / "checkpoints/best.pt")
        model.load_trainable_state(best_payload["parameters"])
        predictions = evaluate_model(
            model,
            selected["calibration"],
            settings["candidate_batch_size"],
            description="calibration",
            batch_size=settings["batch_size"],
        )
        temperature = calibrate(predictions)
        atomic_json(
            directory / "temperature.json",
            {
                "temperature": temperature,
                "split": "calibration",
                "n": len(predictions),
                "checkpoint_sha256": file_hash(directory / "checkpoints/best.pt"),
                "selected_ids_sha256": identity["selected_ids_sha256"]["calibration"],
            },
        )
        save_evaluation(
            directory / "calibration",
            {
                "step": cursor["step"],
                "split": "calibration",
                "temperature": temperature,
                "metrics": summarize(predictions, temperature),
                "predictions": predictions,
            },
            settings["eval_keep"],
            prefix="step",
        )
        atomic_json(
            directory / "status.json",
            {
                "status": "complete",
                "step": cursor["step"],
                "best": best,
                "temperature": temperature,
                "test_evaluated": False,
            },
        )
        logger.info(
            "训练及 calibration 完成；best step=%s temperature=%.6f", best["step"], temperature
        )
        return directory
    except BaseException as error:
        # 失败说明保留，坏参数不覆盖有效 latest。硬杀进程依靠上一个周期快照恢复。
        atomic_json(
            directory / "failure.json",
            {
                "step": cursor["step"],
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        logger.exception("训练失败；未发布无效 checkpoint")
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock_handle.close()
        for handler in logger.handlers:
            handler.close()
