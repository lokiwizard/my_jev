"""独立 test/OOD 评估与冻结底座基线；不会用测试集重新选模或校准。"""

import argparse
import json
from pathlib import Path

from my_jev.artifacts import file_hash, save_evaluation
from my_jev.checkpoint import seed_everything
from my_jev.config import load_config
from my_jev.data import read_split, verify_manifest
from my_jev.evaluation import calibrate, evaluate_model, summarize
from my_jev.inference import load_run
from my_jev.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--base", action="store_true", help="未训练 Yes-No 评分头基线")
    parser.add_argument("--config", type=Path, default=Path("configs/qwen35_2b.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--splits", nargs="+", choices=["test", "ood"], default=["test", "ood"])
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--uncalibrated", action="store_true")
    arguments = parser.parse_args()
    if arguments.run:
        if arguments.set:
            parser.error("已训练实验使用保存的配置，不能通过 --set 更改测试条件")
        model, config, temperature, saved, path = load_run(
            arguments.run,
            arguments.checkpoint,
            uncalibrated=arguments.uncalibrated,
        )
        output = arguments.run / "evaluation"
        identity = {"checkpoint_sha256": file_hash(path), "checkpoint": arguments.checkpoint}
        step = saved.get("step", saved.get("cursor", {}).get("step", 0))
    else:
        config = load_config(arguments.config, arguments.set)
        verify_manifest(config["data"]["directory"], config["data"])
        seed_everything(config["experiment"]["seed"])
        config["model"]["lora_rank"] = 0
        model = build_model(config)
        calibration = read_split(
            config["data"]["directory"],
            "calibration",
            limit=config["data"]["calibration_limit"],
            seed=config["experiment"]["seed"],
        )
        predictions = evaluate_model(
            model,
            calibration,
            config["train"]["candidate_batch_size"],
            description="base calibration",
            batch_size=config["train"]["batch_size"],
        )
        temperature = 1.0 if arguments.uncalibrated else calibrate(predictions)
        output = Path(config["experiment"]["output_root"]) / "base_evaluation"
        identity, step = {"model": config["model"], "baseline": "untrained Yes-No readout"}, 0
    manifest = verify_manifest(config["data"]["directory"], config["data"])
    metrics, predictions_by_split = {}, {}
    for split in dict.fromkeys(arguments.splits):
        rows = read_split(
            config["data"]["directory"],
            split,
            limit=config["data"][f"{split}_limit"],
            seed=config["experiment"]["seed"],
        )
        predictions = evaluate_model(
            model,
            rows,
            config["train"]["candidate_batch_size"],
            description=split,
            batch_size=config["train"]["batch_size"],
        )
        metrics[split] = {
            "uncalibrated": summarize(predictions),
            "calibrated": summarize(predictions, temperature),
        }
        predictions_by_split[split] = predictions
    path = save_evaluation(
        output,
        {
            "step": step,
            "identity": identity,
            "config": config,
            "data_source": manifest["source"],
            "temperature": temperature,
            "metrics": metrics,
            "predictions": predictions_by_split,
        },
        config["train"]["eval_keep"],
        prefix="evaluation",
    )
    print(json.dumps({"result": str(path), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
