"""训练入口；正式运行前请先完成 prepare_data、download_model 和 preflight。"""

import argparse
from pathlib import Path

from my_jev.config import load_config
from my_jev.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen35_2b_gpu32.yaml")
    parser.add_argument("--set", action="append", default=[], help="例如 train.epochs=2")
    parser.add_argument("--resume", type=Path, help="运行目录或 checkpoints/latest.pt")
    parser.add_argument(
        "--stop-after-steps", type=int, default=0, help="只用于受控中断验收，不修改正式总训练预算"
    )
    args = parser.parse_args()
    if args.stop_after_steps < 0:
        parser.error("stop-after-steps 必须非负")
    resume = args.resume
    if resume and resume.is_dir():
        resume = resume / "checkpoints/latest.pt"
    config = load_config(args.config, args.set)
    directory = train(config, resume=resume, stop_after_steps=args.stop_after_steps)
    print(f"运行目录: {directory.resolve()}")


if __name__ == "__main__":
    main()
