"""JSON 请求 → 三种类型的概率/答案；推理只加载本地缓存的固定基础模型。"""

import argparse
import json
from pathlib import Path

from my_jev.inference import load_run, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--uncalibrated", action="store_true", help="显式使用 T=1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prefix-cache", action="store_true", help="实验性共享前缀缓存")
    arguments = parser.parse_args()
    model, config, temperature, _, _ = load_run(
        arguments.run,
        arguments.checkpoint,
        uncalibrated=arguments.uncalibrated,
        device=arguments.device,
    )
    result = predict(
        model,
        json.loads(arguments.request.read_text()),
        temperature=temperature,
        candidate_batch_size=config["train"]["candidate_batch_size"],
        prefix_cache=arguments.prefix_cache,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
