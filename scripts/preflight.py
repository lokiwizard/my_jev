"""全量核验数据及真实 tokenizer 的完整输入长度，不加载模型权重或启动训练。"""

import argparse
import json

from transformers import AutoTokenizer

from my_jev.artifacts import atomic_json, object_hash
from my_jev.config import load_config
from my_jev.data import SPLITS, check_input_lengths, read_split, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen35_2b.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", default="data/preflight.json")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    manifest = verify_manifest(config["data"]["directory"], config["data"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        local_files_only=True,
    )
    result = {"data_manifest_sha256": object_hash(manifest), "model": config["model"], "splits": {}}
    for split in SPLITS:
        rows = read_split(
            config["data"]["directory"],
            split,
            limit=config["data"][f"{split}_limit"],
            seed=config["experiment"]["seed"],
        )
        result["splits"][split] = check_input_lengths(
            rows,
            tokenizer,
            config["model"]["max_length"],
            description=f"preflight {split}",
        )
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
