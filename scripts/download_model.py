"""只下载 pinned 模型，不执行模型前向；运行时统一使用 local_files_only。"""

import argparse

from huggingface_hub import snapshot_download

from my_jev.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen35_2b.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--parallel-curl", action="store_true", help="可选的公开模型分段下载")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="分段下载字节来源；始终核验固定 HF revision 的哈希",
    )
    args = parser.parse_args()
    model = load_config(args.config, args.set)["model"]
    if args.parallel_curl:
        from parallel_download import download

        path = download(model, args.connections, args.source)
    else:
        path = snapshot_download(
            model["id"],
            revision=model["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE*"],
            max_workers=4,
        )
    print(f"模型已缓存: {path}")


if __name__ == "__main__":
    main()
