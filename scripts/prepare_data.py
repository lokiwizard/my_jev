"""下载 pinned 原始 JSONL，校验公开清单的解压哈希，再执行全量划分审计。"""

import argparse
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from my_jev.artifacts import atomic_json, file_hash
from my_jev.config import load_config
from my_jev.data import SPLITS, audit_dataset, verify_manifest


def prepare(config: dict) -> dict:
    data = config["data"]
    directory = Path(data["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    # 已准备数据只做字节核验，绝不悄悄覆盖已被训练使用的数据。
    if (directory / "manifest.json").exists():
        manifest = verify_manifest(directory)
        expected = {key: data[key] for key in ("repo", "revision", "config")}
        if manifest["source"] != expected:
            raise ValueError("已存在数据的来源与配置不一致，请使用另一个 data.directory")
        return manifest
    upstream_path = hf_hub_download(
        data["repo"],
        "export-manifest.json",
        repo_type="dataset",
        revision=data["revision"],
    )
    upstream = json.loads(Path(upstream_path).read_text())
    if data["config"] not in upstream["configs"]:
        raise ValueError("当前下载器支持 export-manifest.json 中有原始字节哈希的 configs")
    entry = upstream["configs"][data["config"]]
    for split in tqdm(SPLITS, desc="download splits", unit="split"):
        record = entry["splits"][split]
        downloaded = hf_hub_download(
            data["repo"],
            record["raw_path"],
            repo_type="dataset",
            revision=data["revision"],
        )
        descriptor, temporary = tempfile.mkstemp(prefix=".decompress-", dir=directory)
        try:
            with gzip.open(downloaded, "rb") as source, os.fdopen(descriptor, "wb") as target:
                shutil.copyfileobj(source, target)
            if file_hash(temporary) != record["raw_uncompressed_sha256"]:
                raise ValueError(f"{split} 原始数据哈希不符")
            os.replace(temporary, directory / f"{split}.jsonl")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    audit = audit_dataset(directory)
    for split in SPLITS:
        if audit["counts"][split] != entry["splits"][split]["count"]:
            raise ValueError(f"{split} 实际数量与公开清单不符")
    manifest = {
        "source": {key: data[key] for key in ("repo", "revision", "config")},
        "upstream_manifest_sha256": file_hash(upstream_path),
        "audit": audit,
        "input_fields": ["state", "question", "kind", "options"],
        "label_field": "target",
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen35_2b.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config, args.set)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
