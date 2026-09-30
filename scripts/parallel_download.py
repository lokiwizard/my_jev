"""可选的 curl 分段下载：用于 AutoDL 代理下单连接下载长时间停滞的情况。

只处理公开模型；先取固定 revision 的文件元数据，再校验完整文件哈希。
输出仍采用 HF 标准缓存目录，所以训练配置与离线加载方式不变。
"""

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, constants, hf_hub_url
from tqdm.auto import tqdm


def content_hash(path, *, git_blob=False):
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_part(url, directory, index, start, end, cancelled):
    """精确核验 Content-Range，防止 CDN 忽略 Range 后拼出错误文件。"""
    path = directory / f"part_{index:05d}"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return path
    temporary, headers = path.with_suffix(".incomplete"), path.with_suffix(".headers")
    if cancelled.is_set():
        raise RuntimeError("已取消下载")
    process = subprocess.Popen(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            "300",
            "--retry",
            "2",
            "--retry-all-errors",
            "--retry-delay",
            "1",
            "--range",
            f"{start}-{end}",
            "--max-filesize",
            str(expected),
            "--dump-header",
            str(headers),
            "--output",
            str(temporary),
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        while True:
            try:
                process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancelled.is_set():
                    process.terminate()
                    process.communicate(timeout=5)
                    raise RuntimeError("已取消下载")
        if process.returncode:
            raise RuntimeError(f"curl 分段下载失败: part={index}, exit={process.returncode}")
        ranges = re.findall(
            r"content-range:\s*bytes\s+(\d+)-(\d+)/\d+", headers.read_text(), flags=re.IGNORECASE
        )
        if not ranges or tuple(map(int, ranges[-1])) != (start, end):
            raise ValueError(f"HTTP Content-Range 与请求不符: part={index}")
        if temporary.stat().st_size != expected:
            raise ValueError(f"分段字节数不符: part={index}")
        os.replace(temporary, path)
        return path
    finally:
        # 不保留包含临时 CDN 签名的响应头。
        headers.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def download(model: dict, connections: int, source: str = "huggingface"):
    if not 1 <= connections <= 32:
        raise ValueError("connections 必须在 [1,32] 内")
    info = HfApi().model_info(model["id"], revision=model["revision"], files_metadata=True)
    if info.sha != model["revision"]:
        raise ValueError("Hub 返回的模型 revision 不符")
    repository = Path(constants.HF_HUB_CACHE) / ("models--" + model["id"].replace("/", "--"))
    blobs, snapshot = repository / "blobs", repository / "snapshots" / info.sha
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE*"]
    for entry in info.siblings:
        if not any(fnmatch.fnmatch(entry.rfilename, pattern) for pattern in patterns):
            continue
        is_lfs = entry.lfs is not None
        checksum = entry.lfs.sha256 if is_lfs else entry.blob_id
        path = blobs / checksum
        size = entry.size
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"缺少公开文件大小: {entry.rfilename}")
        valid = (
            path.exists()
            and path.stat().st_size == size
            and content_hash(path, git_blob=not is_lfs) == checksum
        )
        if not valid:
            parts_directory = repository / ".download-parts" / checksum
            parts_directory.mkdir(parents=True, exist_ok=True)
            chunk_size = 32 * 1024 * 1024
            ranges = [
                (index, start, min(start + chunk_size, size) - 1)
                for index, start in enumerate(range(0, size, chunk_size))
            ]
            url = (
                f"https://www.modelscope.cn/models/{model['id']}/resolve/master/{entry.rfilename}"
                if source == "modelscope"
                else hf_hub_url(model["id"], entry.rfilename, revision=info.sha)
            )
            # 镜像只提供候选字节，接受条件始终是固定 HF revision 的完整文件哈希。
            cancelled = threading.Event()
            with ThreadPoolExecutor(max_workers=connections) as executor:
                futures = {
                    executor.submit(fetch_part, url, parts_directory, *item, cancelled): item
                    for item in ranges
                }
                with tqdm(total=size, desc=entry.rfilename, unit="B", unit_scale=True) as progress:
                    try:
                        for future in as_completed(futures):
                            future.result()
                            _, start, end = futures[future]
                            progress.update(end - start + 1)
                    except BaseException:
                        cancelled.set()
                        for future in futures:
                            future.cancel()
                        raise
            assembled = blobs / f".{checksum}.assembling"
            try:
                with assembled.open("wb") as handle:
                    for index, _, _ in ranges:
                        with (parts_directory / f"part_{index:05d}").open("rb") as source:
                            shutil.copyfileobj(source, handle, 8 * 1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
                if content_hash(assembled, git_blob=not is_lfs) != checksum:
                    raise ValueError(f"完整模型文件哈希不符: {entry.rfilename}")
                os.replace(assembled, path)
                shutil.rmtree(parts_directory)
            finally:
                assembled.unlink(missing_ok=True)
        link = snapshot / entry.rfilename
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name("." + link.name + ".link")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(os.path.relpath(path, link.parent))
        os.replace(temporary, link)
    return snapshot
