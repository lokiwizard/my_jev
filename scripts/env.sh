#!/usr/bin/env bash
# source 此文件后，uv 与 Hugging Face 缓存都放在数据盘，避免占满系统盘。
JEV_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$JEV_PROJECT_ROOT/../cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$JEV_PROJECT_ROOT/../cache/python}"
export HF_HOME="${HF_HOME:-$JEV_PROJECT_ROOT/../cache/huggingface}"
# AutoDL 代理下优先使用普通 HTTP 下载，避免 Xet 的独立连接长时间重试。
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
