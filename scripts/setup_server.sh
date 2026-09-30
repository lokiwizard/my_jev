#!/usr/bin/env bash
# 只在 Linux GPU 服务器运行。首次生成 uv.lock；之后严格使用锁文件。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv python install 3.12
if [[ -f uv.lock ]]; then
  uv sync --locked --python 3.12
else
  uv sync --python 3.12
fi
