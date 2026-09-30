#!/usr/bin/env bash
# 加速只作用于这个下载子进程；无论成功或失败都清除代理，不污染训练环境。
set -euo pipefail
trap 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY' EXIT
if [[ -f /etc/network_turbo ]]; then
  # AutoDL 脚本未保证 unset 变量兼容性，因此 source 时临时关闭 nounset。
  set +u
  source /etc/network_turbo
  set -u
fi
"$@"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
