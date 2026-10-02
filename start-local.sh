#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export GRADIO_ANALYTICS_ENABLED=False
export PYTORCH_ENABLE_MPS_FALLBACK=1

exec .venv/bin/python kohya_gui.py \
  --noverify \
  --listen 127.0.0.1 \
  --server_port 7860 \
  --do_not_share \
  --language zh-CN \
  "$@"
