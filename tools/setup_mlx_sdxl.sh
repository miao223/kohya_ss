#!/usr/bin/env bash
# Install the experimental backend separately from the existing kohya GUI.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo 'MLX SDXL requires an Apple Silicon Mac.' >&2
    exit 1
fi
UV_BIN="${UV_BIN:-uv}"
command -v "$UV_BIN" >/dev/null || {
    echo 'Set UV_BIN to your uv executable, or install uv first.' >&2
    exit 1
}
MLX_PYTHON="${MLX_PYTHON:-.venv/bin/python}"
export UV_CACHE_DIR="$PWD/.cache/uv"
SOURCE="$PWD/.cache/mlx/mlx-examples"
REVISION=796f5b53cab69a3d48a44233ce21aae889e94a08
mkdir -p .cache/mlx
if [[ ! -d "$SOURCE" ]]; then
    git init "$SOURCE"
    git -C "$SOURCE" remote add origin https://github.com/ml-explore/mlx-examples.git
    git -C "$SOURCE" sparse-checkout init --cone
    git -C "$SOURCE" sparse-checkout set stable_diffusion
    git -C "$SOURCE" fetch --depth 1 --filter=blob:none origin "$REVISION"
    git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$SOURCE" rev-parse HEAD)" != "$REVISION" ]]; then
    echo 'MLX example checkout has a different revision; keep it and use a separate checkout.' >&2
    exit 1
fi
if [[ ! -x .cache/mlx/venv/bin/python ]]; then
    "$UV_BIN" venv --python "$MLX_PYTHON" .cache/mlx/venv
fi
"$UV_BIN" pip install --python .cache/mlx/venv/bin/python -r requirements_mlx.txt
.cache/mlx/venv/bin/python -c 'import mlx.core as mx; print("MLX Metal available:", mx.metal.is_available())'
echo 'MLX environment ready. See docs/LoRA/MLX.md for validation and offline preparation.'
