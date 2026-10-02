# SDXL LoRA on Apple Silicon with MLX (experimental)

This is an independent command-line backend. It does not change the kohya GUI,
the PyTorch/MPS environment, or the `sd-scripts` submodule. The GUI's existing
Start training button still uses `sd-scripts`.

The implementation reuses Apple's
[MLX Stable Diffusion UNet](https://github.com/ml-explore/mlx-examples/tree/796f5b53cab69a3d48a44233ce21aae889e94a08/stable_diffusion),
with SDXL base configuration and Diffusers-compatible normalization. Apple's
example itself is an inference example, not a ready-made SDXL LoRA trainer.
The new training loop and LoRA adapters are experimental.

## What runs where

1. `mlx_sdxl_prepare.py` uses the existing PyTorch/Diffusers environment to load
   a **local SDXL base checkpoint**, save an independent UNet, and cache the VAE
   latents, both CLIP hidden states, pooled text features, and SDXL size/crop IDs.
   Preparation defaults to CPU; `--device mps` is optional.
2. `mlx_sdxl_train.py` runs UNet forward/backward and AdamW entirely in MLX on
   Metal. VAE and text encoders are no longer in the training process.
3. Only attention Q/K/V/output LoRA matrices train. Frozen model weights stay
   unchanged. Features are cached once, and gradients can accumulate or use
   transformer activation recomputation to reduce memory.

The first acceptance test uses a tiny randomly initialized SDXL-shaped UNet.
Passing it proves the implemented numerical path, **not successful training or
image quality with the real SDXL base checkpoint**. A real checkpoint run and
generation check are required before relying on this backend. No speedup over
MPS is claimed; benchmarking must use matching precision, resolution, batch,
adapter targets, and exclude preparation/first-step compilation costs.

## Isolated setup

Run commands from the repository root. Python 3.11 and an Apple Silicon Mac are
required. The existing GUI environment should already be installed.

```bash
bash tools/setup_mlx_sdxl.sh
```

If `uv` is not on PATH, pass its location using `UV_BIN`. On the current local
machine it is available as `$HOME/.hermes/bin/uv`:

```bash
UV_BIN="$HOME/.hermes/bin/uv" bash tools/setup_mlx_sdxl.sh
```

This installs MLX 0.31.2 into `.cache/mlx/venv` and pins Apple's example checkout.
It does not install MLX into `.venv`, change system Python, or download weights.
A sandbox can hide Metal devices; run the GPU commands from a normal local
terminal if Metal is unavailable inside an agent's sandbox.

## Verify the numerical path without downloading a model

Each fixture/output path must be new. Reruns can use a different directory.

```bash
.venv/bin/python tools/mlx_sdxl_smoke.py prepare .cache/mlx/check-001
.cache/mlx/venv/bin/python tools/mlx_sdxl_smoke.py check .cache/mlx/check-001 \
  --examples-path .cache/mlx/mlx-examples
.venv/bin/python tools/mlx_sdxl_smoke.py compare-export .cache/mlx/check-001

.cache/mlx/venv/bin/python tools/mlx_sdxl_train.py \
  --cache-dir .cache/mlx/check-001 \
  --output-dir outputs/mlx-synthetic-001 \
  --steps 20 --rank 4 --alpha 4 --checkpoint
```

The harness compares PyTorch and MLX forward outputs, checks zero-initialized
LoRA equivalence, finite gradients, frozen base weights, actual adapter updates,
checkpoint forward/gradient equivalence, and both exported adapter formats by
merging them back into the PyTorch UNet. Reports explicitly mark synthetic data.

## Prepare SDXL base locally

Use an ordinary SDXL base 1.0 `.safetensors` checkpoint. The other local session
is downloading `models/sdxl/sd_xl_base_1.0.safetensors`; wait for that session's
download and checksum verification before preparing it. Partial download files
are not usable. No second download of model weights is needed.

The loader also needs the small official Diffusers config/tokenizer files.
They are already prepared on this machine under `.cache/mlx/sdxl-config`.
To reproduce that download on another machine:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    revision="462165984030d82259a11f4367a4eed129e94a7b",
    allow_patterns=[
        "model_index.json", "scheduler/scheduler_config.json", "unet/config.json",
        "vae/config.json", "text_encoder/config.json", "text_encoder_2/config.json",
        "tokenizer/*", "tokenizer_2/*",
    ],
    local_dir=".cache/mlx/sdxl-config",
    max_workers=2,
)
PY
```

An image needs an adjacent UTF-8 `.txt` caption, or a parent folder named
`N_caption` (Kohya style). The nearest such folder provides the repeat count and
fallback caption. Images are resized and center-cropped; original dimensions
and the actual crop are retained for conditioning. Captions beyond 77 tokens
are truncated with a warning. Dataset contents are not modified.

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python tools/mlx_sdxl_prepare.py \
  --checkpoint models/sdxl/sd_xl_base_1.0.safetensors \
  --config-dir .cache/mlx/sdxl-config \
  --dataset 'test/img/10_darius kawasaki person' \
  --output-dir .cache/mlx/sdxl-data-001 \
  --resolution 1024 --device cpu --seed 42
```

Preparation is offline and refuses existing outputs. It needs approximately
10 GB for the FP32 UNet plus feature caches, in addition to the original
checkpoint. The bundled test pictures are only 512×512: upscaling them to 1024
validates execution, not training quality. Use suitable original images for a
real subject/style LoRA. Rebuild the cache in a new directory when changing the
checkpoint, images, captions, crops, resolution, or VAE sampling seed.

## Train and inspect results

```bash
.cache/mlx/venv/bin/python tools/mlx_sdxl_train.py \
  --cache-dir .cache/mlx/sdxl-data-001 \
  --output-dir outputs/mlx-sdxl-001 \
  --steps 20 --rank 4 --alpha 4 --learning-rate 0.0001 \
  --dtype float32 --checkpoint --seed 42
```

Start with FP32 as the correctness baseline. The optional `--dtype bfloat16`
reduces frozen weight/activation storage while adapter weights and optimizer
state remain FP32. It must be evaluated separately for numerical stability on
the full model. `--gradient-accumulation N` gives effective batch N with a
physical batch of one. `--save-every N` saves periodic Diffusers adapters.

Outputs:

| File | Meaning |
| --- | --- |
| `metrics.jsonl` | Step loss, gradient norm, seconds per optimizer step, MLX peak memory |
| `run.json` | Parameters, synthetic flag, completion/failure state, completed steps, timing |
| `adapter.safetensors` | Diffusers/PEFT `unet.*.lora_A/B.weight`; alpha/rank is baked into B |
| `adapter-kohya.safetensors` | Kohya `lora_unet_*` down/up weights and explicit per-module alpha |

For the Diffusers export, load with the default alpha=rank and adapter strength
1.0; applying the training alpha/rank again changes its strength incorrectly.
The Kohya export retains the original alpha and applies it once when loaded.
Both formats are numerically tested by reloading and merging their tensors;
actual GUI/ComfyUI image-generation compatibility still needs a real-model test.
Outputs never overwrite an existing run or adapter. Do not benchmark MLX and
MPS training simultaneously on the same GPU.

## Current limits

- SDXL base architecture only; no SDXL refiner, inpainting, SD 1.5, or FLUX.
- Attention LoRA only; no text-encoder, convolution, or feed-forward LoRA.
- Fixed square center crop, fixed cached latent sample and caption; no buckets,
  random augmentation, caption dropout/shuffle, or extended CLIP tokens.
- Constant learning rate and ordinary AdamW; no quantized training, resume of
  optimizer state, prior preservation, or validation image generation.
- No GUI integration yet. Existing kohya presets are not accepted by this CLI.
- The original model license still applies to converted weights and derivatives.

The CPU regression suite remains runnable without MLX or a GPU:

```bash
.venv/bin/python -m pytest tests/ test/test_allowed_paths.py
```
