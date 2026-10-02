"""Prepare a local SDXL checkpoint and static dataset for native MLX training.

The output is a new, immutable cache: Diffusers UNet weights plus one NPZ per
image. Captions and VAE samples are fixed, so random crops/flips, caption
shuffling/dropout and text-encoder training are intentionally unsupported.
This command never downloads model/config/tokenizer files.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import warnings

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CONFIG_FILES = (
    "model_index.json",
    "unet/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
)


@dataclass(frozen=True)
class Sample:
    path: Path
    relative_path: str
    caption: str
    repeats: int


def discover_samples(dataset):
    """Read adjacent UTF-8 captions; inherit the nearest N_class repeat folder."""
    from PIL import Image

    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    samples = []
    for path in sorted(dataset.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not path.resolve().is_relative_to(dataset):
            raise ValueError(f"Image symlink escapes the dataset: {path}")
        repeats, fallback = 1, None
        for folder in path.parents:
            if not folder.is_relative_to(dataset):
                break
            match = re.fullmatch(r"(\d+)_(.+)", folder.name)
            if match:
                repeats, fallback = int(match[1]), match[2]
                if repeats < 1:
                    raise ValueError(f"Repeat count must be positive: {folder}")
                break
        caption_path = path.with_suffix(".txt")
        if caption_path.is_file():
            caption = caption_path.read_text(encoding="utf-8-sig").strip()
        else:
            caption = fallback
        if not caption:
            raise ValueError(f"Missing/empty caption: {caption_path}")
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Animated/multipage images are unsupported: {path}")
            image.verify()
        with Image.open(path) as image:
            image.load()
        samples.append(
            Sample(path, path.relative_to(dataset).as_posix(), caption, repeats)
        )
    if not samples:
        raise ValueError(f"No supported images in {dataset}")
    return samples


def validate_inputs(checkpoint, config_dir, dataset, output_dir, resolution, seed):
    """Fail on configuration and image errors before importing/loading models."""
    if Path(output_dir).expanduser().is_symlink():
        raise ValueError("Output must not be an existing symlink")
    checkpoint, config_dir, dataset, output_dir = (
        Path(path).expanduser().resolve()
        for path in (checkpoint, config_dir, dataset, output_dir)
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(
            f"Output already exists; choose a fresh cache directory: {output_dir}"
        )
    if resolution < 64 or resolution % 64:
        raise ValueError("--resolution must be a positive multiple of 64 (at least 64)")
    if not 0 <= seed < 2**63:
        raise ValueError("--seed must be between 0 and 2**63 - 1")
    if checkpoint.suffix.lower() != ".safetensors" or not checkpoint.is_file():
        raise ValueError("--checkpoint must be a local .safetensors checkpoint file")
    if checkpoint.stat().st_size == 0:
        raise ValueError(f"Checkpoint is empty: {checkpoint}")
    for source in (dataset, config_dir):
        if output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise ValueError(
                "Output must be separate from the dataset and config directories"
            )
    missing = [name for name in CONFIG_FILES if not (config_dir / name).is_file()]
    if missing:
        raise ValueError(
            "Missing local SDXL configuration/tokenizer files: " + ", ".join(missing)
        )
    config = json.loads((config_dir / "unet/config.json").read_text())
    if (
        config.get("in_channels") != 4
        or config.get("cross_attention_dim") != 2048
        or config.get("addition_embed_type") != "text_time"
    ):
        raise ValueError(
            "Only SDXL base UNet configuration is supported (not refiner/inpainting)"
        )
    samples = discover_samples(dataset)
    return checkpoint, config_dir, dataset, output_dir, samples


def prepare_image(path, resolution):
    """Return normalized CHW pixels and exact SDXL size/crop conditioning."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = resolution / min(width, height)
        resized = (
            max(resolution, round(width * scale)),
            max(resolution, round(height * scale)),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)
        left = (resized[0] - resolution) // 2
        top = (resized[1] - resolution) // 2
        image = image.crop((left, top, left + resolution, top + resolution))
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return pixels.transpose(2, 0, 1).copy(), [height, width], [top, left]


@contextmanager
def offline_environment():
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_pipeline(checkpoint, config_dir):
    import torch

    with offline_environment():
        from diffusers import StableDiffusionXLPipeline

        return StableDiffusionXLPipeline.from_single_file(
            str(checkpoint),
            config=str(config_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
            add_watermarker=False,
        )


def check_tokenizers(pipe, samples):
    for index, tokenizer in enumerate((pipe.tokenizer, pipe.tokenizer_2), start=1):
        if tokenizer is None or tokenizer.model_max_length != 77:
            raise ValueError(f"SDXL tokenizer {index} must have model_max_length=77")
        for sample in samples:
            token_ids = tokenizer(sample.caption, truncation=False).input_ids
            if len(token_ids) > 77:
                warnings.warn(
                    f"{sample.relative_path}: tokenizer {index} has {len(token_ids)} "
                    "tokens; caption will be truncated to CLIP's 77-token limit.",
                    stacklevel=2,
                )


def numpy_tensor(tensor):
    """Preserve single-sample batch dimensions in a portable float32 cache."""
    import numpy as np

    result = tensor.detach().float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError(
            "Encoder produced non-finite values; no usable cache was written"
        )
    return result


def prepare(
    checkpoint, config_dir, dataset, output_dir, resolution=1024, device="cpu", seed=0
):
    checkpoint, config_dir, dataset, output_dir, samples = validate_inputs(
        checkpoint, config_dir, dataset, output_dir, resolution, seed
    )
    if device not in ("cpu", "mps"):
        raise ValueError("Device must be cpu or mps")
    import numpy as np
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu for cache preparation")
    print(
        f"Loading local SDXL checkpoint on CPU; preparing {len(samples)} images",
        flush=True,
    )
    pipe = load_pipeline(checkpoint, config_dir)
    check_tokenizers(pipe, samples)
    scale = float(pipe.vae.config.scaling_factor)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("VAE scaling_factor must be finite and positive")
    if pipe.vae_scale_factor != 8:
        raise ValueError("SDXL cache requires a VAE downsampling factor of 8")
    scheduler = dict(pipe.scheduler.config)
    if scheduler.get("prediction_type", "epsilon") not in ("epsilon", "v_prediction"):
        raise ValueError("Only epsilon/v_prediction training targets are supported")
    if scheduler.get("beta_schedule", "scaled_linear") not in (
        "linear",
        "scaled_linear",
    ):
        raise ValueError("Only linear/scaled_linear noise schedules are supported")
    if scheduler.get("trained_betas") is not None:
        raise ValueError("Explicit trained_betas are unsupported")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        # Keep the full UNet off the GPU; only frozen encoders need computation.
        pipe.unet.cpu().save_pretrained(
            staging / "unet", safe_serialization=True, max_shard_size="20GB"
        )
        pipe.unet = None
        gc.collect()
        for encoder in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
            encoder.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
        pipe.vae.enable_slicing()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        (staging / "data").mkdir()
        entries = []
        with torch.inference_mode():
            for index, sample in enumerate(samples):
                pixels, original_size, crop = prepare_image(sample.path, resolution)
                image_tensor = torch.from_numpy(pixels).unsqueeze(0).to(device)
                distribution = pipe.vae.encode(image_tensor).latent_dist
                # Draw on CPU for a reproducible seed on both CPU and MPS.
                mean, std = distribution.mean.cpu(), distribution.std.cpu()
                noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
                latents = ((mean + std * noise) * scale).permute(0, 2, 3, 1)
                prompt, _, pooled, _ = pipe.encode_prompt(
                    prompt=sample.caption,
                    prompt_2=sample.caption,
                    device=torch.device(device),
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    clip_skip=None,
                )
                if tuple(prompt.shape) != (1, 77, 2048) or tuple(pooled.shape) != (
                    1,
                    1280,
                ):
                    raise ValueError(
                        "Text encoders did not produce SDXL base conditioning"
                    )
                if tuple(latents.shape) != (1, resolution // 8, resolution // 8, 4):
                    raise ValueError(
                        "VAE did not produce the expected SDXL NHWC latent shape"
                    )
                relative_file = f"data/{index:06d}.npz"
                np.savez(
                    staging / relative_file,
                    latents=numpy_tensor(latents),
                    prompt_embeds=numpy_tensor(prompt),
                    pooled_prompt_embeds=numpy_tensor(pooled),
                    time_ids=np.asarray(
                        [original_size + crop + [resolution, resolution]],
                        dtype=np.float32,
                    ),
                )
                entries.append(
                    dict(
                        file=relative_file,
                        image=sample.relative_path,
                        caption=sample.caption,
                        repeats=sample.repeats,
                        original_size=original_size,
                        crop_top_left=crop,
                        target_size=[resolution, resolution],
                    )
                )
                print(
                    f"Cached {index + 1}/{len(samples)}: {sample.relative_path}",
                    flush=True,
                )
        checkpoint_stat = checkpoint.stat()
        manifest = dict(
            format="mlx-sdxl-cache",
            version=1,
            resolution=resolution,
            latent_layout="NHWC",
            vae_scaling_factor=scale,
            seed=seed,
            unet="unet",
            scheduler=scheduler,
            source=dict(
                checkpoint=str(checkpoint),
                checkpoint_size=checkpoint_stat.st_size,
                checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns,
                config_dir=str(config_dir),
                dataset=str(dataset),
            ),
            samples=entries,
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        # Recheck before publishing: a previous run must never be overwritten.
        if output_dir.exists():
            raise ValueError(f"Output appeared during preparation: {output_dir}")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Prepared cache: {output_dir / 'manifest.json'}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        prepare(**vars(args))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
