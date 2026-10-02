"""Experimental MLX SDXL attention LoRA training from an offline feature cache.

The separate prepare command runs frozen encoders once. This process uses MLX
only, and never imports PyTorch, downloads weights, or modifies the base model.
"""

import argparse
import json
import math
from pathlib import Path
import time


def load_manifest(cache_dir):
    root = Path(cache_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "mlx-sdxl-cache" or manifest.get("version") != 1:
        raise ValueError("Expected a version 1 mlx-sdxl-cache manifest")
    if manifest.get("latent_layout") != "NHWC" or not manifest.get("samples"):
        raise ValueError("Cache must contain NHWC latent samples")
    # Resolve all paths before creating output or loading the heavyweight model.
    for relative in [manifest["unet"]] + [s["file"] for s in manifest["samples"]]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError(f"Missing or out-of-cache path: {relative}")
    for sample in manifest["samples"]:
        repeats = sample.get("repeats", 1)
        if type(repeats) is not int or not 1 <= repeats <= 10000:
            raise ValueError("Dataset repeats must be integers in [1, 10000]")
    return root, manifest


def scheduler_alphas(config):
    import numpy as np

    count = int(config.get("num_train_timesteps", 1000))
    prediction = config.get("prediction_type", "epsilon")
    if prediction not in {"epsilon", "v_prediction"}:
        raise ValueError(f"Unsupported prediction_type: {prediction}")
    if config.get("rescale_betas_zero_snr", False):
        raise ValueError("Zero-terminal-SNR schedules are not supported")
    if config.get("trained_betas") is not None:
        betas = np.asarray(config["trained_betas"], dtype=np.float32)
        if betas.shape != (count,):
            raise ValueError("trained_betas length differs from num_train_timesteps")
    else:
        start, end = config.get("beta_start", 0.00085), config.get("beta_end", 0.012)
        schedule = config.get("beta_schedule", "scaled_linear")
        if schedule == "scaled_linear":
            betas = (
                np.linspace(math.sqrt(start), math.sqrt(end), count, dtype=np.float32)
                ** 2
            )
        elif schedule == "linear":
            betas = np.linspace(start, end, count, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported beta_schedule: {schedule}")
    if count < 2 or not np.all(np.isfinite(betas) & (betas > 0) & (betas < 1)):
        raise ValueError("Invalid noise schedule")
    return np.cumprod(1 - betas), prediction


def validate_sample(arrays, config):
    import numpy as np

    cross_dim = config["cross_attention_dim"]
    if isinstance(cross_dim, (list, tuple)):
        if not cross_dim or len(set(cross_dim)) != 1:
            raise ValueError(
                "Cached prompt features require equal cross-attention dimensions"
            )
        cross_dim = cross_dim[0]
    shapes = {
        "latents": (1, None, None, 4),
        "prompt_embeds": (1, None, cross_dim),
        "pooled_prompt_embeds": (
            1,
            config["projection_class_embeddings_input_dim"]
            - 6 * config["addition_time_embed_dim"],
        ),
        "time_ids": (1, 6),
    }
    for name, expected in shapes.items():
        if name not in arrays:
            raise ValueError(f"Cache sample missing {name}")
        value = arrays[name]
        if len(value.shape) != len(expected) or any(
            want is not None and got != want for got, want in zip(value.shape, expected)
        ):
            raise ValueError(
                f"Invalid {name} shape: {value.shape}; expected {expected}"
            )
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Non-finite or non-floating cached {name}")
    height, width = arrays["latents"].shape[1:3]
    factor = 2 ** (len(config["block_out_channels"]) - 1)
    if min(height, width) < factor or height % factor or width % factor:
        raise ValueError("Latent dimensions must match the U-Net downsampling factor")
    if not 1 <= arrays["prompt_embeds"].shape[1] <= 77:
        raise ValueError("Only captions up to 77 tokens are supported")


def train(args):
    import numpy as np
    from mlx_sdxl_backend import config_to_unet_kwargs

    root, manifest = load_manifest(args.cache_dir)
    model_dir = root / manifest["unet"]
    config = json.loads((model_dir / "config.json").read_text())
    config_to_unet_kwargs(config)
    alphas, prediction = scheduler_alphas(manifest["scheduler"])
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    # Preflight every sample before allocating the model. Cached features are small.
    for entry in manifest["samples"]:
        with np.load(root / entry["file"], allow_pickle=False) as arrays:
            validate_sample(arrays, config)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_sdxl_backend import (
        enable_gradient_checkpointing,
        export_kohya_lora,
        export_lora,
        inject_lora,
        load_unet,
    )

    if not mx.metal.is_available():
        raise RuntimeError(
            "MLX Metal GPU unavailable; run outside a GPU-restricted sandbox"
        )
    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_unet(
        config,
        model_dir / "diffusion_pytorch_model.safetensors",
        args.mlx_examples,
        dtype=args.dtype,
    )
    adapters = inject_lora(model, args.rank, args.alpha)
    checkpointed = enable_gradient_checkpointing(model) if args.checkpoint else 0
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    mx.eval(model.parameters())
    model.train()
    schedule = mx.array(alphas)
    dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16

    def loss_fn(net, clean, prompt, pooled, time_ids, noise, timesteps):
        a = schedule[timesteps].reshape(-1, 1, 1, 1)
        noisy = (mx.sqrt(a) * clean + mx.sqrt(1 - a) * noise).astype(dtype)
        predicted = net(
            noisy,
            timesteps,
            prompt.astype(dtype),
            text_time=(pooled.astype(dtype), time_ids),
        ).astype(mx.float32)
        target = (
            noise
            if prediction == "epsilon"
            else mx.sqrt(a) * noise - mx.sqrt(1 - a) * clean
        )
        return mx.mean(mx.square(predicted - target))

    value_and_grad = nn.value_and_grad(model, loss_fn)
    indices = np.repeat(
        np.arange(len(manifest["samples"])),
        [s.get("repeats", 1) for s in manifest["samples"]],
    )
    order = iter(())

    def next_sample():
        nonlocal order
        try:
            index = next(order)
        except StopIteration:
            order = iter(rng.permutation(indices))
            index = next(order)
        with np.load(
            root / manifest["samples"][index]["file"], allow_pickle=False
        ) as data:
            return [
                mx.array(data[k])
                for k in (
                    "latents",
                    "prompt_embeds",
                    "pooled_prompt_embeds",
                    "time_ids",
                )
            ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "status": "running",
        "backend": "mlx",
        "experimental": True,
        "cache_dir": str(root),
        "synthetic": manifest.get("synthetic", False),
        "steps": args.steps,
        "completed_steps": 0,
        "rank": args.rank,
        "alpha": args.alpha,
        "dtype": args.dtype,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "adapter_modules": len(adapters),
        "checkpointed_blocks": checkpointed,
        "trainable_parameters": sum(
            v.size for _, v in tree_flatten(model.trainable_parameters())
        ),
    }
    report_path = args.output_dir / "run.json"
    report_path.write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run), flush=True)
    elapsed_steps = []
    try:
        with (args.output_dir / "metrics.jsonl").open("x") as log:
            for step in range(1, args.steps + 1):
                start = time.perf_counter()
                accumulated = None
                loss_total = 0.0
                for _ in range(args.gradient_accumulation):
                    clean, prompt, pooled, time_ids = next_sample()
                    noise = mx.random.normal(clean.shape)
                    timesteps = mx.random.randint(0, len(alphas), shape=(1,))
                    loss, grads = value_and_grad(
                        model, clean, prompt, pooled, time_ids, noise, timesteps
                    )
                    mx.eval(loss, grads)
                    if not math.isfinite(loss.item()):
                        raise FloatingPointError(
                            "Non-finite loss; try float32 and inspect cached features"
                        )
                    loss_total += loss.item() / args.gradient_accumulation
                    accumulated = (
                        grads
                        if accumulated is None
                        else tree_map(lambda a, b: a + b, accumulated, grads)
                    )
                    mx.eval(accumulated)
                grads = tree_map(lambda g: g / args.gradient_accumulation, accumulated)
                grads, norm = optim.clip_grad_norm(grads, args.max_grad_norm)
                mx.eval(norm)
                if not math.isfinite(norm.item()):
                    raise FloatingPointError(
                        "Non-finite gradient; weights were not updated"
                    )
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                seconds = time.perf_counter() - start
                elapsed_steps.append(seconds)
                run["completed_steps"] = step
                metrics = {
                    "step": step,
                    "loss": loss_total,
                    "grad_norm": norm.item(),
                    "seconds": seconds,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                print(json.dumps(metrics), flush=True)
                if (
                    args.save_every
                    and step % args.save_every == 0
                    and step != args.steps
                ):
                    export_lora(
                        model,
                        args.output_dir / f"step-{step:06d}.safetensors",
                        args.alpha,
                        args.rank,
                    )
            export_lora(
                model, args.output_dir / "adapter.safetensors", args.alpha, args.rank
            )
            export_kohya_lora(
                model,
                args.output_dir / "adapter-kohya.safetensors",
                args.alpha,
                args.rank,
            )
        run.update(
            status="complete",
            completed_steps=args.steps,
            peak_memory_gb=mx.get_peak_memory() / 1e9,
            mean_step_seconds=float(np.mean(elapsed_steps)),
            warm_mean_step_seconds=(
                float(np.mean(elapsed_steps[1:])) if len(elapsed_steps) > 1 else None
            ),
        )
    except BaseException as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_path.write_text(json.dumps(run, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mlx-examples", type=Path, default=Path(".cache/mlx/mlx-examples")
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Recompute transformer activations during backward",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()
    for name in ("steps", "rank", "gradient_accumulation"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("alpha", "learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.save_every < 0:
        parser.error("--save-every must be nonnegative")
    train(args)


if __name__ == "__main__":
    main()
