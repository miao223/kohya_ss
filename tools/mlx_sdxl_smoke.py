"""Synthetic SDXL parity smoke test; this does not validate real-model training.

Run prepare and compare-export with the Torch/Diffusers Python environment, and
check with the isolated MLX Python environment. No model downloads are performed.
Each stage writes new files and refuses to overwrite previous results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_npz(path, **arrays):
    with Path(path).open("xb") as handle:
        np.savez(handle, **arrays)


def new_outputs(root, *names):
    for name in names:
        path = root / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")


def compare(actual, expected, atol, rtol, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{label}: non-finite values")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    metrics = {
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "atol": atol,
        "rtol": rtol,
    }
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: {metrics}")
    return metrics


def load_reference(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "mlx-sdxl-cache"
        or manifest.get("synthetic") is not True
    ):
        raise ValueError("This harness accepts only its synthetic smoke fixture")
    with np.load(root / "reference.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def torch_forward(model, arrays):
    import torch

    with torch.no_grad():
        result = model(
            torch.from_numpy(arrays["latents"].transpose(0, 3, 1, 2)),
            torch.from_numpy(arrays["timesteps"]),
            encoder_hidden_states=torch.from_numpy(arrays["prompt_embeds"]),
            added_cond_kwargs={
                "text_embeds": torch.from_numpy(arrays["pooled_prompt_embeds"]),
                "time_ids": torch.from_numpy(arrays["time_ids"]),
            },
            return_dict=False,
        )[0]
    return result.permute(0, 2, 3, 1).contiguous().numpy()


def prepare(args):
    import torch
    import diffusers
    from diffusers import UNet2DConditionModel
    from mlx_sdxl_backend import tiny_sdxl_config

    started = time.perf_counter()
    root = args.fixture.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Use a new fixture directory: {root}")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = tiny_sdxl_config()
    model = UNet2DConditionModel(**config).float().eval()
    pooled_dim = (
        config["projection_class_embeddings_input_dim"]
        - 6 * config["addition_time_embed_dim"]
    )
    arrays = {
        "latents": rng.standard_normal((1, 8, 8, 4)).astype(np.float32),
        "prompt_embeds": rng.standard_normal(
            (1, 77, config["cross_attention_dim"])
        ).astype(np.float32),
        "pooled_prompt_embeds": rng.standard_normal((1, pooled_dim)).astype(np.float32),
        "time_ids": np.array([[64, 64, 0, 0, 64, 64]], dtype=np.float32),
        "timesteps": np.array([321], dtype=np.int64),
        "target": rng.standard_normal((1, 8, 8, 4)).astype(np.float32),
    }
    expected = torch_forward(model, arrays)
    if not np.isfinite(expected).all():
        raise AssertionError("Torch reference contains non-finite values")
    root.mkdir(parents=True, exist_ok=False)
    (root / "data").mkdir()
    model.save_pretrained(root / "unet", safe_serialization=True)
    write_npz(root / "reference.npz", **arrays, expected=expected)
    write_npz(
        root / "data/000000.npz",
        **{
            key: arrays[key]
            for key in ("latents", "prompt_embeds", "pooled_prompt_embeds", "time_ids")
        },
    )
    manifest = {
        "format": "mlx-sdxl-cache",
        "version": 1,
        "latent_layout": "NHWC",
        "unet": "unet",
        "resolution": 64,
        "scheduler": {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        "samples": [{"file": "data/000000.npz", "repeats": 1}],
        "synthetic": True,
        "seed": args.seed,
    }
    write_json(root / "manifest.json", manifest)
    metrics = {
        "stage": "prepare",
        "synthetic": True,
        "fixture": str(root),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "seconds": time.perf_counter() - started,
    }
    write_json(root / "prepare.json", metrics)
    return metrics


def check(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_sdxl_backend import (
        enable_gradient_checkpointing,
        export_kohya_lora,
        export_lora,
        inject_lora,
        load_unet,
    )

    started = time.perf_counter()
    root = args.fixture.expanduser().resolve()
    new_outputs(
        root,
        "native_after.npz",
        "adapter.safetensors",
        "adapter-kohya.safetensors",
        "check.json",
    )
    reference = load_reference(root)
    config = json.loads((root / "unet/config.json").read_text(encoding="utf-8"))
    mx.random.seed(args.seed)
    model = load_unet(
        config,
        root / "unet/diffusion_pytorch_model.safetensors",
        args.examples_path,
        dtype="float32",
    )
    arrays = {
        key: mx.array(value) for key, value in reference.items() if key != "expected"
    }

    def forward(net):
        return net(
            arrays["latents"],
            arrays["timesteps"],
            arrays["prompt_embeds"],
            text_time=(arrays["pooled_prompt_embeds"], arrays["time_ids"]),
        )

    def loss_fn(net):
        return mx.mean(mx.square(forward(net).astype(mx.float32) - arrays["target"]))

    baseline = np.array(forward(model))
    parity = compare(
        baseline, reference["expected"], args.atol, args.rtol, "Torch / MLX forward"
    )
    targets = inject_lora(model, rank=args.rank, alpha=args.alpha)
    compare(np.array(forward(model)), baseline, 0.0, 0.0, "Zero-initialized LoRA")
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if len(trainable) != 2 * len(targets) or not all(
        key.endswith((".lora_A", ".lora_B")) for key in trainable
    ):
        raise AssertionError("Unexpected trainable base parameters")
    before_adapters = {key: np.array(value) for key, value in trainable.items()}
    before_base = {
        key: np.array(value)
        for key, value in tree_flatten(model.parameters())
        if not key.endswith((".lora_A", ".lora_B"))
    }
    value_and_grad = nn.value_and_grad(model, loss_fn)
    loss_plain, grads_plain = value_and_grad(model)
    mx.eval(loss_plain, grads_plain)
    grads_plain = {key: np.array(value) for key, value in tree_flatten(grads_plain)}
    initial_loss = float(loss_plain.item())
    checkpoint_count = enable_gradient_checkpointing(model)
    checkpoint_forward = compare(
        np.array(forward(model)), baseline, args.atol, args.rtol, "Checkpoint forward"
    )
    loss_checkpoint, grads_checkpoint = value_and_grad(model)
    mx.eval(loss_checkpoint, grads_checkpoint)
    compare(
        np.array(loss_checkpoint),
        np.array(loss_plain),
        args.atol,
        args.rtol,
        "Checkpoint loss",
    )
    checkpoint_grads = dict(tree_flatten(grads_checkpoint))
    if checkpoint_grads.keys() != grads_plain.keys():
        raise AssertionError("Checkpoint changed the gradient parameter set")
    grad_error = max(
        compare(
            np.array(value),
            grads_plain[key],
            args.atol,
            args.rtol,
            f"Checkpoint gradient {key}",
        )["max_absolute_error"]
        for key, value in checkpoint_grads.items()
    )
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    losses = []
    for _ in range(args.steps):
        loss, grads = value_and_grad(model)
        mx.eval(loss, grads)
        if not np.isfinite(loss.item()) or not all(
            np.isfinite(np.array(value)).all() for _, value in tree_flatten(grads)
        ):
            raise AssertionError("Non-finite training loss or gradient")
        losses.append(float(loss.item()))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    after = dict(tree_flatten(model.parameters()))
    for key, value in before_base.items():
        if not np.array_equal(np.array(after[key]), value):
            raise AssertionError(f"Frozen base parameter changed: {key}")
    changed_keys = [
        key
        for key, value in before_adapters.items()
        if not np.array_equal(np.array(after[key]), value)
    ]
    if not changed_keys:
        raise AssertionError("No adapter parameters changed")
    if args.steps >= 2 and not all(
        any(key.endswith(suffix) for key in changed_keys)
        for suffix in (".lora_A", ".lora_B")
    ):
        raise AssertionError("Both LoRA factors must learn after multiple steps")
    updated = np.array(forward(model))
    if not np.isfinite(updated).all() or np.array_equal(updated, baseline):
        raise AssertionError("Adapter failed to change the finite model output")
    export = export_lora(model, root / "adapter.safetensors", args.alpha, args.rank)
    kohya_export = export_kohya_lora(
        model, root / "adapter-kohya.safetensors", args.alpha, args.rank
    )
    write_npz(root / "native_after.npz", expected=updated)
    metrics = {
        "stage": "check",
        "synthetic": True,
        "forward_parity": parity,
        "checkpoint_forward": checkpoint_forward,
        "checkpoint_gradient_max_absolute_error": grad_error,
        "checkpoint_blocks": checkpoint_count,
        "adapter_targets": len(targets),
        "adapter_tensors_changed": len(changed_keys),
        "frozen_base_tensors_checked": len(before_base),
        "initial_loss": initial_loss,
        "training_losses": losses,
        "final_loss": float(loss_fn(model).item()),
        "peak_memory_bytes": mx.get_peak_memory(),
        "seconds": time.perf_counter() - started,
        "export": export,
        "kohya_export": kohya_export,
    }
    write_json(root / "check.json", metrics)
    return metrics


def compare_export(args):
    import torch
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file

    started = time.perf_counter()
    root = args.fixture.expanduser().resolve()
    new_outputs(root, "compare-export.json")
    reference = load_reference(root)
    torch.set_num_threads(2)
    model = (
        UNet2DConditionModel.from_pretrained(root / "unet", local_files_only=True)
        .float()
        .eval()
    )
    parameters = dict(model.named_parameters())
    base_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    adapter = load_file(root / "adapter.safetensors", device="cpu")
    consumed = set()
    with torch.no_grad():
        for key, down in adapter.items():
            if not key.startswith("unet.") or not key.endswith(".lora_A.weight"):
                continue
            prefix = key[: -len(".lora_A.weight")]
            up_key = prefix + ".lora_B.weight"
            weight_key = prefix[len("unet.") :] + ".weight"
            if up_key not in adapter or weight_key not in parameters:
                raise AssertionError(f"Unmapped exported adapter: {key}")
            delta = adapter[up_key] @ down
            if delta.shape != parameters[weight_key].shape:
                raise AssertionError(f"Export shape mismatch: {weight_key}")
            # export_lora bakes alpha/rank into B; do not apply it twice.
            parameters[weight_key].add_(delta)
            consumed.update((key, up_key))
    if not consumed or consumed != set(adapter):
        raise AssertionError("Export contains missing or unconsumed adapter tensors")
    with np.load(root / "native_after.npz", allow_pickle=False) as data:
        expected = data["expected"]
    parity = compare(
        torch_forward(model, reference),
        expected,
        args.atol,
        args.rtol,
        "Export round trip",
    )
    model.load_state_dict(base_state)
    kohya = load_file(root / "adapter-kohya.safetensors", device="cpu")
    kohya_map = {
        "lora_unet_" + key[: -len(".weight")].replace(".", "_"): key
        for key in parameters
        if key.endswith(".weight")
    }
    kohya_consumed = set()
    with torch.no_grad():
        for key, down in kohya.items():
            if not key.endswith(".lora_down.weight"):
                continue
            prefix = key[: -len(".lora_down.weight")]
            up_key, alpha_key = prefix + ".lora_up.weight", prefix + ".alpha"
            if prefix not in kohya_map or up_key not in kohya or alpha_key not in kohya:
                raise AssertionError(f"Unmapped Kohya adapter: {key}")
            delta = (kohya[up_key] @ down) * (kohya[alpha_key].item() / down.shape[0])
            parameters[kohya_map[prefix]].add_(delta)
            kohya_consumed.update((key, up_key, alpha_key))
    if not kohya_consumed or kohya_consumed != set(kohya):
        raise AssertionError("Kohya export contains missing or unconsumed tensors")
    kohya_parity = compare(
        torch_forward(model, reference),
        expected,
        args.atol,
        args.rtol,
        "Kohya export round trip",
    )
    metrics = {
        "stage": "compare-export",
        "synthetic": True,
        "exported_tensor_count": len(consumed),
        "parity": parity,
        "kohya_exported_tensor_count": len(kohya_consumed),
        "kohya_parity": kohya_parity,
        "seconds": time.perf_counter() - started,
    }
    write_json(root / "compare-export.json", metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("prepare", prepare),
        ("check", check),
        ("compare-export", compare_export),
    ):
        command = commands.add_parser(name)
        command.add_argument("fixture", type=Path, help="Synthetic fixture directory")
        command.set_defaults(function=function)
        if name != "compare-export":
            command.add_argument("--seed", type=int, default=42)
        if name != "prepare":
            command.add_argument("--atol", type=float, default=2e-4)
            command.add_argument("--rtol", type=float, default=2e-4)
        if name == "check":
            command.add_argument("--examples-path", type=Path, required=True)
            command.add_argument("--rank", type=int, default=2)
            command.add_argument("--alpha", type=float, default=1.0)
            command.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    if getattr(args, "steps", 1) < 1:
        parser.error("--steps must be positive")
    for name in ("atol", "rtol"):
        value = getattr(args, name, 0.0)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and non-negative")
    print(json.dumps(args.function(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
