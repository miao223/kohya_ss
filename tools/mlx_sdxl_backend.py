"""Experimental SDXL UNet LoRA training backend using Apple's MLX examples.

The upstream code is imported from an explicit local checkout, never copied or
modified. Inputs and outputs are NHWC. Only the SDXL UNet is trained; text
embeddings and scaled VAE latents must be prepared separately.

Exports use Diffusers/PEFT ``unet.*.lora_A/B.weight`` names and PyTorch matrix
orientation. The training scale alpha/rank is baked into B, so the exported
adapter is loaded with the default PEFT alpha=rank (adapter weight 1.0).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import types

UPSTREAM_REVISION = "796f5b53cab69a3d48a44233ce21aae889e94a08"
_ATTENTION_PATH = re.compile(
    r"^(?:down_blocks\.\d+\.attentions\.\d+|"
    r"up_blocks\.\d+\.attentions\.\d+|mid_blocks\.1)"
    r"\.transformer_blocks\.\d+\.attn[12]\."
    r"(?:query_proj|key_proj|value_proj|out_proj)$"
)


def _per_block(value, count, name):
    values = [value] * count if isinstance(value, int) else value
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise ValueError(f"{name} must be an integer or a list of {count} integers")
    if any(type(item) is not int or item < 1 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return list(values)


def config_to_unet_kwargs(config):
    """Validate a Diffusers SDXL config and translate Apple's UNetConfig fields.

    Reject features Apple's example does not implement instead of silently
    training a different denoiser. Extra descriptive Diffusers metadata is OK.
    """
    required = (
        "block_out_channels",
        "down_block_types",
        "up_block_types",
        "cross_attention_dim",
        "attention_head_dim",
        "addition_time_embed_dim",
        "projection_class_embeddings_input_dim",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing SDXL UNet config fields: {', '.join(missing)}")
    defaults = {
        "addition_embed_type": "text_time",
        "act_fn": "silu",
        "attention_type": "default",
        "center_input_sample": False,
        "class_embed_type": None,
        "class_embeddings_concat": False,
        "cross_attention_norm": None,
        "downsample_padding": 1,
        "dropout": 0.0,
        "dual_cross_attention": False,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 4,
        "mid_block_only_cross_attention": None,
        "mid_block_scale_factor": 1.0,
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "num_attention_heads": None,
        "num_class_embeds": None,
        "out_channels": 4,
        "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": False,
        "resnet_time_scale_shift": "default",
        "reverse_transformer_layers_per_block": None,
        "time_cond_proj_dim": None,
        "time_embedding_act_fn": None,
        "time_embedding_dim": None,
        "time_embedding_type": "positional",
        "timestep_post_act": None,
        "use_linear_projection": True,
    }
    for key, expected in defaults.items():
        if config.get(key, expected) != expected:
            raise ValueError(
                f"Unsupported SDXL config {key}={config[key]!r}; expected {expected!r}"
            )
    if config.get("addition_embed_type") != "text_time":
        raise ValueError("SDXL requires addition_embed_type='text_time'")
    if config.get("upcast_attention") not in (None, False):
        raise ValueError("upcast_attention is not supported")
    only_cross = config.get("only_cross_attention", False)
    if isinstance(only_cross, (list, tuple)):
        only_cross = any(only_cross)
    if only_cross:
        raise ValueError("only_cross_attention is not supported")

    channels = config["block_out_channels"]
    if not isinstance(channels, (tuple, list)) or not channels:
        raise ValueError("block_out_channels must be a nonempty list")
    count = len(channels)
    channels = _per_block(channels, count, "block_out_channels")
    groups = config.get("norm_num_groups", 32)
    if type(groups) is not int or groups < 1 or any(c % groups for c in channels):
        raise ValueError("norm_num_groups must divide every block_out_channels value")
    for key, allowed in (
        ("down_block_types", {"DownBlock2D", "CrossAttnDownBlock2D"}),
        ("up_block_types", {"UpBlock2D", "CrossAttnUpBlock2D"}),
    ):
        if len(config[key]) != count or any(v not in allowed for v in config[key]):
            raise ValueError(f"Unsupported {key}")
    heads = _per_block(config["attention_head_dim"], count, "attention_head_dim")
    if any(c % h for c, h in zip(channels, heads)):
        raise ValueError("attention_head_dim (head count) must divide block channels")
    time_dim = config["addition_time_embed_dim"]
    projection_dim = config["projection_class_embeddings_input_dim"]
    if type(time_dim) is not int or time_dim < 2 or time_dim % 2:
        raise ValueError("addition_time_embed_dim must be a positive even integer")
    # The upstream Transformer2D constructor initially creates 32-group norms;
    # they are replaced below, but its initial construction must also be valid.
    if any(c % 32 for c in channels):
        raise ValueError("Apple's UNet example requires block channels divisible by 32")
    if type(projection_dim) is not int or projection_dim <= 6 * time_dim:
        raise ValueError("Projection input must include pooled text and six time IDs")
    norm_eps = config.get("norm_eps", 1e-5)
    if (
        not isinstance(norm_eps, (int, float))
        or not math.isfinite(norm_eps)
        or norm_eps <= 0
    ):
        raise ValueError("norm_eps must be positive and finite")
    kernels = {}
    for key in ("conv_in_kernel", "conv_out_kernel"):
        value = config.get(key, 3)
        if type(value) is not int or value < 1 or value % 2 != 1:
            raise ValueError(f"{key} must be a positive odd integer")
        kernels[key] = value
    return dict(
        in_channels=4,
        out_channels=4,
        **kernels,
        block_out_channels=channels,
        layers_per_block=_per_block(
            config.get("layers_per_block", 2), count, "layers_per_block"
        ),
        transformer_layers_per_block=_per_block(
            config.get("transformer_layers_per_block", 1),
            count,
            "transformer_layers_per_block",
        ),
        num_attention_heads=heads,
        cross_attention_dim=_per_block(
            config["cross_attention_dim"], count, "cross_attention_dim"
        ),
        norm_num_groups=groups,
        down_block_types=list(config["down_block_types"]),
        # Apple's constructor reverses its block traversal a second time.
        up_block_types=list(reversed(config["up_block_types"])),
        addition_embed_type="text_time",
        addition_time_embed_dim=time_dim,
        projection_class_embeddings_input_dim=projection_dim,
    )


def tiny_sdxl_config():
    """Small random SDXL-shaped fixture for forward/gradient parity checks."""
    return {
        "sample_size": 8,
        "in_channels": 4,
        "out_channels": 4,
        "block_out_channels": [32, 64],
        "layers_per_block": 1,
        "transformer_layers_per_block": [1, 1],
        "attention_head_dim": [4, 8],
        "cross_attention_dim": 64,
        "norm_num_groups": 32,
        "norm_eps": 1e-5,
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "UpBlock2D"],
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 8,
        "projection_class_embeddings_input_dim": 80,
        "use_linear_projection": True,
    }


def normalize_examples_path(path):
    """Accept either the mlx-examples checkout or its stable_diffusion folder."""
    path = Path(path).expanduser().resolve()
    for package in (
        path / "stable_diffusion" / "stable_diffusion",
        path / "stable_diffusion",
    ):
        if all(
            (package / name).is_file()
            for name in ("unet.py", "config.py", "model_io.py")
        ):
            return package
    raise ValueError(f"No Apple stable_diffusion package under {path}")


@functools.lru_cache(maxsize=None)
def _upstream(package_path):
    # Import under a private package name so another installed stable_diffusion
    # package cannot accidentally provide the model implementation.
    name = "_kohya_mlx_sd_" + hashlib.sha256(package_path.encode()).hexdigest()[:12]
    package = types.ModuleType(name)
    package.__path__ = [package_path]
    sys.modules[name] = package
    return (
        importlib.import_module(name + ".unet"),
        importlib.import_module(name + ".config"),
        importlib.import_module(name + ".model_io"),
    )


def _mlx_dtype(dtype):
    import mlx.core as mx

    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16 or bfloat16")
    return getattr(mx, dtype)


def create_unet(config, examples_path, dtype="float32"):
    """Construct the validated UNet, correcting Diffusers normalization details."""
    kwargs = config_to_unet_kwargs(config)
    unet, upstream_config, _ = _upstream(str(normalize_examples_path(examples_path)))
    import mlx.nn as nn

    model = unet.UNetModel(upstream_config.UNetConfig(**kwargs))
    groups = kwargs["norm_num_groups"]
    eps = config.get("norm_eps", 1e-5)
    for _, module in model.named_modules():
        if isinstance(module, unet.Transformer2D):
            # Diffusers Transformer2D fixes eps=1e-6 independently of norm_eps.
            module.norm = nn.GroupNorm(
                groups,
                module.proj_in.weight.shape[1],
                eps=1e-6,
                pytorch_compatible=True,
            )
        elif isinstance(module, unet.ResnetBlock2D):
            for attr in ("norm1", "norm2"):
                norm = getattr(module, attr)
                setattr(
                    module,
                    attr,
                    nn.GroupNorm(
                        groups, norm.weight.size, eps=eps, pytorch_compatible=True
                    ),
                )
    model.conv_norm_out = nn.GroupNorm(
        groups, kwargs["block_out_channels"][0], eps=eps, pytorch_compatible=True
    )
    model.set_dtype(_mlx_dtype(dtype))
    return model


def load_unet(config, weights_path, examples_path, dtype="float32"):
    """Load a local, unsharded Diffusers UNet safetensors file strictly."""
    import mlx.core as mx

    path = Path(weights_path).expanduser().resolve()
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError("Expected an existing UNet .safetensors file")
    model = create_unet(config, examples_path, dtype)
    _, _, model_io = _upstream(str(normalize_examples_path(examples_path)))
    tensor_dtype = _mlx_dtype(dtype)
    weights = [
        mapped
        for key, value in mx.load(str(path)).items()
        for mapped in model_io.map_unet_weights(key, value.astype(tensor_dtype))
    ]
    model.load_weights(weights, strict=True)
    return model


@functools.lru_cache(maxsize=1)
def _lora_class():
    import mlx.core as mx
    import mlx.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base, rank, alpha):
            super().__init__()
            self.linear = base
            self.rank = rank
            self.alpha = float(alpha)
            out_features, in_features = base.weight.shape
            limit = 1 / math.sqrt(in_features)
            self.lora_A = mx.random.uniform(
                -limit, limit, (rank, in_features), dtype=mx.float32
            )
            self.lora_B = mx.zeros((out_features, rank), dtype=mx.float32)

        def __call__(self, x):
            base = self.linear(x)
            update = (x.astype(mx.float32) @ self.lora_A.T) @ self.lora_B.T
            return base + (update * (self.alpha / self.rank)).astype(base.dtype)

    return LoRALinear


def _validate_lora(rank, alpha):
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be positive and finite")


def inject_lora(model, rank=4, alpha=4.0):
    """Freeze the base UNet, replace attention projections, return their paths."""
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    _validate_lora(rank, alpha)
    adapter_type = _lora_class()
    if any(isinstance(module, adapter_type) for _, module in model.named_modules()):
        raise ValueError("This model already has LoRA adapters")
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _ATTENTION_PATH.fullmatch(name) and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise ValueError("No SDXL attention projections found")
    model.freeze()
    model.update_modules(
        tree_unflatten(
            [(name, adapter_type(module, rank, alpha)) for name, module in targets]
        )
    )
    return [name for name, _ in targets]


@functools.lru_cache(maxsize=None)
def _checkpoint_class(original_type):
    from mlx.nn.utils import checkpoint

    class CheckpointedTransformerBlock(original_type):
        _mlx_lora_checkpointed = True

        def __call__(self, *args, **kwargs):
            forward = functools.partial(original_type.__call__, self)
            return checkpoint(self, forward)(*args, **kwargs)

    return CheckpointedTransformerBlock


def enable_gradient_checkpointing(model):
    """Recompute transformer blocks on backward without changing parameter paths.

    MLX's module-aware helper passes trainable parameters as explicit checkpoint
    inputs. Only pure transformer blocks are wrapped; UNet blocks mutate their
    residual-input lists and cannot safely be replayed as-is.
    """
    count = 0
    for name, module in model.named_modules():
        if re.search(r"\.transformer_blocks\.\d+$", name):
            if not getattr(type(module), "_mlx_lora_checkpointed", False):
                object.__setattr__(module, "__class__", _checkpoint_class(type(module)))
            count += 1
    if not count:
        raise ValueError("No SDXL transformer blocks to checkpoint")
    return count


def diffusers_lora_prefix(module_path):
    """Reverse the upstream attention naming without changing tensor orientation."""
    if not _ATTENTION_PATH.fullmatch(module_path):
        raise ValueError(f"Not an SDXL attention projection: {module_path}")
    path = module_path.replace("mid_blocks.1.", "mid_block.attentions.0.")
    for source, target in (
        ("query_proj", "to_q"),
        ("key_proj", "to_k"),
        ("value_proj", "to_v"),
        ("out_proj", "to_out.0"),
    ):
        if path.endswith("." + source):
            path = path[: -len(source)] + target
            break
    return "unet." + path


def _export_lora(model, path, alpha, rank, kohya):
    import mlx.core as mx

    _validate_lora(rank, alpha)
    path = Path(path).expanduser().resolve()
    if path.suffix != ".safetensors":
        raise ValueError("LoRA output must end in .safetensors")
    if path.exists():
        raise FileExistsError(path)
    adapter_type = _lora_class()
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, adapter_type):
            if module.rank != rank or module.alpha != float(alpha):
                raise ValueError("Export rank/alpha must match the injected adapters")
            prefix = diffusers_lora_prefix(name)
            if kohya:
                prefix = "lora_" + prefix.replace(".", "_")
                tensors[prefix + ".lora_down.weight"] = module.lora_A.astype(mx.float32)
                tensors[prefix + ".lora_up.weight"] = module.lora_B.astype(mx.float32)
                tensors[prefix + ".alpha"] = mx.array(alpha, dtype=mx.float32)
            else:
                tensors[prefix + ".lora_A.weight"] = module.lora_A.astype(mx.float32)
                tensors[prefix + ".lora_B.weight"] = (
                    module.lora_B * (alpha / rank)
                ).astype(mx.float32)
    if not tensors:
        raise ValueError("No LoRA adapters to export")
    mx.eval(tensors)
    if not all(bool(mx.all(mx.isfinite(value)).item()) for value in tensors.values()):
        raise ValueError("Refusing to export non-finite LoRA weights")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "pt",
        "backend": "mlx-sdxl-experimental",
        "lora_rank": str(rank),
        "training_alpha": str(alpha),
        "export_alpha": str(alpha if kohya else rank),
        "scaling": (
            "stored per-module alpha"
            if kohya
            else "alpha/rank baked into lora_B; load with alpha=rank"
        ),
        "adapter_format": "kohya" if kohya else "diffusers_peft",
        "mlx_examples_reference_revision": UPSTREAM_REVISION,
    }
    if kohya:
        metadata.update(
            ss_network_module="networks.lora",
            ss_network_dim=str(rank),
            ss_network_alpha=str(alpha),
            ss_base_model_version="sdxl_base_v1-0",
        )
    fd, temporary = tempfile.mkstemp(
        prefix=".mlx-lora-", suffix=".safetensors", dir=path.parent
    )
    os.close(fd)
    try:
        mx.save_safetensors(temporary, tensors, metadata=metadata)
        # A hard link publishes the finished file and also prevents race-driven
        # overwrites, unlike os.replace(). Both files are on the same filesystem.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return {
        "path": str(path),
        "tensor_count": len(tensors),
        "adapter_count": len(tensors) // (3 if kohya else 2),
        **metadata,
    }


def export_lora(model, path, alpha, rank):
    """Export Diffusers/PEFT adapters with baked scale, refusing output overwrite."""
    return _export_lora(model, path, alpha, rank, kohya=False)


def export_kohya_lora(model, path, alpha, rank):
    """Export Kohya adapter names with an explicit alpha tensor for each layer."""
    return _export_lora(model, path, alpha, rank, kohya=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-config",
        type=Path,
        help="Validate a local Diffusers UNet config.json without loading MLX",
    )
    parser.add_argument(
        "--print-tiny-config",
        action="store_true",
        help="Print the random SDXL parity fixture configuration",
    )
    args = parser.parse_args()
    if args.validate_config:
        print(
            json.dumps(
                config_to_unet_kwargs(json.loads(args.validate_config.read_text())),
                indent=2,
            )
        )
    elif args.print_tiny_config:
        print(json.dumps(tiny_sdxl_config(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
