"""Backend guards that do not require MLX, a GPU, or pretrained model weights."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "mlx_sdxl_backend.py"
SPEC = importlib.util.spec_from_file_location("mlx_sdxl_backend", SCRIPT)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


def test_sdxl_config_preserves_block_order_and_attention_heads():
    config = backend.tiny_sdxl_config()
    converted = backend.config_to_unet_kwargs(config)
    assert converted["num_attention_heads"] == [4, 8]
    assert converted["up_block_types"] == ["UpBlock2D", "CrossAttnUpBlock2D"]
    assert converted["layers_per_block"] == [1, 1]
    assert converted["cross_attention_dim"] == [64, 64]


@pytest.mark.parametrize(
    "key,value",
    [
        ("addition_embed_type", None),
        ("time_cond_proj_dim", 256),
        ("resnet_time_scale_shift", "scale_shift"),
        ("dropout", 0.1),
        ("upcast_attention", True),
        ("only_cross_attention", [False, True]),
        ("flip_sin_to_cos", False),
        ("transformer_layers_per_block", [[1], [1]]),
        ("attention_head_dim", [3, 8]),
        ("norm_num_groups", 5),
        ("norm_eps", float("nan")),
        ("addition_time_embed_dim", 3),
        ("addition_time_embed_dim", None),
        ("projection_class_embeddings_input_dim", 48),
        ("conv_out_kernel", 2),
        ("down_block_types", ["ResnetDownsampleBlock2D"] * 2),
    ],
)
def test_unsupported_configs_fail_before_model_allocation(key, value):
    config = backend.tiny_sdxl_config()
    config[key] = value
    with pytest.raises(ValueError):
        backend.config_to_unet_kwargs(config)


def test_missing_config_field_has_clear_error():
    config = backend.tiny_sdxl_config()
    del config["projection_class_embeddings_input_dim"]
    with pytest.raises(ValueError, match="projection_class_embeddings_input_dim"):
        backend.config_to_unet_kwargs(config)


@pytest.mark.parametrize("root_is_stable_diffusion", [False, True])
def test_explicit_upstream_path_accepts_checkout_and_example(
    tmp_path, root_is_stable_diffusion
):
    example = tmp_path / "stable_diffusion"
    package = example / "stable_diffusion"
    package.mkdir(parents=True)
    for name in ("unet.py", "config.py", "model_io.py"):
        (package / name).touch()
    root = example if root_is_stable_diffusion else tmp_path
    assert backend.normalize_examples_path(root) == package


def test_incomplete_upstream_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="No Apple stable_diffusion package"):
        backend.normalize_examples_path(tmp_path)


@pytest.mark.parametrize(
    "path,expected",
    [
        (
            "down_blocks.1.attentions.0.transformer_blocks.1.attn1.query_proj",
            "unet.down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q",
        ),
        (
            "up_blocks.0.attentions.1.transformer_blocks.0.attn2.value_proj",
            "unet.up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v",
        ),
        (
            "mid_blocks.1.transformer_blocks.0.attn2.out_proj",
            "unet.mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0",
        ),
        (
            "mid_blocks.1.transformer_blocks.0.attn1.key_proj",
            "unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_k",
        ),
    ],
)
def test_export_keys_follow_diffusers_attention_paths(path, expected):
    assert backend.diffusers_lora_prefix(path) == expected


@pytest.mark.parametrize(
    "path",
    ["conv_in", "time_embedding.linear_1", "down_blocks.0.resnets.0.time_emb_proj"],
)
def test_export_refuses_non_attention_modules(path):
    with pytest.raises(ValueError, match="Not an SDXL attention projection"):
        backend.diffusers_lora_prefix(path)


@pytest.mark.parametrize("rank,alpha", [(0, 1), (True, 1), (1, 0), (1, float("inf"))])
def test_invalid_adapter_scale_is_rejected(rank, alpha):
    with pytest.raises(ValueError):
        backend._validate_lora(rank, alpha)
