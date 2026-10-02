"""Offline preparation checks; no GPU or downloaded model weights are required."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "mlx_sdxl_prepare",
    Path(__file__).resolve().parents[1] / "tools/mlx_sdxl_prepare.py",
)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


@pytest.fixture
def inputs(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (128, 64), color=(255, 0, 128)).save(dataset / "sample.png")
    (dataset / "sample.txt").write_text("a", encoding="utf-8")
    config = tmp_path / "config"
    for name in prepare.CONFIG_FILES:
        path = config / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("{}" if path.suffix == ".json" else "#version: 0.2\n")
    (config / "unet/config.json").write_text(
        json.dumps(
            dict(
                in_channels=4, cross_attention_dim=2048, addition_embed_type="text_time"
            )
        )
    )
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"fixture; the mocked loader never opens this")
    return dict(
        checkpoint=checkpoint,
        config_dir=config,
        dataset=dataset,
        output_dir=tmp_path / "cache",
        resolution=64,
        seed=19,
    )


def test_repeat_folders_and_adjacent_captions(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    folder = tmp_path / "5_portrait person" / "nested"
    folder.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(folder / "a.png")
    Image.new("RGB", (8, 8)).save(folder / "b.png")
    (folder / "b.txt").write_text("\ufeff具体人物\n", encoding="utf-8")
    samples = prepare.discover_samples(tmp_path)
    assert [(s.caption, s.repeats) for s in samples] == [
        ("portrait person", 5),
        ("具体人物", 5),
    ]
    assert samples[0].relative_path == "5_portrait person/nested/a.png"


@pytest.mark.parametrize(
    "failure", ["output", "caption", "config", "image", "resolution", "refiner"]
)
def test_invalid_inputs_fail_before_model_load(inputs, monkeypatch, failure):
    def must_not_load(*args):
        pytest.fail("Pipeline loading must not run for invalid inputs")

    monkeypatch.setattr(prepare, "load_pipeline", must_not_load)
    if failure == "output":
        inputs["output_dir"].mkdir()
        (inputs["output_dir"] / "keep").write_text("keep")
    elif failure == "caption":
        (inputs["dataset"] / "sample.txt").write_text("   ")
    elif failure == "config":
        (inputs["config_dir"] / "tokenizer/vocab.json").unlink()
    elif failure == "image":
        (inputs["dataset"] / "sample.png").write_bytes(b"not an image")
    elif failure == "resolution":
        inputs["resolution"] = 65
    else:
        (inputs["config_dir"] / "unet/config.json").write_text("{}")
    with pytest.raises((ValueError, OSError)):
        prepare.prepare(**inputs)
    if failure == "output":
        assert (inputs["output_dir"] / "keep").read_text() == "keep"
    else:
        assert not inputs["output_dir"].exists()


def test_center_crop_and_time_id_coordinates(inputs):
    np = pytest.importorskip("numpy")
    pixels, original, crop = prepare.prepare_image(inputs["dataset"] / "sample.png", 64)
    assert pixels.shape == (3, 64, 64)
    assert original == [64, 128]
    assert crop == [0, 32]
    np.testing.assert_allclose(pixels[0], 1)
    np.testing.assert_allclose(pixels[1], -1)


def test_tokenizer_length_and_truncation_warning(tmp_path):
    tokenizer = lambda prompt, **kwargs: SimpleNamespace(input_ids=list(range(80)))
    tokenizer.model_max_length = 77
    pipe = SimpleNamespace(tokenizer=tokenizer, tokenizer_2=tokenizer)
    samples = [prepare.Sample(tmp_path / "a.png", "a.png", "long caption", 1)]
    with pytest.warns(UserWarning, match="truncated") as caught:
        prepare.check_tokenizers(pipe, samples)
    assert len(caught) == 2
    tokenizer.model_max_length = 512
    with pytest.raises(ValueError, match="model_max_length=77"):
        prepare.check_tokenizers(pipe, samples)


def test_rejects_unsupported_augmentation_cli():
    with pytest.raises(SystemExit) as error:
        prepare.main(
            [
                "--checkpoint",
                "a",
                "--config-dir",
                "b",
                "--dataset",
                "c",
                "--output-dir",
                "d",
                "--random-crop",
            ]
        )
    assert error.value.code == 2


def test_loader_uses_only_local_files_and_preserves_process_environment(
    inputs, monkeypatch
):
    import os

    diffusers = pytest.importorskip("diffusers")
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        return "loaded"

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_single_file", load)
    assert prepare.load_pipeline(inputs["checkpoint"], inputs["config_dir"]) == "loaded"
    assert calls[0][0] == str(inputs["checkpoint"])
    assert calls[0][1]["config"] == str(inputs["config_dir"])
    assert calls[0][1]["local_files_only"] is True
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def tiny_pipeline(config_dir):
    """Real Diffusers VAE/dual CLIP modules with random, small offline weights."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(123)
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}
    (config_dir / "tokenizer/vocab.json").write_text(json.dumps(vocab))
    tokenizer = transformers.CLIPTokenizer(
        str(config_dir / "tokenizer/vocab.json"),
        str(config_dir / "tokenizer/merges.txt"),
        model_max_length=77,
    )

    def text_config(width, heads):
        return transformers.CLIPTextConfig(
            vocab_size=3,
            hidden_size=width,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=heads,
            max_position_embeddings=77,
            projection_dim=1280,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )

    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4),
        norm_num_groups=4,
        layers_per_block=1,
        sample_size=64,
        scaling_factor=0.13025,
    )
    unet = diffusers.UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(4,),
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        norm_num_groups=4,
        cross_attention_dim=2048,
        attention_head_dim=1,
        addition_embed_type="text_time",
        addition_time_embed_dim=2,
        projection_class_embeddings_input_dim=1292,
    )
    return diffusers.StableDiffusionXLPipeline(
        vae=vae,
        unet=unet,
        text_encoder=transformers.CLIPTextModel(text_config(768, 12)),
        text_encoder_2=transformers.CLIPTextModelWithProjection(text_config(1280, 20)),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer,
        scheduler=diffusers.DDPMScheduler(
            beta_schedule="scaled_linear", prediction_type="epsilon"
        ),
        add_watermarker=False,
    )


def test_real_offline_encoders_cache_layout_and_fixed_seed(inputs, monkeypatch):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pipe = tiny_pipeline(inputs["config_dir"])
    original_unet = pipe.unet
    with torch.inference_mode():
        expected_prompt, _, expected_pooled, _ = pipe.encode_prompt(
            "a", do_classifier_free_guidance=False, device=torch.device("cpu")
        )
    monkeypatch.setattr(prepare, "load_pipeline", lambda *args: pipe)
    manifest = prepare.prepare(**inputs)
    assert manifest["format"] == "mlx-sdxl-cache"
    assert manifest["vae_scaling_factor"] == 0.13025
    assert manifest["samples"][0]["crop_top_left"] == [0, 32]
    assert (inputs["output_dir"] / "unet/diffusion_pytorch_model.safetensors").is_file()
    with np.load(inputs["output_dir"] / "data/000000.npz") as data:
        first_latents = data["latents"].copy()
        assert first_latents.shape == (1, 8, 8, 4)
        np.testing.assert_array_equal(data["time_ids"], [[64, 128, 0, 32, 64, 64]])
        np.testing.assert_allclose(
            data["prompt_embeds"], expected_prompt.numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            data["pooled_prompt_embeds"], expected_pooled.numpy(), rtol=1e-5, atol=1e-5
        )
        assert all(data[name].dtype == np.float32 for name in data.files)
    pipe.unet = original_unet
    inputs["output_dir"] = inputs["output_dir"].with_name("cache-again")
    prepare.prepare(**inputs)
    with np.load(inputs["output_dir"] / "data/000000.npz") as data:
        np.testing.assert_array_equal(data["latents"], first_latents)


def test_failed_encoder_does_not_publish_or_keep_partial_cache(inputs, monkeypatch):
    pipe = tiny_pipeline(inputs["config_dir"])

    def fail(**kwargs):
        raise ValueError("fixture encoding failure")

    monkeypatch.setattr(pipe, "encode_prompt", fail)
    monkeypatch.setattr(prepare, "load_pipeline", lambda *args: pipe)
    with pytest.raises(ValueError, match="fixture encoding failure"):
        prepare.prepare(**inputs)
    assert not inputs["output_dir"].exists()
    assert not list(inputs["output_dir"].parent.glob(".cache-*"))
