"""Offline cache validation must fail before allocating the MLX model."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "mlx_sdxl_train", Path(__file__).parents[1] / "tools" / "mlx_sdxl_train.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def manifest_at(tmp_path, **updates):
    (tmp_path / "unet").mkdir()
    (tmp_path / "sample.npz").touch()
    data = {
        "format": "mlx-sdxl-cache",
        "version": 1,
        "latent_layout": "NHWC",
        "unet": "unet",
        "samples": [{"file": "sample.npz", "repeats": 3}],
    }
    data.update(updates)
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    return data


def test_manifest_accepts_cache_with_repeats(tmp_path):
    expected = manifest_at(tmp_path)
    root, actual = trainer.load_manifest(tmp_path)
    assert root == tmp_path.resolve()
    assert actual == expected


@pytest.mark.parametrize("path", ["../outside.npz", "/tmp/outside.npz", "missing.npz"])
def test_manifest_rejects_outside_or_missing_sample(tmp_path, path):
    manifest_at(tmp_path, samples=[{"file": path}])
    with pytest.raises(ValueError, match="Missing or out-of-cache"):
        trainer.load_manifest(tmp_path)


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True, 10001])
def test_invalid_repeats(tmp_path, repeats):
    manifest_at(tmp_path, samples=[{"file": "sample.npz", "repeats": repeats}])
    with pytest.raises(ValueError, match="repeats"):
        trainer.load_manifest(tmp_path)


def test_linear_noise_schedule_is_cumulative():
    alphas, prediction = trainer.scheduler_alphas(
        {
            "num_train_timesteps": 3,
            "beta_schedule": "linear",
            "beta_start": 0.1,
            "beta_end": 0.3,
        }
    )
    np.testing.assert_allclose(alphas, [0.9, 0.72, 0.504], rtol=1e-6)
    assert prediction == "epsilon"


def test_scaled_linear_schedule():
    alphas, _ = trainer.scheduler_alphas(
        {
            "num_train_timesteps": 3,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.01,
            "beta_end": 0.09,
        }
    )
    np.testing.assert_allclose(alphas, np.cumprod([0.99, 0.96, 0.91]), rtol=1e-6)


@pytest.mark.parametrize(
    "config",
    [
        {"prediction_type": "sample"},
        {"rescale_betas_zero_snr": True},
        {"beta_schedule": "cosine"},
        {"trained_betas": [0.1]},
        {"num_train_timesteps": 2, "trained_betas": [0.1, float("nan")]},
    ],
)
def test_unsupported_noise_schedule_fails(config):
    with pytest.raises(ValueError):
        trainer.scheduler_alphas(config)


def sample_and_config():
    sample = {
        "latents": np.zeros((1, 8, 8, 4), np.float32),
        "prompt_embeds": np.zeros((1, 77, 64), np.float32),
        "pooled_prompt_embeds": np.zeros((1, 32), np.float32),
        "time_ids": np.zeros((1, 6), np.float32),
    }
    config = {
        "cross_attention_dim": 64,
        "projection_class_embeddings_input_dim": 80,
        "addition_time_embed_dim": 8,
        "block_out_channels": [32, 64],
    }
    return sample, config


def test_valid_sample():
    trainer.validate_sample(*sample_and_config())


def test_equal_per_block_cross_attention_dimensions():
    sample, config = sample_and_config()
    config["cross_attention_dim"] = [64, 64]
    trainer.validate_sample(sample, config)


def test_unequal_per_block_cross_attention_dimensions():
    sample, config = sample_and_config()
    config["cross_attention_dim"] = [64, 128]
    with pytest.raises(ValueError, match="equal cross-attention"):
        trainer.validate_sample(sample, config)


@pytest.mark.parametrize(
    "key", ["latents", "prompt_embeds", "pooled_prompt_embeds", "time_ids"]
)
def test_nonfinite_features_rejected(key):
    sample, config = sample_and_config()
    sample[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        trainer.validate_sample(sample, config)


def test_nchw_latents_rejected():
    sample, config = sample_and_config()
    sample["latents"] = sample["latents"].transpose(0, 3, 1, 2)
    with pytest.raises(ValueError, match="shape"):
        trainer.validate_sample(sample, config)


def test_bad_latent_resolution_rejected():
    sample, config = sample_and_config()
    sample["latents"] = np.zeros((1, 7, 8, 4), np.float32)
    with pytest.raises(ValueError, match="downsampling"):
        trainer.validate_sample(sample, config)
