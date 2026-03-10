"""Runtime-order regression tests for training entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gcmulator.training as training_mod
from gcmulator.config import load_config


def _config_dict(*, device: str) -> dict[str, object]:
    """Return a small but valid training config."""
    step_days = 240.0 / 86400.0
    fixed_jump_days = 2.0 * step_days
    return {
        "paths": {
            "dataset_dir": "raw",
            "processed_dir": "processed",
            "model_dir": "models",
            "overwrite_dataset": False,
        },
        "solver": {
            "M": 42,
            "dt_seconds": 240.0,
            "default_time_days": 0.05,
            "starttime_index": 2,
        },
        "geometry": {
            "flip_latitude_to_north_south": True,
            "roll_longitude_to_0_2pi": True,
        },
        "sampling": {
            "seed": 0,
            "n_sims": 2,
            "generation_workers": 0,
            "burn_in_days": 0.0,
            "saved_checkpoint_interval_days": fixed_jump_days,
            "live_pairs_per_sequence": 2,
            "live_transition_days_min": fixed_jump_days,
            "live_transition_days_max": fixed_jump_days,
            "live_transition_tolerance_fraction": 0.1,
            "parameters": [
                {"name": "a_m", "dist": "fixed", "value": 8.2e7},
                {"name": "omega_rad_s", "dist": "fixed", "value": 3.2e-5},
                {"name": "Phibar", "dist": "fixed", "value": 3.0e5},
                {"name": "DPhieq", "dist": "fixed", "value": 1.0e6},
                {"name": "taurad_hours", "dist": "fixed", "value": 10.0},
                {"name": "taudrag_hours", "dist": "fixed", "value": 6.0},
                {"name": "g_m_s2", "dist": "fixed", "value": 9.8},
            ],
        },
        "normalization": {
            "state": {"field_transforms": {}},
            "params": {"mode": "zscore", "eps": 1.0e-8},
        },
        "model": {
            "grid": "legendre-gauss",
            "grid_internal": "legendre-gauss",
            "scale_factor": 2,
            "embed_dim": 32,
            "num_layers": 1,
            "encoder_layers": 1,
            "activation_function": "gelu",
            "use_mlp": True,
            "mlp_ratio": 2.0,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "normalization_layer": "instance_norm",
            "hard_thresholding_fraction": 1.0,
            "residual_prediction": True,
            "residual_init_scale": 1.0e-2,
            "pos_embed": "spectral",
            "bias": False,
        },
        "training": {
            "seed": 0,
            "device": device,
            "amp_mode": "none",
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "shuffle": True,
            "preload_to_gpu": True,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "split_seed": 0,
            "scheduler": {"type": "none", "warmup_epochs": 0, "min_lr": 0.0},
        },
    }


def test_train_emulator_rejects_cpu_before_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported CPU training should fail before dependency/setup work starts."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config_dict(device="cpu")), encoding="utf-8")
    cfg = load_config(config_path)

    def _unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("runtime validation should fail before setup helpers run")

    monkeypatch.setattr(training_mod, "ensure_torch_harmonics_importable", _unexpected)
    monkeypatch.setattr(training_mod, "preprocess_dataset", _unexpected)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        training_mod.train_emulator(cfg, config_path=config_path)

    assert not (tmp_path / "processed").exists()
    assert not (tmp_path / "models").exists()
