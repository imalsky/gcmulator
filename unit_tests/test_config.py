"""Config-schema regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import load_config


def _minimal_config_dict() -> dict[str, object]:
    """Return a small but fully valid config payload."""
    return {
        "paths": {
            "dataset_dir": "data/raw",
            "processed_dir": "data/processed",
            "model_dir": "models/model_flow",
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
            "generation_workers": 2,
            "burn_in_days": 0.0,
            "transitions_per_simulation": 2,
            "transition_jump_steps": 1,
            "transition_jump_steps_max": 1,
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
            "state": {
                "field_transforms": {
                    "Phi": "signed_log1p",
                    "U": "signed_log1p",
                    "V": "signed_log1p",
                    "eta": "signed_log1p",
                    "delta": "signed_log1p",
                },
            },
            "params": {"mode": "zscore", "eps": 1.0e-8},
        },
        "model": {
            "grid": "legendre-gauss",
            "grid_internal": "legendre-gauss",
            "scale_factor": 2,
            "embed_dim": 64,
            "num_layers": 2,
            "encoder_layers": 2,
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
            "include_coord_channels": True,
        },
        "training": {
            "seed": 0,
            "device": "cpu",
            "amp_mode": "none",
            "optimizer": "adamw",
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": False,
            "preload_to_gpu": False,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "split_seed": 0,
            "scheduler": {"type": "none", "warmup_epochs": 0, "min_lr": 0.0},
        },
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Serialize a config dict into a temp JSON file."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_load_config_allows_zero_burn_in(tmp_path: Path) -> None:
    """Zero burn-in must remain a supported configuration."""
    config_path = _write_config(tmp_path, _minimal_config_dict())
    cfg = load_config(config_path)

    assert cfg.sampling.burn_in_days == pytest.approx(0.0)
    assert cfg.sampling.generation_workers == 2


def test_load_config_accepts_variable_transition_jump_range(tmp_path: Path) -> None:
    """Variable jump training should parse as an explicit solver-step range."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["transition_jump_steps"] = 2
    sampling_section["transition_jump_steps_max"] = 4
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.sampling.min_transition_jump_steps() == 2
    assert cfg.sampling.max_transition_jump_steps() == 4
    assert cfg.sampling.uses_variable_transition_jump() is True


def test_load_config_rejects_inverted_transition_jump_range(tmp_path: Path) -> None:
    """The max jump must not be smaller than the min jump."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["transition_jump_steps"] = 4
    sampling_section["transition_jump_steps_max"] = 2
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="transition_jump_steps_max"):
        load_config(config_path)


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    """Unknown public config keys must be rejected early."""
    payload = _minimal_config_dict()
    model_section = dict(payload["model"])
    model_section["unexpected"] = 123
    payload["model"] = model_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(config_path)


def test_load_config_defaults_to_plateau_with_relative_min_lr(tmp_path: Path) -> None:
    """Default scheduler settings should resolve from the configured start LR."""
    payload = _minimal_config_dict()
    training_section = dict(payload["training"])
    training_section.pop("scheduler")
    training_section["learning_rate"] = 1.0e-3
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.training.scheduler.type == "plateau"
    assert cfg.training.scheduler.warmup_epochs == 10
    assert cfg.training.scheduler.factor == pytest.approx(0.5)
    assert cfg.training.scheduler.patience == 10
    assert cfg.training.scheduler.min_lr == pytest.approx(2.0e-5)
    assert cfg.training.scheduler.eps == pytest.approx(1.0e-10)
