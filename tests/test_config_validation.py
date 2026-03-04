from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config


def _valid_config_dict() -> dict:
    return {
        "paths": {
            "dataset_dir": "raw",
            "processed_dir": "processed",
            "model_dir": "model",
        },
        "solver": {
            "M": 42,
            "dt_seconds": 60.0,
            "default_time_days": 100.0,
            "starttime_index": 2,
        },
        "sampling": {
            "seed": 1,
            "n_sims": 8,
            "parameters": [
                {"name": "a_m", "dist": "fixed", "value": 8.2e7},
                {"name": "omega_rad_s", "dist": "fixed", "value": 3.2e-5},
                {"name": "Phibar", "dist": "fixed", "value": 3.0e5},
                {"name": "DPhieq", "dist": "fixed", "value": 1.0e6},
                {"name": "taurad_hours", "dist": "uniform", "min": 1.0, "max": 30.0},
                {"name": "taudrag_hours", "dist": "uniform", "min": 1.0, "max": 30.0},
                {"name": "g_m_s2", "dist": "fixed", "value": 9.8},
                {"name": "K6", "dist": "fixed", "value": 1.24e33},
                {"name": "K6Phi", "dist": "fixed", "value": 0.0},
            ],
        },
        "model": {
            "rollout_steps_at_default_time": 16,
            "ic": {
                "basis": ["const", "sin_lat"],
                "hidden_dim": 16,
                "num_layers": 1,
                "activation": "gelu",
                "rand_basis_count": 1,
                "rand_basis_max_k": 3,
            },
        },
        "training": {
            "device": "cpu",
            "amp_mode": "none",
            "epochs": 1,
            "batch_size": 2,
            "val_fraction": 0.25,
            "scheduler": {"type": "none"},
        },
    }


def _write_config(tmp_path: Path, cfg_dict: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    return p


def test_load_config_valid_minimal(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path, _valid_config_dict()))
    assert cfg.solver.M == 42


def test_load_config_fails_on_duplicate_sampling_names(tmp_path: Path) -> None:
    cfg = _valid_config_dict()
    cfg["sampling"]["parameters"].append({"name": "K6", "dist": "fixed", "value": 1.0e30})
    p = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="duplicate names"):
        load_config(p)


def test_load_config_requires_exactly_one_taurad_alias(tmp_path: Path) -> None:
    cfg = _valid_config_dict()
    cfg["sampling"]["parameters"].append({"name": "taurad_s", "dist": "fixed", "value": 3600.0})
    p = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="exactly one of \\['taurad_s','taurad_hours'\\]"):
        load_config(p)


def test_load_config_fails_on_invalid_val_fraction(tmp_path: Path) -> None:
    cfg = _valid_config_dict()
    cfg["training"]["val_fraction"] = 1.0
    p = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="val_fraction"):
        load_config(p)


def test_load_config_fails_on_negative_generation_workers(tmp_path: Path) -> None:
    cfg = _valid_config_dict()
    cfg["sampling"]["generation_workers"] = -1
    p = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="generation_workers"):
        load_config(p)


def test_load_config_fails_on_unknown_keys(tmp_path: Path) -> None:
    cfg = _valid_config_dict()
    cfg["solver"]["dt_secs_typo"] = 60.0
    p = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(p)
