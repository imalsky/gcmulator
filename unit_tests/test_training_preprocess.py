"""Processed-shard regression tests for time-conditioned training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CONDITIONING_PARAM_NAMES, TRANSITION_TIME_NAME, load_config
from geometry import geometry_shift_for_nlon
from training import preprocess_dataset


def _config_dict() -> dict[str, object]:
    """Return a small preprocessing-only config."""
    fixed_jump_days = 2.0 * 240.0 / 86400.0
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
            "n_sims": 3,
            "generation_workers": 0,
            "burn_in_days": 0.0,
            "transitions_per_simulation": 2,
            "transition_jump_days_min": fixed_jump_days,
            "transition_jump_days_max": fixed_jump_days,
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


def _write_raw_payload(raw_dir: Path, *, sim_idx: int) -> None:
    """Write one small raw trajectory payload."""
    nlat = 2
    nlon = 4
    transition_jump_steps = 2
    transition_days = np.full((2,), transition_jump_steps * 240.0 / 86400.0, dtype=np.float64)
    anchor_steps = np.array([0, 2], dtype=np.int64)
    base = float(sim_idx + 1)
    state_inputs = np.stack(
        [
            np.full((5, nlat, nlon), fill_value=base, dtype=np.float64),
            np.full((5, nlat, nlon), fill_value=base + 1.0, dtype=np.float64),
        ],
        axis=0,
    )
    state_targets = np.stack(
        [
            np.full((3, nlat, nlon), fill_value=base + 2.0, dtype=np.float64),
            np.full((3, nlat, nlon), fill_value=base + 3.0, dtype=np.float64),
        ],
        axis=0,
    )
    params = np.array(
        [
            8.2e7 + 1.0e5 * sim_idx,
            3.2e-5,
            3.0e5,
            1.0e6,
            10.0 * 3600.0,
            6.0 * 3600.0,
            9.8,
        ],
        dtype=np.float64,
    )
    payload = {
        "state_inputs": state_inputs,
        "state_targets": state_targets,
        "transition_days": transition_days,
        "anchor_steps": anchor_steps,
        "input_fields": np.asarray(["Phi", "U", "V", "eta", "delta"], dtype=object),
        "target_fields": np.asarray(["Phi", "eta", "delta"], dtype=object),
        "params": params,
        "param_names": np.asarray(list(CONDITIONING_PARAM_NAMES), dtype=object),
        "default_time_days": np.asarray(0.05, dtype=np.float64),
        "burn_in_days": np.asarray(0.0, dtype=np.float64),
        "dt_seconds": np.asarray(240.0, dtype=np.float64),
        "starttime_index": np.asarray(2, dtype=np.int64),
        "transition_jump_days_min": np.asarray(transition_days[0], dtype=np.float64),
        "transition_jump_days_max": np.asarray(transition_days[0], dtype=np.float64),
        "n_transitions": np.asarray(2, dtype=np.int64),
        "M": np.asarray(42, dtype=np.int64),
        "nlat": np.asarray(nlat, dtype=np.int64),
        "nlon": np.asarray(nlon, dtype=np.int64),
        "lat_order": np.asarray("north_to_south", dtype=object),
        "lon_origin": np.asarray("0_to_2pi", dtype=object),
        "lon_shift": np.asarray(geometry_shift_for_nlon(nlon, True), dtype=np.int64),
    }
    np.save(raw_dir / f"sim_{sim_idx:06d}.npy", payload, allow_pickle=True)


def test_preprocess_dataset_writes_time_conditioned_shards(tmp_path: Path) -> None:
    """Processed shards should store per-sample conditioning with transition time appended."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config_dict()), encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for sim_idx in range(3):
        _write_raw_payload(raw_dir, sim_idx=sim_idx)

    cfg = load_config(config_path)
    meta = preprocess_dataset(cfg, config_path=config_path)

    assert meta["conditioning_names"] == list(CONDITIONING_PARAM_NAMES) + [TRANSITION_TIME_NAME]

    train_shard_path = tmp_path / "processed" / meta["splits"]["train"][0]["file"]
    with np.load(train_shard_path, allow_pickle=False) as npz:
        conditioning_norm = np.asarray(npz["conditioning_norm"], dtype=np.float32)
        params_norm = np.asarray(npz["params_norm"], dtype=np.float32)
        transition_days = np.asarray(npz["transition_days"], dtype=np.float64)
        anchor_steps = np.asarray(npz["anchor_steps"], dtype=np.int64)

    assert conditioning_norm.shape == (2, len(CONDITIONING_PARAM_NAMES) + 1)
    assert params_norm.shape == (len(CONDITIONING_PARAM_NAMES),)
    assert np.allclose(transition_days, np.full((2,), 2 * 240.0 / 86400.0))
    assert np.array_equal(anchor_steps, np.array([0, 2], dtype=np.int64))
