"""High-value generation integrity tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from config import load_config
from data_generation import generate_dataset
from my_swamp_backend import run_trajectory_window, run_trajectory_windows_batched
from sampling import to_extended9


pytest.importorskip("my_swamp")


def _sample_params() -> list:
    """Return two deterministic parameter sets for trajectory tests."""
    return [
        to_extended9(
            {
                "a_m": 8.2e7,
                "omega_rad_s": 3.2e-5,
                "Phibar": 3.0e5,
                "DPhieq": 1.0e6,
                "taurad_s": 10.0 * 3600.0,
                "taudrag_s": 6.0 * 3600.0,
                "g_m_s2": 9.8,
            }
        ),
        to_extended9(
            {
                "a_m": 8.4e7,
                "omega_rad_s": 3.0e-5,
                "Phibar": 3.2e5,
                "DPhieq": 9.0e5,
                "taurad_s": 12.0 * 3600.0,
                "taudrag_s": 8.0 * 3600.0,
                "g_m_s2": 9.8,
            }
        ),
    ]


def test_batched_trajectory_windows_match_serial() -> None:
    """The batched trajectory extractor must preserve serial results."""
    params_list = _sample_params()
    anchor_steps_batch = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int64)
    target_steps_batch = np.array([[1, 4, 5], [2, 5, 7]], dtype=np.int64)
    serial = [
        run_trajectory_window(
            params,
            M=42,
            dt_seconds=240.0,
            time_days=0.05,
            starttime_index=2,
            anchor_steps=anchor_steps_batch[index],
            target_steps=target_steps_batch[index],
        )
        for index, params in enumerate(params_list)
    ]
    params_matrix = np.stack([params.to_vector() for params in params_list], axis=0)
    batched_inputs, batched_targets = run_trajectory_windows_batched(
        params_matrix,
        M=42,
        dt_seconds=240.0,
        time_days=0.05,
        starttime_index=2,
        anchor_steps_batch=anchor_steps_batch,
        target_steps_batch=target_steps_batch,
        k6=params_list[0].K6,
        k6phi=params_list[0].K6Phi,
    )

    for index in range(len(params_list)):
        state_inputs, state_targets = serial[index]
        assert np.allclose(state_inputs, batched_inputs[index])
        assert np.allclose(state_targets, batched_targets[index])


def test_generate_dataset_supports_zero_burn_in_and_batched_generation() -> None:
    """A minimal generation run should work with burn-in disabled and batch size > 1."""
    step_days = 240.0 / 86400.0
    cfg_dict = {
        "paths": {
            "dataset_dir": "raw",
            "processed_dir": "processed",
            "model_dir": "models",
            "overwrite_dataset": True,
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
            "transition_jump_days_min": step_days,
            "transition_jump_days_max": 2.0 * step_days,
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(cfg_dict), encoding="utf-8")
        cfg = load_config(config_path)
        manifest = generate_dataset(cfg, config_path=config_path)
        raw_files = sorted((root / "raw").glob("sim_*.npy"))
        raw_payloads = [
            np.load(path, allow_pickle=True).item()
            for path in raw_files
        ]

    assert manifest["n_sims_written"] == 2
    assert len(raw_files) == 2
    assert manifest["sampling"]["generation_workers"] == 2
    assert manifest["sampling"]["resolved_generation_batch_size"] == 2
    assert manifest["sampling"]["uses_variable_transition_jump"] is True
    for payload in raw_payloads:
        transition_days = np.asarray(payload["transition_days"], dtype=np.float64)
        anchor_steps = np.asarray(payload["anchor_steps"], dtype=np.int64)
        jump_steps = np.rint(transition_days * 86400.0 / 240.0).astype(np.int64)
        assert np.all(jump_steps >= 1)
        assert np.all(jump_steps <= 2)
        assert np.all(anchor_steps[:-1] <= anchor_steps[1:])
