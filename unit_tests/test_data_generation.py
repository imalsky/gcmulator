"""High-value generation integrity tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from gcmulator.config import load_config
from gcmulator.data_generation import generate_dataset
from gcmulator.my_swamp_backend import run_trajectory_checkpoints, run_trajectory_checkpoints_batched
from gcmulator.sampling import build_uniform_checkpoint_schedule, to_extended9


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


def test_batched_checkpoint_sequences_match_serial() -> None:
    """The batched checkpoint extractor must preserve serial results."""
    params_list = _sample_params()
    checkpoint_steps = np.array([0, 2, 4, 6], dtype=np.int64)
    serial = [
        run_trajectory_checkpoints(
            params,
            M=42,
            dt_seconds=240.0,
            time_days=0.05,
            starttime_index=2,
            checkpoint_steps=checkpoint_steps,
        )
        for params in params_list
    ]
    params_matrix = np.stack([params.to_vector() for params in params_list], axis=0)
    checkpoint_steps_batch = np.repeat(checkpoint_steps[None, :], len(params_list), axis=0)
    batched = run_trajectory_checkpoints_batched(
        params_matrix,
        M=42,
        dt_seconds=240.0,
        time_days=0.05,
        starttime_index=2,
        checkpoint_steps_batch=checkpoint_steps_batch,
        k6=params_list[0].K6,
        k6phi=params_list[0].K6Phi,
    )

    for index in range(len(params_list)):
        assert np.allclose(serial[index], batched[index])


def test_generate_dataset_supports_zero_burn_in_and_batched_generation() -> None:
    """A minimal generation run should write checkpoint sequences with batch size > 1."""
    step_days = 240.0 / 86400.0
    saved_interval_days = 2.0 * step_days
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
            "saved_checkpoint_interval_days": saved_interval_days,
            "live_pairs_per_sequence": 4,
            "live_transition_days_min": saved_interval_days,
            "live_transition_days_max": 2.0 * saved_interval_days,
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
            "pos_embed": "spectral",
            "bias": False,
        },
        "training": {
            "seed": 0,
            "device": "cpu",
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(cfg_dict), encoding="utf-8")
        cfg = load_config(config_path)
        manifest = generate_dataset(cfg, config_path=config_path)
        raw_files = sorted((root / "raw").glob("sim_*.npy"))
        raw_payloads = [np.load(path, allow_pickle=True).item() for path in raw_files]

    expected_schedule = build_uniform_checkpoint_schedule(
        time_days=0.05,
        dt_seconds=240.0,
        saved_checkpoint_interval_days=saved_interval_days,
    )
    assert manifest["n_sims_written"] == 2
    assert len(raw_files) == 2
    assert manifest["sampling"]["generation_workers"] == 2
    assert manifest["sampling"]["resolved_generation_batch_size"] == 2
    assert manifest["sampling"]["uses_variable_live_transition"] is True
    assert manifest["n_saved_checkpoints"] == int(expected_schedule.checkpoint_steps.shape[0])
    assert np.allclose(manifest["checkpoint_days"], expected_schedule.checkpoint_days)
    for payload in raw_payloads:
        checkpoint_states = np.asarray(payload["checkpoint_states"], dtype=np.float64)
        checkpoint_steps = np.asarray(payload["checkpoint_steps"], dtype=np.int64)
        checkpoint_days = np.asarray(payload["checkpoint_days"], dtype=np.float64)
        assert "transition_days" not in payload
        assert "anchor_steps" not in payload
        assert checkpoint_states.ndim == 4
        assert checkpoint_states.shape[0] == expected_schedule.checkpoint_steps.shape[0]
        assert checkpoint_states.shape[1] == 3
        assert np.array_equal(checkpoint_steps, expected_schedule.checkpoint_steps)
        assert np.allclose(checkpoint_days, expected_schedule.checkpoint_days)
