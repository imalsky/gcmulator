"""High-value generation integrity tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from gcmulator.config import PHYSICAL_STATE_FIELDS, load_config
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


def _sample_brown_dwarf_params():
    """Return one deterministic unforced brown-dwarf parameter set."""
    return to_extended9(
        {
            "a_m": 7.1492e7,
            "omega_rad_s": 0.0003422214219596725,
            "Phibar": 4.86e6,
            "DPhieq": 0.0,
            "taurad_s": 10.0 * 3600.0,
            "taudrag_s": 6.0 * 3600.0,
            "g_m_s2": 800.0,
        }
    )


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
    """A minimal unforced generation run should write checkpoint sequences with batch size > 1."""
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
            "forcing_mode": "unforced",
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
                {"name": "a_m", "dist": "fixed", "value": 7.1492e7},
                {"name": "omega_rad_s", "dist": "fixed", "value": 0.0003490658503988659},
                {"name": "Phibar", "dist": "fixed", "value": 3.0e5},
                {"name": "DPhieq", "dist": "fixed", "value": 0.0},
                {"name": "taurad_hours", "dist": "fixed", "value": 10.0},
                {"name": "taudrag_hours", "dist": "fixed", "value": 6.0},
                {"name": "g_m_s2", "dist": "fixed", "value": 300.0},
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
            "include_coord_channels": False,
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
    assert manifest["solver"]["forcing_mode"] == "unforced"
    assert manifest["solver"]["initial_condition_mode"] == "legacy"
    assert manifest["solver"]["convective_forcing_mode"] == "none"
    assert manifest["solver"]["convective_forcing_seed"] == 0
    assert manifest["solver"]["initial_phi_noise_temperature_k"] == pytest.approx(0.0)
    assert manifest["n_saved_checkpoints"] == int(expected_schedule.checkpoint_steps.shape[0])
    assert np.allclose(manifest["checkpoint_days"], expected_schedule.checkpoint_days)
    for payload_index, payload in enumerate(raw_payloads):
        checkpoint_states = np.asarray(payload["checkpoint_states"], dtype=np.float64)
        checkpoint_steps = np.asarray(payload["checkpoint_steps"], dtype=np.int64)
        checkpoint_days = np.asarray(payload["checkpoint_days"], dtype=np.float64)
        forcing_mode = str(np.asarray(payload["forcing_mode"], dtype=object).item())
        initial_condition_mode = str(
            np.asarray(payload["initial_condition_mode"], dtype=object).item()
        )
        convective_forcing_mode = str(
            np.asarray(payload["convective_forcing_mode"], dtype=object).item()
        )
        trajectory_seed = int(np.asarray(payload["trajectory_seed"], dtype=np.int64).item())
        assert "transition_days" not in payload
        assert "anchor_steps" not in payload
        assert checkpoint_states.ndim == 4
        assert checkpoint_states.shape[0] == expected_schedule.checkpoint_steps.shape[0]
        assert checkpoint_states.shape[1] == len(PHYSICAL_STATE_FIELDS)
        assert forcing_mode == "unforced"
        assert initial_condition_mode == "legacy"
        assert convective_forcing_mode == "none"
        assert trajectory_seed == payload_index
        assert np.all(np.isfinite(checkpoint_states))
        assert np.array_equal(checkpoint_steps, expected_schedule.checkpoint_steps)
        assert np.allclose(checkpoint_days, expected_schedule.checkpoint_days)


def test_unforced_rest_initial_condition_stays_quiescent() -> None:
    """The explicit rest-state initializer should preserve a quiescent unforced run."""
    params = _sample_brown_dwarf_params()
    checkpoint_steps = np.array([0, 50], dtype=np.int64)
    checkpoint_states = run_trajectory_checkpoints(
        params,
        M=42,
        dt_seconds=30.0,
        time_days=50.0 * 30.0 / 86400.0,
        starttime_index=2,
        checkpoint_steps=checkpoint_steps,
        forcing_mode="unforced",
        initial_condition_mode="rest",
    )

    final_state = checkpoint_states[-1]
    assert np.max(np.abs(final_state[0])) < 1.0e-3
    assert np.max(np.abs(final_state[1])) < 1.0e-6
    assert np.max(np.abs(final_state[2])) < 1.0e-6
    assert np.max(np.abs(final_state[4])) < 1.0e-6


def test_unforced_legacy_initial_condition_evolves_nontrivially() -> None:
    """The legacy unforced initializer should still produce nontrivial evolution."""
    params = _sample_brown_dwarf_params()
    checkpoint_steps = np.array([0, 50], dtype=np.int64)
    checkpoint_states = run_trajectory_checkpoints(
        params,
        M=42,
        dt_seconds=30.0,
        time_days=50.0 * 30.0 / 86400.0,
        starttime_index=2,
        checkpoint_steps=checkpoint_steps,
        forcing_mode="unforced",
        initial_condition_mode="legacy",
    )

    initial_phi = checkpoint_states[0, 0]
    final_phi = checkpoint_states[-1, 0]
    assert np.max(np.abs(final_phi - initial_phi)) > 1.0


def test_stochastic_convective_forcing_is_seed_deterministic_and_batched_matches_serial() -> None:
    """Localized random storms should be deterministic for a fixed seed and batch-consistent."""
    params = _sample_brown_dwarf_params()
    checkpoint_steps = np.array([0, 25, 50], dtype=np.int64)
    kwargs = dict(
        M=42,
        dt_seconds=30.0,
        time_days=50.0 * 30.0 / 86400.0,
        starttime_index=2,
        checkpoint_steps=checkpoint_steps,
        forcing_mode="unforced",
        initial_condition_mode="rest",
        convective_forcing_mode="localized_random_storms",
        initial_phi_noise_temperature_k=10.0,
        r_specific_j_per_kg_k=3600.0,
        storm_radius_degrees=2.0,
        storm_nondim_lifetime=20.0,
        storm_nondim_interval=20.0,
        storm_strength_fraction=0.1,
    )
    serial_a = run_trajectory_checkpoints(params, trajectory_seed=7, **kwargs)
    serial_b = run_trajectory_checkpoints(params, trajectory_seed=7, **kwargs)
    serial_c = run_trajectory_checkpoints(params, trajectory_seed=8, **kwargs)

    assert np.allclose(serial_a, serial_b)
    assert not np.allclose(serial_a, serial_c)

    batched = run_trajectory_checkpoints_batched(
        np.stack([params.to_vector(), params.to_vector()], axis=0),
        M=42,
        dt_seconds=30.0,
        time_days=50.0 * 30.0 / 86400.0,
        starttime_index=2,
        checkpoint_steps_batch=np.repeat(checkpoint_steps[None, :], 2, axis=0),
        k6=params.K6,
        k6phi=params.K6Phi,
        forcing_mode="unforced",
        initial_condition_mode="rest",
        convective_forcing_mode="localized_random_storms",
        trajectory_seeds=np.asarray([7, 8], dtype=np.int64),
        initial_phi_noise_temperature_k=10.0,
        r_specific_j_per_kg_k=3600.0,
        storm_radius_degrees=2.0,
        storm_nondim_lifetime=20.0,
        storm_nondim_interval=20.0,
        storm_strength_fraction=0.1,
    )

    assert np.allclose(serial_a, batched[0])
    assert np.allclose(serial_c, batched[1])


def test_stochastic_convective_forcing_breaks_quiescence_and_keeps_mean_phi_small() -> None:
    """The stochastic brown-dwarf default should not remain flat after 500 steps."""
    from my_swamp import initial_conditions

    params = _sample_brown_dwarf_params()
    checkpoint_steps = np.array([0, 500], dtype=np.int64)
    checkpoint_states = run_trajectory_checkpoints(
        params,
        M=42,
        dt_seconds=30.0,
        time_days=500.0 * 30.0 / 86400.0,
        starttime_index=2,
        checkpoint_steps=checkpoint_steps,
        forcing_mode="unforced",
        initial_condition_mode="rest",
        convective_forcing_mode="localized_random_storms",
        trajectory_seed=42,
        initial_phi_noise_temperature_k=10.0,
        r_specific_j_per_kg_k=3600.0,
        storm_radius_degrees=2.0,
        storm_nondim_lifetime=20.0,
        storm_nondim_interval=20.0,
        storm_strength_fraction=0.1,
    )

    _, _, _, _, _, mus, weights = initial_conditions.spectral_params(42)
    final_phi = checkpoint_states[-1, 0]
    zonal_mean = np.mean(final_phi, axis=1)
    area_mean = float(np.sum(zonal_mean * np.asarray(weights)) / np.sum(np.asarray(weights)))
    area_rms = float(
        np.sqrt(
            np.sum(np.mean(np.square(final_phi), axis=1) * np.asarray(weights))
            / np.sum(np.asarray(weights))
        )
    )

    assert area_rms > 1.0
    assert np.max(np.abs(final_phi)) > 1.0
    assert abs(area_mean) < 0.1 * area_rms
