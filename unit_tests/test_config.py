"""Config-schema regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gcmulator.config import load_config


def _minimal_config_dict() -> dict[str, object]:
    """Return a small but fully valid config payload."""
    step_days = 240.0 / 86400.0
    fixed_jump_days = 2.0 * step_days
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
            "include_coord_channels": False,
            "pos_embed": "spectral",
            "bias": False,
        },
        "training": {
            "seed": 0,
            "device": "cpu",
            "amp_mode": "none",
            "deterministic": True,
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
    assert cfg.sampling.uses_variable_live_transition() is False


def test_load_config_accepts_variable_live_transition_range(tmp_path: Path) -> None:
    """Variable jump training should parse as an explicit day-valued range."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["live_transition_days_min"] = 0.01
    sampling_section["live_transition_days_max"] = 0.04
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.sampling.min_live_transition_days() == pytest.approx(0.01)
    assert cfg.sampling.max_live_transition_days() == pytest.approx(0.04)
    assert cfg.sampling.uses_variable_live_transition() is True


def test_load_config_resolves_fixed_transition_steps(tmp_path: Path) -> None:
    """A fixed solver-step jump should resolve onto the day-valued jump contract."""
    payload = _minimal_config_dict()
    solver_section = dict(payload["solver"])
    solver_section["dt_seconds"] = 120.0
    solver_section["default_time_days"] = 100.0
    payload["solver"] = solver_section

    sampling_section = dict(payload["sampling"])
    sampling_section.pop("saved_checkpoint_interval_days")
    sampling_section["saved_snapshots_per_sim"] = 1000
    sampling_section["fixed_transition_steps"] = 72
    sampling_section.pop("live_transition_days_min")
    sampling_section.pop("live_transition_days_max")
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.sampling.fixed_transition_steps == 72
    assert cfg.sampling.live_transition_days_min == pytest.approx(0.1)
    assert cfg.sampling.live_transition_days_max == pytest.approx(0.1)
    assert cfg.sampling.uses_variable_live_transition() is False


def test_load_config_rejects_inverted_live_transition_range(tmp_path: Path) -> None:
    """The max live jump must not be smaller than the min live jump."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["live_transition_days_min"] = 0.4
    sampling_section["live_transition_days_max"] = 0.1
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="live_transition_days_min"):
        load_config(config_path)


def test_load_config_rejects_fixed_steps_with_day_range(tmp_path: Path) -> None:
    """The fixed-step jump knob must not be combined with day-valued jump fields."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["fixed_transition_steps"] = 2
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(config_path)


def test_load_config_rejects_nonpositive_fixed_transition_steps(tmp_path: Path) -> None:
    """The fixed-step jump knob must stay strictly positive."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["fixed_transition_steps"] = 0
    sampling_section.pop("live_transition_days_min")
    sampling_section.pop("live_transition_days_max")
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="fixed_transition_steps"):
        load_config(config_path)


def test_load_config_rejects_unrepresentable_fixed_transition_steps(tmp_path: Path) -> None:
    """Fixed-step jumps must land exactly on the saved checkpoint cadence."""
    payload = _minimal_config_dict()
    solver_section = dict(payload["solver"])
    solver_section["dt_seconds"] = 120.0
    solver_section["default_time_days"] = 100.0
    payload["solver"] = solver_section

    sampling_section = dict(payload["sampling"])
    sampling_section.pop("saved_checkpoint_interval_days")
    sampling_section["saved_snapshots_per_sim"] = 1000
    sampling_section["fixed_transition_steps"] = 73
    sampling_section.pop("live_transition_days_min")
    sampling_section.pop("live_transition_days_max")
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="exactly representable"):
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


def test_load_config_accepts_include_coord_channels(tmp_path: Path) -> None:
    """The coordinate-channel option should parse as a supported model flag."""
    payload = _minimal_config_dict()
    model_section = dict(payload["model"])
    model_section["include_coord_channels"] = True
    payload["model"] = model_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.model.include_coord_channels is True


def test_load_config_rejects_removed_residual_init_scale(tmp_path: Path) -> None:
    """The scaled residual knob should no longer be part of the public config."""
    payload = _minimal_config_dict()
    model_section = dict(payload["model"])
    model_section["residual_init_scale"] = 1.0
    payload["model"] = model_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(config_path)


def test_load_config_defaults_to_plateau_with_relative_min_lr(tmp_path: Path) -> None:
    """Default scheduler settings should resolve from the configured start LR."""
    payload = _minimal_config_dict()
    training_section = dict(payload["training"])
    training_section.pop("scheduler")
    training_section.pop("deterministic")
    training_section.pop("preload_to_gpu")
    training_section["learning_rate"] = 1.0e-3
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.training.deterministic is False
    assert cfg.training.preload_to_gpu is True
    assert cfg.training.scheduler.type == "plateau"
    assert cfg.training.scheduler.warmup_epochs == 10
    assert cfg.training.scheduler.factor == pytest.approx(0.5)
    assert cfg.training.scheduler.patience == 10
    assert cfg.training.scheduler.min_lr == pytest.approx(2.0e-5)
    assert cfg.training.scheduler.eps == pytest.approx(1.0e-10)


def test_load_config_rejects_legacy_pair_sampling_keys(tmp_path: Path) -> None:
    """Legacy pair-sampling config keys should no longer be accepted."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["transitions_per_simulation"] = 4
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(config_path)


def test_load_config_accepts_channel_loss_weights(tmp_path: Path) -> None:
    """Explicit per-channel loss weights should parse in canonical field order."""
    payload = _minimal_config_dict()
    training_section = dict(payload["training"])
    training_section["channel_loss_weights"] = {
        "Phi": 1.0,
        "U": 2.0,
        "V": 3.0,
        "eta": 4.0,
        "delta": 5.0,
    }
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.training.channel_loss_weights == {
        "Phi": pytest.approx(1.0),
        "U": pytest.approx(2.0),
        "V": pytest.approx(3.0),
        "eta": pytest.approx(4.0),
        "delta": pytest.approx(5.0),
    }


def test_load_config_rejects_partial_channel_loss_weights(tmp_path: Path) -> None:
    """Per-channel weights must be explicit for all visible state fields."""
    payload = _minimal_config_dict()
    training_section = dict(payload["training"])
    training_section["channel_loss_weights"] = {
        "Phi": 1.0,
        "U": 2.0,
        "V": 3.0,
        "eta": 4.0,
    }
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="channel_loss_weights"):
        load_config(config_path)


def test_load_config_rejects_nonpositive_channel_loss_weights(tmp_path: Path) -> None:
    """Per-channel loss weights must stay finite and positive."""
    payload = _minimal_config_dict()
    training_section = dict(payload["training"])
    training_section["channel_loss_weights"] = {
        "Phi": 1.0,
        "U": 2.0,
        "V": 3.0,
        "eta": 0.0,
        "delta": 5.0,
    }
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="channel_loss_weights"):
        load_config(config_path)


def test_load_config_requires_batch_size_divisible_by_live_pairs(tmp_path: Path) -> None:
    """The effective pair batch must decompose into whole sequences."""
    payload = _minimal_config_dict()
    training_section = dict(payload["training"])
    training_section["batch_size"] = 3
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="batch_size"):
        load_config(config_path)


def test_load_config_derives_saved_interval_from_snapshot_count(tmp_path: Path) -> None:
    """Snapshot-count configs should resolve to the matching saved cadence."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section.pop("saved_checkpoint_interval_days")
    sampling_section["saved_snapshots_per_sim"] = 9
    sampling_section["pairs_per_sim"] = 7
    sampling_section["pair_sampling_policy"] = "uniform_pairs"
    payload["sampling"] = sampling_section

    training_section = dict(payload["training"])
    training_section["pair_iteration_mode"] = "resample_from_saved_sequences"
    training_section["preload_to_gpu"] = True
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    expected_interval_days = 2.0 * (240.0 / 86400.0)
    assert cfg.sampling.saved_snapshots_per_sim == 9
    assert cfg.sampling.saved_checkpoint_interval_days == pytest.approx(expected_interval_days)
    assert cfg.sampling.pairs_per_sim == 7
    assert cfg.sampling.pair_sampling_policy == "uniform_pairs"
    assert cfg.training.pair_iteration_mode == "resample_from_saved_sequences"
    assert cfg.training.preload_to_gpu is True


def test_load_config_rejects_saved_interval_and_snapshot_count_together(tmp_path: Path) -> None:
    """Users must choose either an interval cadence or a snapshot count."""
    payload = _minimal_config_dict()
    sampling_section = dict(payload["sampling"])
    sampling_section["saved_snapshots_per_sim"] = 9
    payload["sampling"] = sampling_section

    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(config_path)


def test_load_config_allows_preload_to_gpu_for_resampled_mode(tmp_path: Path) -> None:
    """Resampled mode may preload full sequence splits when VRAM is sufficient."""
    payload = _minimal_config_dict()
    training_section = dict(payload["training"])
    training_section["pair_iteration_mode"] = "resample_from_saved_sequences"
    training_section["preload_to_gpu"] = True
    payload["training"] = training_section

    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.training.pair_iteration_mode == "resample_from_saved_sequences"
    assert cfg.training.preload_to_gpu is True
