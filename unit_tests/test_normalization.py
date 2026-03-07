"""Normalization regression tests."""

from __future__ import annotations

import numpy as np

from config import TRANSITION_TIME_NAME
from normalization import (
    ParamNormalizationStats,
    StateNormalizationStats,
    denormalize_state_tensor,
    normalize_conditioning,
    normalize_params,
    normalize_state_tensor,
    stats_from_json,
)


def test_state_normalization_round_trip() -> None:
    """State transforms and inverse transforms should round-trip cleanly."""
    state = np.array(
        [
            [
                [[1.0, 10.0], [100.0, 1000.0]],
                [[-2.0, -1.0], [1.0, 2.0]],
            ]
        ],
        dtype=np.float32,
    )
    stats = StateNormalizationStats(
        field_names=("Phi", "eta"),
        field_transforms={"Phi": "log10", "eta": "signed_log1p"},
        mean=np.array([1.5, 0.25], dtype=np.float64),
        std=np.array([0.75, 1.25], dtype=np.float64),
        zscore_eps=1.0e-8,
        log10_eps=1.0e-6,
        signed_log1p_scale=2.0,
    )

    normalized = normalize_state_tensor(state, stats)
    restored = denormalize_state_tensor(normalized, stats)

    assert np.allclose(restored, state, atol=1.0e-5)


def test_normalize_params_zeros_constant_channels() -> None:
    """Constant conditioning channels should normalize to zero."""
    params = np.array([[3.0, 9.0], [7.0, 9.0]], dtype=np.float64)
    stats = ParamNormalizationStats(
        param_names=("varying", "constant"),
        mean=np.array([5.0, 9.0], dtype=np.float64),
        std=np.array([2.0, 1.0], dtype=np.float64),
        is_constant=np.array([False, True], dtype=bool),
        zscore_eps=1.0e-8,
    )

    normalized = normalize_params(params, stats)

    assert np.allclose(normalized[:, 1], 0.0)
    assert np.allclose(normalized[:, 0], np.array([-1.0, 1.0], dtype=np.float32))


def test_normalize_conditioning_appends_transition_time_channel() -> None:
    """Time-conditioned inputs should append normalized transition duration."""
    params = np.array([3.0, 9.0], dtype=np.float64)
    transition_days = np.array([0.25, 0.5], dtype=np.float64)
    log10_transition_days = np.log10(transition_days)
    param_stats = ParamNormalizationStats(
        param_names=("varying", "constant"),
        mean=np.array([5.0, 9.0], dtype=np.float64),
        std=np.array([2.0, 1.0], dtype=np.float64),
        is_constant=np.array([False, True], dtype=bool),
        zscore_eps=1.0e-8,
    )
    transition_time_stats = ParamNormalizationStats(
        param_names=(TRANSITION_TIME_NAME,),
        mean=np.array([np.mean(log10_transition_days)], dtype=np.float64),
        std=np.array([np.std(log10_transition_days)], dtype=np.float64),
        is_constant=np.array([False], dtype=bool),
        zscore_eps=1.0e-8,
    )

    conditioning = normalize_conditioning(
        params,
        transition_days,
        param_stats=param_stats,
        transition_time_stats=transition_time_stats,
    )

    assert conditioning.shape == (2, 3)
    assert np.allclose(conditioning[:, 0], np.array([-1.0, -1.0], dtype=np.float32))
    assert np.allclose(conditioning[:, 1], 0.0)
    assert np.allclose(conditioning[:, 2], np.array([-1.0, 1.0], dtype=np.float32))


def test_stats_from_json_requires_transition_time_block() -> None:
    """Strict loading should reject stale normalization payloads."""
    payload = {
        "input_state": {
            "field_names": ["Phi"],
            "field_transforms": {"Phi": "none"},
            "mean": [0.0],
            "std": [1.0],
            "zscore_eps": 1.0e-8,
            "log10_eps": 1.0e-6,
            "signed_log1p_scale": 1.0,
        },
        "target_state": {
            "field_names": ["Phi"],
            "field_transforms": {"Phi": "none"},
            "mean": [0.0],
            "std": [1.0],
            "zscore_eps": 1.0e-8,
            "log10_eps": 1.0e-6,
            "signed_log1p_scale": 1.0,
        },
        "params": {
            "param_names": ["a_m"],
            "mean": [0.0],
            "std": [1.0],
            "is_constant": [False],
            "zscore_eps": 1.0e-8,
        },
    }

    try:
        stats_from_json(payload)
    except ValueError as exc:
        assert "transition_time" in str(exc)
    else:
        raise AssertionError("stats_from_json should reject missing transition_time")
