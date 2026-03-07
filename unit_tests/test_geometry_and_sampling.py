"""Geometry and sampling helper tests."""

from __future__ import annotations

import numpy as np

from config import ParameterSpec
from geometry import apply_geometry_state, geometry_shift_for_nlon
from sampling import sample_parameter_dict, sample_transition_pairs, to_extended9


def test_apply_geometry_state_flips_latitude_and_rolls_longitude() -> None:
    """The canonical storage geometry should be deterministic."""
    state = np.arange(8, dtype=np.float64).reshape(1, 2, 4)

    transformed, info = apply_geometry_state(
        state,
        flip_latitude_to_north_south=True,
        roll_longitude_to_0_2pi=True,
    )

    expected = np.roll(state[:, ::-1, :], shift=geometry_shift_for_nlon(4, True), axis=-1)
    assert np.array_equal(transformed, expected)
    assert info["lat_order"] == "north_to_south"
    assert info["lon_origin"] == "0_to_2pi"
    assert info["lon_shift"] == -2


def test_sample_parameter_dict_converts_hour_aliases_to_seconds() -> None:
    """Hour-valued aliases should be converted into canonical second units."""
    rng = np.random.default_rng(0)
    sampled = sample_parameter_dict(
        rng,
        [
            ParameterSpec(name="a_m", dist="fixed", value=8.2e7),
            ParameterSpec(name="omega_rad_s", dist="fixed", value=3.2e-5),
            ParameterSpec(name="Phibar", dist="fixed", value=3.0e5),
            ParameterSpec(name="DPhieq", dist="fixed", value=1.0e6),
            ParameterSpec(name="taurad_hours", dist="fixed", value=10.0),
            ParameterSpec(name="taudrag_hours", dist="fixed", value=6.0),
            ParameterSpec(name="g_m_s2", dist="fixed", value=9.8),
        ],
    )

    params = to_extended9(sampled)

    assert sampled["taurad_s"] == 10.0 * 3600.0
    assert sampled["taudrag_s"] == 6.0 * 3600.0
    assert params.taurad_s == 10.0 * 3600.0
    assert params.taudrag_s == 6.0 * 3600.0


def test_sample_transition_pairs_respects_bounds_and_integer_step_mapping() -> None:
    """Sampled transition pairs must stay within the valid rollout horizon."""
    rng = np.random.default_rng(0)
    anchor_steps, target_steps, transition_days = sample_transition_pairs(
        rng,
        n_transitions=64,
        burn_in_steps=0,
        n_steps_total=8,
        dt_seconds=240.0,
        transition_jump_days_min=240.0 / 86400.0,
        transition_jump_days_max=3.0 * 240.0 / 86400.0,
    )
    jump_steps = np.rint(transition_days * 86400.0 / 240.0).astype(np.int64)

    assert anchor_steps.dtype == np.int64
    assert target_steps.dtype == np.int64
    assert transition_days.dtype == np.float64
    assert np.all(anchor_steps[:-1] <= anchor_steps[1:])
    assert np.all(target_steps > anchor_steps)
    assert np.array_equal(target_steps - anchor_steps, jump_steps)
    assert np.all(jump_steps >= 1)
    assert np.all(jump_steps <= 3)
    assert np.all(target_steps <= 8)
