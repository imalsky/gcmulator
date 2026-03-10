"""Geometry and sampling helper tests."""

from __future__ import annotations

import numpy as np

from gcmulator.config import ParameterSpec
from gcmulator.geometry import apply_geometry_state, geometry_shift_for_nlon
from gcmulator.sampling import (
    build_live_transition_catalog,
    build_uniform_checkpoint_schedule,
    sample_parameter_dict,
    to_extended9,
)


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


def test_uniform_checkpoint_schedule_maps_to_solver_grid() -> None:
    """Saved checkpoints should land on a uniform discrete solver cadence."""
    step_days = 240.0 / 86400.0
    schedule = build_uniform_checkpoint_schedule(
        time_days=0.05,
        dt_seconds=240.0,
        saved_checkpoint_interval_days=2.0 * step_days,
    )

    assert schedule.interval_steps == 2
    assert schedule.interval_days == np.float64(2.0 * step_days)
    assert np.array_equal(schedule.checkpoint_steps, np.arange(0, 19, 2, dtype=np.int64))
    assert np.allclose(
        schedule.checkpoint_days,
        schedule.checkpoint_steps.astype(np.float64) * step_days,
    )


def test_fixed_live_transition_catalog_picks_exact_gap() -> None:
    """A fixed live jump should resolve to the nearest feasible checkpoint gap."""
    step_days = 240.0 / 86400.0
    schedule = build_uniform_checkpoint_schedule(
        time_days=0.05,
        dt_seconds=240.0,
        saved_checkpoint_interval_days=2.0 * step_days,
    )
    catalog = build_live_transition_catalog(
        checkpoint_days=schedule.checkpoint_days,
        burn_in_days=0.0,
        transition_days_min=4.0 * step_days,
        transition_days_max=4.0 * step_days,
        tolerance_fraction=0.0,
    )

    assert np.array_equal(catalog.gap_offsets, np.array([2], dtype=np.int64))
    assert np.allclose(catalog.transition_days, np.array([4.0 * step_days], dtype=np.float64))
    assert np.allclose(catalog.probabilities, np.array([1.0], dtype=np.float64))
    assert catalog.burn_in_start_index == 0


def test_variable_live_transition_catalog_stays_within_saved_gaps() -> None:
    """Variable live jumps should only use feasible saved checkpoint gaps."""
    step_days = 240.0 / 86400.0
    schedule = build_uniform_checkpoint_schedule(
        time_days=0.05,
        dt_seconds=240.0,
        saved_checkpoint_interval_days=2.0 * step_days,
    )
    catalog = build_live_transition_catalog(
        checkpoint_days=schedule.checkpoint_days,
        burn_in_days=2.0 * step_days,
        transition_days_min=4.0 * step_days,
        transition_days_max=10.0 * step_days,
        tolerance_fraction=0.1,
    )

    assert np.array_equal(catalog.gap_offsets, np.array([2, 3, 4, 5], dtype=np.int64))
    assert np.all(catalog.transition_days >= 4.0 * step_days)
    assert np.all(catalog.transition_days <= 10.0 * step_days)
    assert np.isclose(np.sum(catalog.probabilities), 1.0)
    assert catalog.burn_in_start_index == 1
    assert catalog.probabilities[0] > catalog.probabilities[-1]
