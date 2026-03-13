"""Geometry and sampling helper tests."""

from __future__ import annotations

import numpy as np

from gcmulator.config import ParameterSpec
from gcmulator.geometry import apply_geometry_state, geometry_shift_for_nlon
from gcmulator.sampling import (
    build_live_transition_catalog,
    build_uniform_checkpoint_schedule,
    checkpoint_schedule_kwargs,
    sample_parameter_dict,
    to_extended9,
    valid_anchor_counts_for_catalog,
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


def test_uniform_checkpoint_schedule_supports_snapshot_count() -> None:
    """Snapshot-count scheduling should derive the exact saved cadence."""
    schedule = build_uniform_checkpoint_schedule(
        time_days=100.0,
        dt_seconds=120.0,
        saved_snapshots_per_sim=1000,
    )

    assert schedule.interval_steps == 72
    assert schedule.interval_days == np.float64(0.1)
    assert schedule.checkpoint_steps.shape == (1001,)
    assert schedule.checkpoint_days.shape == (1001,)
    assert schedule.checkpoint_days[0] == np.float64(0.0)
    assert schedule.checkpoint_days[-1] == np.float64(100.0)


def test_checkpoint_schedule_kwargs_prefer_snapshot_count_when_present() -> None:
    """Parsed configs with derived intervals should still pass one schedule knob."""
    kwargs = checkpoint_schedule_kwargs(
        saved_checkpoint_interval_days=0.1,
        saved_snapshots_per_sim=1000,
    )
    schedule = build_uniform_checkpoint_schedule(
        time_days=100.0,
        dt_seconds=120.0,
        **kwargs,
    )

    assert kwargs == {"saved_snapshots_per_sim": 1000}
    assert schedule.interval_days == np.float64(0.1)
    assert schedule.checkpoint_days.shape == (1001,)


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


def test_fixed_72_step_transition_catalog_picks_one_saved_gap() -> None:
    """A config-resolved 72-step jump should map to one exact saved checkpoint gap."""
    fixed_transition_days = 72.0 * 120.0 / 86400.0
    schedule = build_uniform_checkpoint_schedule(
        time_days=100.0,
        dt_seconds=120.0,
        saved_snapshots_per_sim=1000,
    )
    catalog = build_live_transition_catalog(
        checkpoint_days=schedule.checkpoint_days,
        burn_in_days=0.0,
        transition_days_min=fixed_transition_days,
        transition_days_max=fixed_transition_days,
        tolerance_fraction=0.0,
    )

    assert np.array_equal(catalog.gap_offsets, np.array([1], dtype=np.int64))
    assert np.allclose(
        catalog.transition_days,
        np.array([fixed_transition_days], dtype=np.float64),
    )
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


def test_uniform_gap_policy_uses_flat_catalog_weights() -> None:
    """Uniform-gap mode should not retain the inverse-time short-gap bias."""
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
        pair_sampling_policy="uniform_gaps",
    )

    assert np.allclose(catalog.probabilities, np.full((4,), 0.25, dtype=np.float64))


def test_valid_anchor_counts_follow_burn_in_and_gap_offsets() -> None:
    """Per-gap anchor counts should respect both burn-in and target reachability."""
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

    counts = valid_anchor_counts_for_catalog(
        sequence_length=int(schedule.checkpoint_days.shape[0]),
        catalog=catalog,
    )

    assert np.array_equal(counts, np.array([7, 6, 5, 4], dtype=np.int64))
