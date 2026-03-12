"""Sampling utilities for user-configured physical parameters and live jumps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Sequence

import numpy as np

from .config import (
    CONDITIONING_PARAM_NAMES,
    Extended9Params,
    ParameterSpec,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
    SECONDS_PER_DAY,
)

# Sampling and unit-conversion constants.
SECONDS_PER_HOUR = 3600.0
TRANSITION_STD_FLOOR = 1.0e-12

# Hyperdiffusion controls are fixed internally and intentionally excluded from
# the user-facing conditioning vector.  K6 is the 6th-order (nabla^6)
# hyperdiffusion coefficient that damps small-scale spectral ringing in the
# barotropic shallow-water solver.  The value 1.24e33 was tuned for the M=42
# Legendre-Gauss grid to give stable integrations across the full parameter
# range without visibly damping resolved dynamics.  K6Phi=None disables
# separate geopotential diffusion, reusing the same K6 for all fields.
INTERNAL_FIXED_K6 = 1.24e33
INTERNAL_FIXED_K6PHI = None


@dataclass(frozen=True)
class CheckpointSchedule:
    """Uniformly spaced checkpoint schedule for one simulated trajectory."""

    checkpoint_steps: np.ndarray
    checkpoint_days: np.ndarray
    interval_steps: int
    interval_days: float


@dataclass(frozen=True)
class LiveTransitionCatalog:
    """Discrete transition-day catalog used for live GPU pair sampling."""

    gap_offsets: np.ndarray
    transition_days: np.ndarray
    probabilities: np.ndarray
    burn_in_start_index: int


def _sample_one(rng: np.random.Generator, spec: ParameterSpec) -> float:
    """Draw one scalar sample according to one ``ParameterSpec`` distribution."""
    if spec.dist == "uniform":
        if spec.min is None or spec.max is None:
            raise ValueError(f"uniform requires min/max for {spec.name}")
        return float(rng.uniform(spec.min, spec.max))

    if spec.dist == "loguniform":
        if spec.min is None or spec.max is None or spec.min <= 0 or spec.max <= 0:
            raise ValueError(f"loguniform requires positive min/max for {spec.name}")
        lo = math.log10(float(spec.min))
        hi = math.log10(float(spec.max))
        return float(10.0 ** rng.uniform(lo, hi))

    if spec.dist in {"const", "fixed"}:
        if spec.value is None:
            raise ValueError(f"fixed/const requires value for {spec.name}")
        return float(spec.value)

    if spec.dist == "mixture_off_loguniform":
        if spec.p_off is None or spec.on_min is None or spec.on_max is None:
            raise ValueError(
                "mixture_off_loguniform requires p_off/off_value/on_min/on_max "
                f"for {spec.name}"
            )
        if float(spec.on_min) <= 0 or float(spec.on_max) <= 0:
            raise ValueError(
                "mixture_off_loguniform requires positive "
                f"on_min/on_max for {spec.name}"
            )
        if float(spec.on_min) >= float(spec.on_max):
            raise ValueError(f"mixture_off_loguniform requires on_min < on_max for {spec.name}")
        if not (PROBABILITY_MIN <= float(spec.p_off) <= PROBABILITY_MAX):
            raise ValueError(f"p_off must be in [0,1] for {spec.name}")
        if float(rng.uniform(PROBABILITY_MIN, PROBABILITY_MAX)) < float(spec.p_off):
            if spec.off_value is None:
                raise ValueError(f"mixture_off_loguniform requires off_value for {spec.name}")
            return float(spec.off_value)
        lo = math.log10(float(spec.on_min))
        hi = math.log10(float(spec.on_max))
        return float(10.0 ** rng.uniform(lo, hi))

    raise ValueError(f"Unsupported dist '{spec.dist}' for {spec.name}")


def sample_parameter_dict(
    rng: np.random.Generator,
    specs: Sequence[ParameterSpec],
) -> Dict[str, float]:
    """Sample all configured parameters and normalize alias units to seconds."""
    sampled: Dict[str, float] = {}
    for spec in specs:
        sampled[spec.name] = _sample_one(rng, spec)

    # Canonical conversion from retrieval-friendly hour parameters to seconds.
    if "taurad_hours" in sampled and "taurad_s" not in sampled:
        sampled["taurad_s"] = SECONDS_PER_HOUR * float(sampled["taurad_hours"])
    if "taudrag_hours" in sampled and "taudrag_s" not in sampled:
        sampled["taudrag_s"] = SECONDS_PER_HOUR * float(sampled["taudrag_hours"])

    return sampled


def to_extended9(sampled: Dict[str, float]) -> Extended9Params:
    """Convert a sampled parameter map into validated ``Extended9Params``."""
    missing = [name for name in CONDITIONING_PARAM_NAMES if name not in sampled]
    if missing:
        raise ValueError(f"Missing sampled conditioning parameters: {missing}")

    return Extended9Params(
        a_m=float(sampled["a_m"]),
        omega_rad_s=float(sampled["omega_rad_s"]),
        Phibar=float(sampled["Phibar"]),
        DPhieq=float(sampled["DPhieq"]),
        taurad_s=float(sampled["taurad_s"]),
        taudrag_s=float(sampled["taudrag_s"]),
        g_m_s2=float(sampled["g_m_s2"]),
        K6=float(INTERNAL_FIXED_K6),
        K6Phi=INTERNAL_FIXED_K6PHI,
    )


def build_uniform_checkpoint_schedule(
    *,
    time_days: float,
    dt_seconds: float,
    saved_checkpoint_interval_days: float | None = None,
    saved_snapshots_per_sim: int | None = None,
) -> CheckpointSchedule:
    """Resolve a uniform checkpoint cadence onto the discrete solver grid."""
    if time_days <= 0:
        raise ValueError("time_days must be > 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")
    if (saved_checkpoint_interval_days is None) == (saved_snapshots_per_sim is None):
        raise ValueError(
            "Provide exactly one of saved_checkpoint_interval_days or saved_snapshots_per_sim"
        )

    n_steps_total = max(
        1,
        int(round(float(time_days) * SECONDS_PER_DAY / float(dt_seconds))),
    )
    if saved_snapshots_per_sim is not None:
        if saved_snapshots_per_sim < 1:
            raise ValueError("saved_snapshots_per_sim must be >= 1")
        if int(saved_snapshots_per_sim) > int(n_steps_total):
            raise ValueError(
                "saved_snapshots_per_sim must be <= the total number of solver steps"
            )
        if int(n_steps_total) % int(saved_snapshots_per_sim) != 0:
            raise ValueError(
                "saved_snapshots_per_sim must divide the total number of solver steps "
                f"exactly; total_steps={n_steps_total}, "
                f"saved_snapshots_per_sim={saved_snapshots_per_sim}"
            )
        interval_steps = max(1, int(n_steps_total) // int(saved_snapshots_per_sim))
    else:
        if saved_checkpoint_interval_days is None or saved_checkpoint_interval_days <= 0:
            raise ValueError("saved_checkpoint_interval_days must be > 0")
        interval_steps = max(
            1,
            int(round(float(saved_checkpoint_interval_days) * SECONDS_PER_DAY / float(dt_seconds))),
        )
    checkpoint_steps = np.arange(0, n_steps_total + 1, interval_steps, dtype=np.int64)
    checkpoint_days = checkpoint_steps.astype(np.float64) * float(dt_seconds) / SECONDS_PER_DAY
    return CheckpointSchedule(
        checkpoint_steps=checkpoint_steps,
        checkpoint_days=checkpoint_days,
        interval_steps=int(interval_steps),
        interval_days=float(interval_steps) * float(dt_seconds) / SECONDS_PER_DAY,
    )


def build_live_transition_catalog(
    *,
    checkpoint_days: np.ndarray,
    burn_in_days: float,
    transition_days_min: float,
    transition_days_max: float,
    tolerance_fraction: float,
    pair_sampling_policy: str = "inverse_time",
) -> LiveTransitionCatalog:
    """Build the feasible discrete jump catalog induced by the saved cadence."""
    checkpoint_days = np.asarray(checkpoint_days, dtype=np.float64)
    if checkpoint_days.ndim != 1:
        raise ValueError("checkpoint_days must be rank-1")
    if checkpoint_days.shape[0] < 2:
        raise ValueError("At least two checkpoints are required for live pair sampling")
    if np.any(np.diff(checkpoint_days) <= 0.0):
        raise ValueError("checkpoint_days must be strictly increasing")
    if burn_in_days < 0:
        raise ValueError("burn_in_days must be >= 0")
    if transition_days_min <= 0 or transition_days_max <= 0:
        raise ValueError("transition_days_min/max must be > 0")
    if transition_days_min > transition_days_max:
        raise ValueError("transition_days_min must be <= transition_days_max")
    if tolerance_fraction < 0.0 or tolerance_fraction > 1.0:
        raise ValueError("tolerance_fraction must be in [0,1]")
    if pair_sampling_policy not in {"uniform_pairs", "uniform_gaps", "inverse_time"}:
        raise ValueError(
            "pair_sampling_policy must be one of "
            "['uniform_pairs','uniform_gaps','inverse_time']"
        )

    burn_in_start_index = int(np.searchsorted(checkpoint_days, float(burn_in_days), side="left"))
    max_gap_offset = int(checkpoint_days.shape[0] - 1 - burn_in_start_index)
    if max_gap_offset < 1:
        raise ValueError("No valid live transition gaps remain after burn-in")

    candidate_offsets = np.arange(1, max_gap_offset + 1, dtype=np.int64)
    candidate_days = checkpoint_days[candidate_offsets] - checkpoint_days[0]

    if math.isclose(float(transition_days_min), float(transition_days_max), rel_tol=0.0, abs_tol=0.0):
        target_days = float(transition_days_min)
        nearest_index = int(np.argmin(np.abs(candidate_days - target_days)))
        nearest_days = float(candidate_days[nearest_index])
        relative_error = abs(nearest_days - target_days) / target_days
        if relative_error > float(tolerance_fraction):
            raise ValueError(
                "The requested fixed transition day is not representable within tolerance: "
                f"requested={target_days:.6f}, realized={nearest_days:.6f}, "
                f"relative_error={relative_error:.6f}, "
                f"tolerance_fraction={float(tolerance_fraction):.6f}"
            )
        return LiveTransitionCatalog(
            gap_offsets=np.asarray([candidate_offsets[nearest_index]], dtype=np.int64),
            transition_days=np.asarray([nearest_days], dtype=np.float64),
            probabilities=np.asarray([1.0], dtype=np.float64),
            burn_in_start_index=int(burn_in_start_index),
        )

    mask = (candidate_days >= float(transition_days_min)) & (
        candidate_days <= float(transition_days_max)
    )
    if not np.any(mask):
        raise ValueError(
            "No feasible discrete transition days fall within the requested live range"
        )
    transition_days = candidate_days[mask].astype(np.float64)
    if pair_sampling_policy == "inverse_time":
        weights = (1.0 / np.maximum(transition_days, 1.0e-30)).astype(np.float64)
    else:
        weights = np.ones_like(transition_days, dtype=np.float64)
    weights /= np.sum(weights)
    return LiveTransitionCatalog(
        gap_offsets=candidate_offsets[mask].astype(np.int64),
        transition_days=transition_days,
        probabilities=weights,
        burn_in_start_index=int(burn_in_start_index),
    )


def valid_anchor_counts_for_catalog(
    *,
    sequence_length: int,
    catalog: LiveTransitionCatalog,
) -> np.ndarray:
    """Return the number of valid anchors for each catalog gap offset."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    max_anchor = int(sequence_length) - 1 - catalog.gap_offsets.astype(np.int64)
    counts = max_anchor - int(catalog.burn_in_start_index) + 1
    if np.any(counts <= 0):
        raise ValueError("Catalog produced one or more invalid anchor counts")
    return counts.astype(np.int64)


def weighted_log10_transition_stats(
    catalog: LiveTransitionCatalog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted log10 transition-time mean/std for normalization."""
    log10_days = np.log10(np.maximum(catalog.transition_days.astype(np.float64), 1.0e-30))
    weights = catalog.probabilities.astype(np.float64)
    mean = np.sum(weights * log10_days)
    var = max(0.0, float(np.sum(weights * (log10_days - mean) ** 2)))
    std_raw = math.sqrt(var)
    is_constant = std_raw <= TRANSITION_STD_FLOOR
    std = 1.0 if is_constant else max(std_raw, TRANSITION_STD_FLOOR)
    return (
        np.asarray([mean], dtype=np.float64),
        np.asarray([std], dtype=np.float64),
        np.asarray([is_constant], dtype=bool),
    )
