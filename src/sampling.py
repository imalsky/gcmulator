"""Sampling utilities for user-configured physical parameters.

This module owns the translation between config-level sampling specifications
and the canonical conditioning vector used throughout the emulator pipeline.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

from config import (
    CONDITIONING_PARAM_NAMES,
    Extended9Params,
    ParameterSpec,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
    SECONDS_PER_DAY,
)

# Sampling and unit-conversion constants.
SECONDS_PER_HOUR = 3600.0

# Diffusion controls are fixed internally and intentionally excluded from the
# user-facing conditioning vector.
INTERNAL_FIXED_K6 = 1.24e33
INTERNAL_FIXED_K6PHI = None


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


def vector_to_extended9(vector: np.ndarray) -> Extended9Params:
    """Convert a canonical conditioning vector back into ``Extended9Params``."""
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (len(CONDITIONING_PARAM_NAMES),):
        raise ValueError(
            "Conditioning vector must have shape "
            f"({len(CONDITIONING_PARAM_NAMES)},), got {values.shape}"
        )
    sampled = {name: float(values[index]) for index, name in enumerate(CONDITIONING_PARAM_NAMES)}
    return to_extended9(sampled)


def sample_transition_pairs(
    rng: np.random.Generator,
    *,
    n_transitions: int,
    burn_in_steps: int,
    n_steps_total: int,
    dt_seconds: float,
    transition_jump_days_min: float,
    transition_jump_days_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random (anchor, target) pairs with log-uniform time jumps.

    Each transition independently draws a random anchor time from the
    post-burn-in portion of the trajectory and a random time jump from a
    log-uniform distribution.  The jump is clamped to fit within the
    available simulation horizon.  This means early anchor times can reach
    large dt (including near-equilibrium states), while late anchor times
    are naturally limited to shorter jumps.

    Returns:
        anchor_steps: (n_transitions,) int64 — solver step indices of inputs
        target_steps: (n_transitions,) int64 — solver step indices of targets
        transition_days: (n_transitions,) float64 — actual jump durations
    """
    if n_transitions < 1:
        raise ValueError("n_transitions must be >= 1")
    if n_steps_total < 1:
        raise ValueError("n_steps_total must be >= 1")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")
    if transition_jump_days_min <= 0:
        raise ValueError("transition_jump_days_min must be > 0")
    if transition_jump_days_min > transition_jump_days_max:
        raise ValueError("transition_jump_days_min must be <= transition_jump_days_max")

    min_jump_steps = max(1, int(math.ceil(
        transition_jump_days_min * SECONDS_PER_DAY / dt_seconds
    )))

    # Anchors can start anywhere from burn-in up to the point where at
    # least the minimum jump still fits within the trajectory.
    max_anchor = n_steps_total - min_jump_steps
    if max_anchor < burn_in_steps:
        raise ValueError(
            "No valid anchor positions exist: "
            f"burn_in_steps={burn_in_steps}, "
            f"n_steps_total={n_steps_total}, "
            f"min_jump_steps={min_jump_steps}"
        )

    log_min = math.log10(transition_jump_days_min)
    log_max = math.log10(transition_jump_days_max)

    anchor_steps = np.empty(n_transitions, dtype=np.int64)
    target_steps = np.empty(n_transitions, dtype=np.int64)
    transition_days = np.empty(n_transitions, dtype=np.float64)

    for i in range(n_transitions):
        anchor = int(rng.integers(burn_in_steps, max_anchor + 1))

        # Clamp the upper bound so the target stays within the trajectory.
        remaining_steps = n_steps_total - anchor
        remaining_days = float(remaining_steps) * dt_seconds / SECONDS_PER_DAY
        effective_log_max = min(log_max, math.log10(max(remaining_days, transition_jump_days_min)))

        # Sample log-uniform jump, clamp to at least one solver step.
        dt_days = float(10.0 ** rng.uniform(log_min, max(log_min, effective_log_max)))
        dt_steps = max(1, int(round(dt_days * SECONDS_PER_DAY / dt_seconds)))
        dt_steps = min(dt_steps, remaining_steps)

        target = anchor + dt_steps
        actual_days = float(dt_steps) * dt_seconds / SECONDS_PER_DAY

        anchor_steps[i] = anchor
        target_steps[i] = target
        transition_days[i] = actual_days

    # Sort by anchor time for deterministic ordering.
    order = np.argsort(anchor_steps)
    return anchor_steps[order], target_steps[order], transition_days[order]
