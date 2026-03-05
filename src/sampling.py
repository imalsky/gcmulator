"""Sampling utilities for user-configured physical parameters."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .config import Extended9Params, ParameterSpec

# Sampling and unit-conversion constants.
PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0
SECONDS_PER_HOUR = 3600.0


CONDITIONING_PARAM_NAMES: List[str] = [
    "a_m",
    "omega_rad_s",
    "Phibar",
    "DPhieq",
    "taurad_s",
    "taudrag_s",
    "g_m_s2",
]
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
            raise ValueError(f"mixture_off_loguniform requires p_off/off_value/on_min/on_max for {spec.name}")
        if float(spec.on_min) <= 0 or float(spec.on_max) <= 0:
            raise ValueError(f"mixture_off_loguniform requires positive on_min/on_max for {spec.name}")
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


def sample_parameter_dict(rng: np.random.Generator, specs: Sequence[ParameterSpec]) -> Dict[str, float]:
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
    """Convert sampled parameter map into validated ``Extended9Params``."""
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
