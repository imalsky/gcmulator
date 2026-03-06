"""Normalization helpers for states and conditioning parameters.

The emulator normalizes full visible states and prognostic targets slightly
differently, so this module keeps the shared bookkeeping in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from config import NormalizationConfig, TRANSITION_TIME_NAME


STD_FLOOR = 1.0e-12


@dataclass(frozen=True)
class StateNormalizationStats:
    """Normalization statistics for one state tensor contract."""

    field_names: Tuple[str, ...]
    field_transforms: Dict[str, str]
    mean: np.ndarray
    std: np.ndarray
    zscore_eps: float
    log10_eps: float
    signed_log1p_scale: float


@dataclass(frozen=True)
class ParamNormalizationStats:
    """Normalization statistics for the conditioning parameter vector."""

    param_names: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    is_constant: np.ndarray
    zscore_eps: float


@dataclass(frozen=True)
class NormalizationStats:
    """Complete normalization bundle stored in metadata and checkpoints."""

    input_state: StateNormalizationStats
    target_state: StateNormalizationStats
    params: ParamNormalizationStats
    transition_time: ParamNormalizationStats


def subset_state_stats(
    stats: StateNormalizationStats,
    field_names: Sequence[str],
) -> StateNormalizationStats:
    """Return state normalization statistics restricted to selected fields."""
    indices = [stats.field_names.index(str(field_name)) for field_name in field_names]
    return StateNormalizationStats(
        field_names=tuple(str(field_name) for field_name in field_names),
        field_transforms={
            str(name): str(stats.field_transforms.get(str(name), "none"))
            for name in field_names
        },
        mean=np.asarray(stats.mean, dtype=np.float64)[indices],
        std=np.asarray(stats.std, dtype=np.float64)[indices],
        zscore_eps=float(stats.zscore_eps),
        log10_eps=float(stats.log10_eps),
        signed_log1p_scale=float(stats.signed_log1p_scale),
    )


def _transform_channel(
    x: np.ndarray,
    mode: str,
    *,
    log10_eps: float,
    signed_log1p_scale: float,
) -> np.ndarray:
    """Apply one configured forward transform to one state channel."""
    if mode == "none":
        return x
    if mode == "log10":
        finite = np.isfinite(x)
        if not np.all(finite):
            raise ValueError("log10 transform received non-finite values")
        if np.any(x <= 0.0):
            raise ValueError("log10 transform requires strictly positive values")
        return np.log10(np.maximum(x, log10_eps))
    if mode == "signed_log1p":
        scale = float(signed_log1p_scale)
        if scale <= 0:
            raise ValueError("signed_log1p_scale must be > 0")
        return np.sign(x) * np.log1p(np.abs(x) / scale)
    raise ValueError(f"Unsupported normalization transform mode: {mode}")


def _inverse_transform_channel(
    x: np.ndarray,
    mode: str,
    *,
    signed_log1p_scale: float,
) -> np.ndarray:
    """Invert one state-channel transform."""
    if mode == "none":
        return x
    if mode == "log10":
        return np.power(10.0, x)
    if mode == "signed_log1p":
        scale = float(signed_log1p_scale)
        if scale <= 0:
            raise ValueError("signed_log1p_scale must be > 0")
        return np.sign(x) * (np.expm1(np.abs(x)) * scale)
    raise ValueError(f"Unsupported normalization transform mode: {mode}")


def apply_state_transforms(
    states_nchw: np.ndarray,
    field_names: Sequence[str],
    cfg: NormalizationConfig,
) -> np.ndarray:
    """Apply per-field configured transforms to a ``[N,C,H,W]`` state tensor."""
    if states_nchw.ndim != 4:
        raise ValueError(f"states_nchw must be [N,C,H,W], got {states_nchw.shape}")
    out = states_nchw.astype(np.float64, copy=True)
    for channel_index, field_name in enumerate(field_names):
        mode = str(cfg.field_transforms.get(str(field_name), "none"))
        out[:, channel_index] = _transform_channel(
            out[:, channel_index],
            mode,
            log10_eps=float(cfg.log10_eps),
            signed_log1p_scale=float(cfg.signed_log1p_scale),
        )
    return out


def normalize_state_tensor(states_nchw: np.ndarray, stats: StateNormalizationStats) -> np.ndarray:
    """Transform and z-score normalize state tensors."""
    cfg = NormalizationConfig(
        field_transforms={str(k): str(v) for k, v in stats.field_transforms.items()},
        zscore_eps=float(stats.zscore_eps),
        log10_eps=float(stats.log10_eps),
        signed_log1p_scale=float(stats.signed_log1p_scale),
    )
    transformed = apply_state_transforms(states_nchw, stats.field_names, cfg)
    normalized = (
        transformed - stats.mean.reshape(1, -1, 1, 1)
    ) / (stats.std.reshape(1, -1, 1, 1) + stats.zscore_eps)
    return normalized.astype(np.float32)


def denormalize_state_tensor(
    states_norm_nchw: np.ndarray,
    stats: StateNormalizationStats,
) -> np.ndarray:
    """Invert z-score and field transforms for normalized state tensors."""
    states = states_norm_nchw.astype(np.float64, copy=False)
    restored = (
        states * (stats.std.reshape(1, -1, 1, 1) + stats.zscore_eps)
        + stats.mean.reshape(1, -1, 1, 1)
    )
    physical = restored.astype(np.float64, copy=True)
    for channel_index, field_name in enumerate(stats.field_names):
        mode = str(stats.field_transforms.get(str(field_name), "none"))
        physical[:, channel_index] = _inverse_transform_channel(
            physical[:, channel_index],
            mode,
            signed_log1p_scale=float(stats.signed_log1p_scale),
        )
    return physical.astype(np.float32)


def normalize_params(params_np: np.ndarray, stats: ParamNormalizationStats) -> np.ndarray:
    """Z-score normalize parameter vectors and zero constant channels."""
    # Keep the accumulation math in float64, then cast to float32 to match the
    # tensors stored in processed shards and checkpoints.
    out = (params_np.astype(np.float64, copy=False) - stats.mean.reshape(1, -1)) / (
        stats.std.reshape(1, -1) + stats.zscore_eps
    )
    const_mask = np.asarray(stats.is_constant, dtype=bool)
    if const_mask.shape != (out.shape[1],):
        raise ValueError("ParamNormalizationStats.is_constant shape mismatch")
    if np.any(const_mask):
        out[:, const_mask] = 0.0
    return out.astype(np.float32)


def normalize_conditioning(
    params_np: np.ndarray,
    transition_days_np: np.ndarray,
    *,
    param_stats: ParamNormalizationStats,
    transition_time_stats: ParamNormalizationStats,
) -> np.ndarray:
    """Normalize physical parameters and transition duration into one conditioning matrix."""
    params = np.asarray(params_np, dtype=np.float64)
    if params.ndim == 1:
        params = params[None, :]
    if params.ndim != 2:
        raise ValueError(f"params_np must be rank-1 or rank-2, got {params.shape}")

    transition_days = np.asarray(transition_days_np, dtype=np.float64)
    if transition_days.ndim == 0:
        transition_days = transition_days.reshape(1)
    elif transition_days.ndim != 1:
        raise ValueError(
            "transition_days_np must be rank-0 or rank-1, "
            f"got {transition_days.shape}"
        )

    if params.shape[0] == 1 and transition_days.shape[0] > 1:
        params = np.repeat(params, int(transition_days.shape[0]), axis=0)
    elif params.shape[0] != int(transition_days.shape[0]):
        raise ValueError(
            "params_np and transition_days_np must align by sample count, "
            f"got {params.shape[0]} and {transition_days.shape[0]}"
        )

    params_norm = normalize_params(params, param_stats)
    transition_days_norm = normalize_params(
        transition_days.reshape(-1, 1),
        transition_time_stats,
    )
    return np.concatenate([params_norm, transition_days_norm], axis=1).astype(np.float32)


def stats_to_json(stats: NormalizationStats) -> Dict[str, object]:
    """Serialize normalization stats into JSON-friendly primitives."""
    return {
        "input_state": {
            "field_names": list(stats.input_state.field_names),
            "field_transforms": dict(stats.input_state.field_transforms),
            "mean": stats.input_state.mean.tolist(),
            "std": stats.input_state.std.tolist(),
            "zscore_eps": float(stats.input_state.zscore_eps),
            "log10_eps": float(stats.input_state.log10_eps),
            "signed_log1p_scale": float(stats.input_state.signed_log1p_scale),
        },
        "target_state": {
            "field_names": list(stats.target_state.field_names),
            "field_transforms": dict(stats.target_state.field_transforms),
            "mean": stats.target_state.mean.tolist(),
            "std": stats.target_state.std.tolist(),
            "zscore_eps": float(stats.target_state.zscore_eps),
            "log10_eps": float(stats.target_state.log10_eps),
            "signed_log1p_scale": float(stats.target_state.signed_log1p_scale),
        },
        "params": {
            "param_names": list(stats.params.param_names),
            "mean": stats.params.mean.tolist(),
            "std": stats.params.std.tolist(),
            "is_constant": np.asarray(stats.params.is_constant, dtype=bool).tolist(),
            "zscore_eps": float(stats.params.zscore_eps),
        },
        "transition_time": {
            "param_names": list(stats.transition_time.param_names),
            "mean": stats.transition_time.mean.tolist(),
            "std": stats.transition_time.std.tolist(),
            "is_constant": np.asarray(stats.transition_time.is_constant, dtype=bool).tolist(),
            "zscore_eps": float(stats.transition_time.zscore_eps),
        },
    }


def stats_from_json(data: Dict[str, object]) -> NormalizationStats:
    """Deserialize JSON metadata into ``NormalizationStats``."""
    input_state = dict(data["input_state"])
    target_state = dict(data["target_state"])
    params = dict(data["params"])
    transition_time = dict(
        data.get(
            "transition_time",
            {
                "param_names": [TRANSITION_TIME_NAME],
                "mean": [0.0],
                "std": [1.0],
                "is_constant": [True],
                "zscore_eps": float(params["zscore_eps"]),
            },
        )
    )

    param_mean = np.asarray(params["mean"], dtype=np.float64)
    param_is_constant = np.asarray(
        params.get("is_constant", np.zeros_like(param_mean, dtype=bool)),
        dtype=bool,
    )
    if param_is_constant.shape != param_mean.shape:
        raise ValueError("Invalid normalization JSON: params.is_constant shape mismatch")

    transition_time_mean = np.asarray(transition_time["mean"], dtype=np.float64)
    transition_time_is_constant = np.asarray(
        transition_time.get("is_constant", np.zeros_like(transition_time_mean, dtype=bool)),
        dtype=bool,
    )
    if transition_time_is_constant.shape != transition_time_mean.shape:
        raise ValueError(
            "Invalid normalization JSON: transition_time.is_constant shape mismatch"
        )

    return NormalizationStats(
        input_state=StateNormalizationStats(
            field_names=tuple(str(value) for value in input_state["field_names"]),
            field_transforms={
                str(k): str(v)
                for k, v in dict(input_state["field_transforms"]).items()
            },
            mean=np.asarray(input_state["mean"], dtype=np.float64),
            std=np.asarray(input_state["std"], dtype=np.float64),
            zscore_eps=float(input_state["zscore_eps"]),
            log10_eps=float(input_state["log10_eps"]),
            signed_log1p_scale=float(input_state["signed_log1p_scale"]),
        ),
        target_state=StateNormalizationStats(
            field_names=tuple(str(value) for value in target_state["field_names"]),
            field_transforms={
                str(k): str(v)
                for k, v in dict(target_state["field_transforms"]).items()
            },
            mean=np.asarray(target_state["mean"], dtype=np.float64),
            std=np.asarray(target_state["std"], dtype=np.float64),
            zscore_eps=float(target_state["zscore_eps"]),
            log10_eps=float(target_state["log10_eps"]),
            signed_log1p_scale=float(target_state["signed_log1p_scale"]),
        ),
        params=ParamNormalizationStats(
            param_names=tuple(str(value) for value in params["param_names"]),
            mean=param_mean,
            std=np.asarray(params["std"], dtype=np.float64),
            is_constant=param_is_constant,
            zscore_eps=float(params["zscore_eps"]),
        ),
        transition_time=ParamNormalizationStats(
            param_names=tuple(str(value) for value in transition_time["param_names"]),
            mean=transition_time_mean,
            std=np.asarray(transition_time["std"], dtype=np.float64),
            is_constant=transition_time_is_constant,
            zscore_eps=float(transition_time["zscore_eps"]),
        ),
    )
