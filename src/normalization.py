"""State/parameter normalization transforms and stats serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .config import NormalizationConfig

# Numerical guards for stable transform/inverse-transform behavior.
STD_FLOOR = 1.0e-12
LOG10_INVERSE_CLIP_MIN = -30.0
LOG10_INVERSE_CLIP_MAX = 30.0
PARAM_NORM_CLIP_ABS = 1.0e6


@dataclass(frozen=True)
class NormalizationStats:
    """Frozen normalization statistics persisted in metadata and checkpoints."""

    field_names: Tuple[str, ...]
    param_names: Tuple[str, ...]
    field_transforms: Dict[str, str]
    state_mean: np.ndarray
    state_std: np.ndarray
    param_mean: np.ndarray
    param_std: np.ndarray
    param_is_constant: np.ndarray
    state_zscore_eps: float
    param_zscore_eps: float
    log10_eps: float
    signed_log1p_scale: float


def _transform_channel(x: np.ndarray, mode: str, *, log10_eps: float, signed_log1p_scale: float) -> np.ndarray:
    """Apply one configured forward transform to one state channel."""
    if mode == "none":
        return x
    if mode == "log10":
        finite = np.isfinite(x)
        if not np.all(finite):
            n_bad = int(np.count_nonzero(~finite))
            raise ValueError(f"log10 transform received {n_bad} non-finite values")
        non_pos = x <= 0.0
        if np.any(non_pos):
            n_bad = int(np.count_nonzero(non_pos))
            x_min = float(np.min(x))
            raise ValueError(
                "log10 transform requires strictly positive values, "
                f"but found {n_bad} non-positive entries (min={x_min:.6e}). "
                "For signed fields (for example Phi perturbations), use 'signed_log1p' or 'none'."
            )
        return np.log10(np.maximum(x, log10_eps))
    if mode == "signed_log1p":
        s = float(signed_log1p_scale)
        if s <= 0:
            raise ValueError("signed_log1p_scale must be > 0")
        return np.sign(x) * np.log1p(np.abs(x) / s)
    raise ValueError(f"Unsupported normalization transform mode: {mode}")


def _inverse_transform_channel(x: np.ndarray, mode: str, *, signed_log1p_scale: float) -> np.ndarray:
    """Invert one state-channel transform."""
    if mode == "none":
        return x
    if mode == "log10":
        # Guard against numeric overflow from unconstrained model outputs.
        return np.power(10.0, np.clip(x, LOG10_INVERSE_CLIP_MIN, LOG10_INVERSE_CLIP_MAX))
    if mode == "signed_log1p":
        s = float(signed_log1p_scale)
        return np.sign(x) * (np.expm1(np.abs(x)) * s)
    raise ValueError(f"Unsupported normalization transform mode: {mode}")


def apply_state_transforms(states_nchw: np.ndarray, field_names: Sequence[str], cfg: NormalizationConfig) -> np.ndarray:
    """Apply per-field configured transforms to a [N,C,H,W] state tensor."""
    if states_nchw.ndim != 4:
        raise ValueError(f"states_nchw must be [N,C,H,W], got {states_nchw.shape}")
    out = states_nchw.astype(np.float64, copy=True)
    for ci, name in enumerate(field_names):
        mode = cfg.field_transforms.get(name, "none")
        out[:, ci] = _transform_channel(
            out[:, ci],
            mode,
            log10_eps=cfg.log10_eps,
            signed_log1p_scale=cfg.signed_log1p_scale,
        )
    return out


def inverse_state_transforms(states_nchw: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Invert per-field transforms using persisted normalization statistics."""
    out = states_nchw.astype(np.float64, copy=True)
    for ci, name in enumerate(stats.field_names):
        mode = stats.field_transforms.get(name, "none")
        out[:, ci] = _inverse_transform_channel(out[:, ci], mode, signed_log1p_scale=stats.signed_log1p_scale)
    return out


def normalize_states(states_nchw: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Transform + z-score normalize state tensors to float32."""
    cfg = NormalizationConfig(
        field_transforms={k: str(v) for k, v in stats.field_transforms.items()},
        zscore_eps=stats.state_zscore_eps,
        log10_eps=stats.log10_eps,
        signed_log1p_scale=stats.signed_log1p_scale,
    )
    st = apply_state_transforms(states_nchw, stats.field_names, cfg)
    st = (st - stats.state_mean.reshape(1, -1, 1, 1)) / (stats.state_std.reshape(1, -1, 1, 1) + stats.state_zscore_eps)
    return st.astype(np.float32)


def denormalize_states(states_norm_nchw: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Invert z-score and field transforms for normalized state tensors."""
    st = states_norm_nchw.astype(np.float64)
    st = st * (stats.state_std.reshape(1, -1, 1, 1) + stats.state_zscore_eps) + stats.state_mean.reshape(1, -1, 1, 1)
    st = inverse_state_transforms(st, stats)
    return st.astype(np.float32)


def normalize_params(params_np: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Z-score normalize parameter vectors and zero constant channels."""
    out = (params_np.astype(np.float64) - stats.param_mean.reshape(1, -1)) / (stats.param_std.reshape(1, -1) + stats.param_zscore_eps)
    const_mask = np.asarray(stats.param_is_constant, dtype=bool)
    if const_mask.size != out.shape[1]:
        raise ValueError(
            "NormalizationStats.param_is_constant length mismatch: "
            f"got {const_mask.size}, expected {out.shape[1]}"
        )
    if np.any(const_mask):
        out[:, const_mask] = 0.0
    out = np.clip(out, -PARAM_NORM_CLIP_ABS, PARAM_NORM_CLIP_ABS)
    return out.astype(np.float32)


def stats_to_json(stats: NormalizationStats) -> Dict[str, object]:
    """Serialize normalization stats into JSON-encodable primitives."""
    return {
        "field_names": list(stats.field_names),
        "param_names": list(stats.param_names),
        "field_transforms": dict(stats.field_transforms),
        "state_mean": stats.state_mean.tolist(),
        "state_std": stats.state_std.tolist(),
        "param_mean": stats.param_mean.tolist(),
        "param_std": stats.param_std.tolist(),
        "param_is_constant": np.asarray(stats.param_is_constant, dtype=bool).tolist(),
        "state_zscore_eps": stats.state_zscore_eps,
        "param_zscore_eps": stats.param_zscore_eps,
        "log10_eps": stats.log10_eps,
        "signed_log1p_scale": stats.signed_log1p_scale,
    }


def stats_from_json(d: Dict[str, object]) -> NormalizationStats:
    """Deserialize JSON metadata into ``NormalizationStats``."""
    param_mean = np.asarray(d["param_mean"], dtype=np.float64)
    param_const_raw = d.get("param_is_constant")
    if param_const_raw is None:
        param_is_constant = np.zeros(param_mean.shape, dtype=bool)
    else:
        param_is_constant = np.asarray(param_const_raw, dtype=bool)
    if param_is_constant.shape != param_mean.shape:
        raise ValueError(
            "Invalid normalization JSON: param_is_constant shape mismatch with param_mean "
            f"({param_is_constant.shape} vs {param_mean.shape})"
        )

    return NormalizationStats(
        field_names=tuple(str(x) for x in d["field_names"]),
        param_names=tuple(str(x) for x in d["param_names"]),
        field_transforms={str(k): str(v) for k, v in dict(d["field_transforms"]).items()},
        state_mean=np.asarray(d["state_mean"], dtype=np.float64),
        state_std=np.asarray(d["state_std"], dtype=np.float64),
        param_mean=param_mean,
        param_std=np.asarray(d["param_std"], dtype=np.float64),
        param_is_constant=param_is_constant,
        state_zscore_eps=float(d["state_zscore_eps"]),
        param_zscore_eps=float(d["param_zscore_eps"]),
        log10_eps=float(d["log10_eps"]),
        signed_log1p_scale=float(d["signed_log1p_scale"]),
    )
