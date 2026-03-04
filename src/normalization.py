from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .config import NormalizationConfig, ParamNormConfig
from .constants import LOG10_INVERSE_CLIP_MAX, LOG10_INVERSE_CLIP_MIN, PARAM_NORM_CLIP_ABS, STD_FLOOR


@dataclass(frozen=True)
class NormalizationStats:
    field_names: Tuple[str, ...]
    param_names: Tuple[str, ...]
    field_transforms: Dict[str, str]
    state_mean: np.ndarray
    state_std: np.ndarray
    param_mean: np.ndarray
    param_std: np.ndarray
    state_zscore_eps: float
    param_zscore_eps: float
    log10_eps: float
    signed_log1p_scale: float


def _transform_channel(x: np.ndarray, mode: str, *, log10_eps: float, signed_log1p_scale: float) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "log10":
        return np.log10(np.maximum(x, log10_eps))
    if mode == "signed_log1p":
        s = float(signed_log1p_scale)
        if s <= 0:
            raise ValueError("signed_log1p_scale must be > 0")
        return np.sign(x) * np.log1p(np.abs(x) / s)
    raise ValueError(f"Unsupported normalization transform mode: {mode}")


def _inverse_transform_channel(x: np.ndarray, mode: str, *, signed_log1p_scale: float) -> np.ndarray:
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
    out = states_nchw.astype(np.float64, copy=True)
    for ci, name in enumerate(stats.field_names):
        mode = stats.field_transforms.get(name, "none")
        out[:, ci] = _inverse_transform_channel(out[:, ci], mode, signed_log1p_scale=stats.signed_log1p_scale)
    return out


def fit_normalization(
    *,
    train_states_nchw: np.ndarray,
    train_params_np: np.ndarray,
    field_names: Sequence[str],
    param_names: Sequence[str],
    state_cfg: NormalizationConfig,
    param_cfg: ParamNormConfig,
) -> NormalizationStats:
    st = apply_state_transforms(train_states_nchw, field_names, state_cfg)

    # Per-channel moments across N,H,W
    state_mean = st.mean(axis=(0, 2, 3))
    state_std = st.std(axis=(0, 2, 3))
    state_std = np.maximum(state_std, STD_FLOOR)

    if param_cfg.mode == "zscore":
        param_mean = train_params_np.mean(axis=0)
        param_std = train_params_np.std(axis=0)
        param_std = np.maximum(param_std, STD_FLOOR)
    elif param_cfg.mode == "none":
        param_mean = np.zeros(train_params_np.shape[1], dtype=np.float64)
        param_std = np.ones(train_params_np.shape[1], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported param normalization mode: {param_cfg.mode}")

    return NormalizationStats(
        field_names=tuple(str(x) for x in field_names),
        param_names=tuple(str(x) for x in param_names),
        field_transforms=dict(state_cfg.field_transforms),
        state_mean=state_mean.astype(np.float64),
        state_std=state_std.astype(np.float64),
        param_mean=param_mean.astype(np.float64),
        param_std=param_std.astype(np.float64),
        state_zscore_eps=float(state_cfg.zscore_eps),
        param_zscore_eps=float(param_cfg.eps),
        log10_eps=float(state_cfg.log10_eps),
        signed_log1p_scale=float(state_cfg.signed_log1p_scale),
    )


def normalize_states(states_nchw: np.ndarray, stats: NormalizationStats) -> np.ndarray:
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
    st = states_norm_nchw.astype(np.float64)
    st = st * (stats.state_std.reshape(1, -1, 1, 1) + stats.state_zscore_eps) + stats.state_mean.reshape(1, -1, 1, 1)
    st = inverse_state_transforms(st, stats)
    return st.astype(np.float32)


def normalize_params(params_np: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    out = (params_np.astype(np.float64) - stats.param_mean.reshape(1, -1)) / (stats.param_std.reshape(1, -1) + stats.param_zscore_eps)
    out = np.clip(out, -PARAM_NORM_CLIP_ABS, PARAM_NORM_CLIP_ABS)
    return out.astype(np.float32)


def stats_to_json(stats: NormalizationStats) -> Dict[str, object]:
    return {
        "field_names": list(stats.field_names),
        "param_names": list(stats.param_names),
        "field_transforms": dict(stats.field_transforms),
        "state_mean": stats.state_mean.tolist(),
        "state_std": stats.state_std.tolist(),
        "param_mean": stats.param_mean.tolist(),
        "param_std": stats.param_std.tolist(),
        "state_zscore_eps": stats.state_zscore_eps,
        "param_zscore_eps": stats.param_zscore_eps,
        "log10_eps": stats.log10_eps,
        "signed_log1p_scale": stats.signed_log1p_scale,
    }


def stats_from_json(d: Dict[str, object]) -> NormalizationStats:
    return NormalizationStats(
        field_names=tuple(str(x) for x in d["field_names"]),
        param_names=tuple(str(x) for x in d["param_names"]),
        field_transforms={str(k): str(v) for k, v in dict(d["field_transforms"]).items()},
        state_mean=np.asarray(d["state_mean"], dtype=np.float64),
        state_std=np.asarray(d["state_std"], dtype=np.float64),
        param_mean=np.asarray(d["param_mean"], dtype=np.float64),
        param_std=np.asarray(d["param_std"], dtype=np.float64),
        state_zscore_eps=float(d["state_zscore_eps"]),
        param_zscore_eps=float(d["param_zscore_eps"]),
        log10_eps=float(d["log10_eps"]),
        signed_log1p_scale=float(d["signed_log1p_scale"]),
    )
