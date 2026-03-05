"""Preprocessing and training pipeline for trajectory-transition learning."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import GCMulatorConfig, resolve_path
from .geometry import geometry_shift_for_nlon
from .modeling import (
    SphereLoss,
    autocast_context,
    build_state_conditioned_rollout_model,
    choose_device,
    ensure_torch_harmonics_importable,
)
from .normalization import (
    NormalizationStats,
    apply_state_transforms,
    denormalize_states,
    normalize_params,
    normalize_states,
    stats_from_json,
    stats_to_json,
)

LOGGER = logging.getLogger("src.train")

# Lower bound used when computing z-score denominators from empirical variance.
STD_FLOOR = 1.0e-12
FINITE_CHECK_INTERVAL = 25
PREPROCESS_FINGERPRINT_VERSION = 9
EXPECTED_CONDITIONING_PARAM_NAMES = (
    "a_m",
    "omega_rad_s",
    "Phibar",
    "DPhieq",
    "taurad_s",
    "taudrag_s",
    "g_m_s2",
)
TRANSITION_DAYS_CONDITIONING_NAME = "transition_days"


def _load_npy_payload_dict(file_path: Path) -> Dict[str, Any]:
    """Load a dictionary payload stored via ``np.save(..., allow_pickle=True)``."""
    obj = np.load(file_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        payload = obj.item()
    else:
        payload = obj
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {file_path}, got {type(payload).__name__}")
    return payload


def _list_raw_dataset_files(dataset_dir: Path) -> List[Path]:
    """List raw simulation files and fail when dataset is empty."""
    files_npy = sorted(dataset_dir.glob("sim_*.npy"))
    if files_npy:
        return files_npy
    files_npz = sorted(dataset_dir.glob("sim_*.npz"))
    if files_npz:
        LOGGER.warning(
            "Migrating %d legacy raw .npz files to .npy in %s.",
            len(files_npz),
            dataset_dir,
        )
        migrated: List[Path] = []
        for fp in files_npz:
            with np.load(fp, allow_pickle=True) as z:
                payload = {k: z[k] for k in z.files}
            out = fp.with_suffix(".npy")
            np.save(out, payload, allow_pickle=True)
            fp.unlink()
            migrated.append(out)
        return sorted(migrated)
    raise FileNotFoundError(f"No sim_*.npy or sim_*.npz files found in {dataset_dir}")


def _split_files(
    files: Sequence[Path],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Create reproducible file-level train/validation/test split."""
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"training.val_fraction must be in (0,1), got {val_fraction}")
    if not (0.0 < float(test_fraction) < 1.0):
        raise ValueError(f"training.test_fraction must be in (0,1), got {test_fraction}")
    if (float(val_fraction) + float(test_fraction)) >= 1.0:
        raise ValueError(
            "training.val_fraction + training.test_fraction must be < 1, got "
            f"{float(val_fraction) + float(test_fraction):.6f}"
        )
    idx = np.arange(len(files))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_total = len(files)
    n_val = max(1, int(round(float(val_fraction) * n_total)))
    n_test = max(1, int(round(float(test_fraction) * n_total)))
    if (n_val + n_test) >= n_total:
        raise ValueError(
            f"Invalid split sizes for n_files={n_total}: "
            f"n_val={n_val}, n_test={n_test}, n_train={n_total - n_val - n_test}. "
            "Increase dataset size or reduce val/test fractions."
        )

    test_ids = set(idx[:n_test].tolist())
    val_ids = set(idx[n_test : (n_test + n_val)].tolist())
    train = [files[i] for i in range(n_total) if i not in val_ids and i not in test_ids]
    val = [files[i] for i in range(n_total) if i in val_ids]
    test = [files[i] for i in range(n_total) if i in test_ids]
    if not train:
        raise ValueError(
            "Train split is empty. "
            f"(n_files={len(files)}, val_fraction={val_fraction}, test_fraction={test_fraction}). "
            "Increase dataset size or reduce val/test fractions."
        )
    if not val:
        raise ValueError(
            "Validation split is empty. "
            f"(n_files={len(files)}, val_fraction={val_fraction}, test_fraction={test_fraction}). "
            "Increase dataset size or increase val_fraction."
        )
    if not test:
        raise ValueError(
            "Test split is empty. "
            f"(n_files={len(files)}, val_fraction={val_fraction}, test_fraction={test_fraction}). "
            "Increase dataset size or increase test_fraction."
        )
    return train, val, test


def _load_raw_metadata(file_path: Path) -> Dict[str, Any]:
    """Load metadata from one raw simulation file without touching state arrays."""
    if file_path.suffix.lower() == ".npy":
        z = _load_npy_payload_dict(file_path)
        try:
            lat_order = str(np.asarray(z["lat_order"], dtype=object).item())
            lon_origin = str(np.asarray(z["lon_origin"], dtype=object).item())
            lon_shift = int(np.asarray(z["lon_shift"], dtype=np.int64).item())
            nlat = int(np.asarray(z["nlat"], dtype=np.int64).item())
            nlon = int(np.asarray(z["nlon"], dtype=np.int64).item())
            transition_jump_steps = int(np.asarray(z["transition_jump_steps"], dtype=np.int64).item())
            fields = [str(x) for x in np.asarray(z["fields"], dtype=object).tolist()]
            param_names = [str(x) for x in np.asarray(z["param_names"], dtype=object).tolist()]
        except KeyError as exc:
            raise ValueError(
                f"Raw file {file_path} is missing geometry metadata key '{exc.args[0]}'. "
                "Regenerate the raw dataset with current data_generation.py."
            ) from exc
    else:
        with np.load(file_path, allow_pickle=True) as z:
            try:
                lat_order = str(np.asarray(z["lat_order"], dtype=object).item())
                lon_origin = str(np.asarray(z["lon_origin"], dtype=object).item())
                lon_shift = int(np.asarray(z["lon_shift"], dtype=np.int64).item())
                nlat = int(np.asarray(z["nlat"], dtype=np.int64).item())
                nlon = int(np.asarray(z["nlon"], dtype=np.int64).item())
                transition_jump_steps = int(np.asarray(z["transition_jump_steps"], dtype=np.int64).item())
            except KeyError as exc:
                raise ValueError(
                    f"Raw file {file_path} is missing geometry metadata key '{exc.args[0]}'. "
                    "Regenerate the raw dataset with current data_generation.py."
                ) from exc
            fields = [str(x) for x in np.asarray(z["fields"], dtype=object).tolist()]
            param_names = [str(x) for x in np.asarray(z["param_names"], dtype=object).tolist()]
    return {
        "lat_order": lat_order,
        "lon_origin": lon_origin,
        "lon_shift": lon_shift,
        "nlat": nlat,
        "nlon": nlon,
        "transition_jump_steps": transition_jump_steps,
        "fields": fields,
        "param_names": param_names,
    }


def _validate_raw_geometry(files: Sequence[Path], *, cfg: GCMulatorConfig) -> Dict[str, Any]:
    """Validate raw geometry consistency and match against configured conventions."""
    expected_lat_order = "north_to_south" if cfg.geometry.flip_latitude_to_north_south else "south_to_north"
    expected_lon_origin = "0_to_2pi" if cfg.geometry.roll_longitude_to_0_2pi else "minus_pi_to_pi"

    ref: Dict[str, Any] | None = None
    for fp in files:
        meta = _load_raw_metadata(fp)
        expected_lon_shift = geometry_shift_for_nlon(int(meta["nlon"]), cfg.geometry.roll_longitude_to_0_2pi)
        if int(meta["lon_shift"]) != int(expected_lon_shift):
            raise ValueError(
                f"Geometry mismatch in {fp}: lon_shift={meta['lon_shift']} but expected "
                f"{expected_lon_shift} for nlon={meta['nlon']} and roll_longitude_to_0_2pi="
                f"{cfg.geometry.roll_longitude_to_0_2pi}."
            )

        if str(meta["lat_order"]) != expected_lat_order or str(meta["lon_origin"]) != expected_lon_origin:
            raise ValueError(
                f"Geometry mismatch in {fp}: "
                f"lat_order={meta['lat_order']} lon_origin={meta['lon_origin']} but expected "
                f"lat_order={expected_lat_order} lon_origin={expected_lon_origin} from config.geometry."
            )

        if ref is None:
            ref = {
                "lat_order": str(meta["lat_order"]),
                "lon_origin": str(meta["lon_origin"]),
                "lon_shift": int(meta["lon_shift"]),
                "nlat": int(meta["nlat"]),
                "nlon": int(meta["nlon"]),
                "transition_jump_steps": int(meta["transition_jump_steps"]),
            }
        else:
            current = (
                str(meta["lat_order"]),
                str(meta["lon_origin"]),
                int(meta["lon_shift"]),
                int(meta["nlat"]),
                int(meta["nlon"]),
                int(meta["transition_jump_steps"]),
            )
            baseline = (
                str(ref["lat_order"]),
                str(ref["lon_origin"]),
                int(ref["lon_shift"]),
                int(ref["nlat"]),
                int(ref["nlon"]),
                int(ref["transition_jump_steps"]),
            )
            if current != baseline:
                raise ValueError(
                    "Raw dataset mixes geometry conventions or grid shapes across files. "
                    f"First mismatch: {fp} has {current}, expected {baseline}."
                )

    if ref is None:
        raise ValueError("Raw dataset is empty while validating geometry.")
    return ref


def _raw_file_signature(file_path: Path) -> Dict[str, Any]:
    """Return a compact signature used to detect raw-dataset changes."""
    st = file_path.stat()
    return {
        "name": file_path.name,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _build_preprocess_fingerprint(*, cfg: GCMulatorConfig, files: Sequence[Path]) -> Dict[str, Any]:
    """Build reproducibility fingerprint for processed-data cache reuse."""
    return {
        "version": PREPROCESS_FINGERPRINT_VERSION,
        "split_seed": int(cfg.training.split_seed),
        "val_fraction": float(cfg.training.val_fraction),
        "test_fraction": float(cfg.training.test_fraction),
        "normalization": asdict(cfg.normalization),
        "geometry": asdict(cfg.geometry),
        "solver": {
            "M": int(cfg.solver.M),
            "dt_seconds": float(cfg.solver.dt_seconds),
            "default_time_days": float(cfg.solver.default_time_days),
        },
        "model_time_mapping": {
            "default_time_days": float(cfg.solver.default_time_days),
            "transition_jump_steps": int(cfg.model.transition_jump_steps),
            "rollout_steps_at_default_time": int(cfg.model.rollout_steps_at_default_time),
        },
        "raw_files": [_raw_file_signature(fp) for fp in files],
    }


def _processed_cache_is_valid(*, meta: Dict[str, Any], fingerprint: Dict[str, Any], processed_dir: Path) -> bool:
    """Return True when processed data matches the current preprocessing fingerprint."""
    if str(meta.get("task", "")) != "trajectory_transition":
        return False
    if meta.get("build_fingerprint") != fingerprint:
        return False
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        return False
    for split_name in ("train", "val", "test"):
        names = splits.get(split_name)
        if not isinstance(names, list) or not names:
            return False
        for name in names:
            if not (processed_dir / str(name)).is_file():
                return False
    return True


def _load_raw_transitions_and_params(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, List[str], List[str]]:
    """Load one raw simulation file and extract transition tensors and metadata."""
    if file_path.suffix.lower() == ".npy":
        z = _load_npy_payload_dict(file_path)
        try:
            state_inputs = np.asarray(z["state_inputs"], dtype=np.float32)
            state_targets = np.asarray(z["state_targets"], dtype=np.float32)
            transition_days = np.asarray(z["transition_days"], dtype=np.float64)
            params = np.asarray(z["params"], dtype=np.float64)
            time_days = float(np.asarray(z["time_days"]).item())
            _transition_jump_steps = int(np.asarray(z["transition_jump_steps"], dtype=np.int64).item())
            fields = [str(x) for x in np.asarray(z["fields"], dtype=object).tolist()]
            param_names = [str(x) for x in np.asarray(z["param_names"], dtype=object).tolist()]
        except KeyError as exc:
            raise ValueError(
                f"Raw file {file_path} is missing trajectory-transition key '{exc.args[0]}'. "
                "Regenerate raw data with current data_generation.py."
            ) from exc
    else:
        with np.load(file_path, allow_pickle=True) as z:
            try:
                state_inputs = np.asarray(z["state_inputs"], dtype=np.float32)
                state_targets = np.asarray(z["state_targets"], dtype=np.float32)
                transition_days = np.asarray(z["transition_days"], dtype=np.float64)
                params = np.asarray(z["params"], dtype=np.float64)
                time_days = float(np.asarray(z["time_days"]).item())
                _transition_jump_steps = int(np.asarray(z["transition_jump_steps"], dtype=np.int64).item())
                fields = [str(x) for x in np.asarray(z["fields"], dtype=object).tolist()]
                param_names = [str(x) for x in np.asarray(z["param_names"], dtype=object).tolist()]
            except KeyError as exc:
                raise ValueError(
                    f"Raw file {file_path} is missing trajectory-transition key '{exc.args[0]}'. "
                    "Regenerate raw data with current data_generation.py."
                ) from exc
    return state_inputs, state_targets, transition_days, params, time_days, fields, param_names


def _fit_stats_streaming(
    *,
    train_files: Sequence[Path],
    state_norm_cfg,
    param_norm_cfg,
) -> NormalizationStats:
    """Compute normalization statistics from train split using streaming moments."""
    fields0: List[str] | None = None
    pnames0: List[str] | None = None

    state_sum: np.ndarray | None = None
    state_sum2: np.ndarray | None = None
    state_count = 0

    param_mean_accum: np.ndarray | None = None
    param_m2: np.ndarray | None = None
    param_count = 0

    for fp in train_files:
        st_in, st_out, _dt, p, _t, fields, pnames = _load_raw_transitions_and_params(fp)
        if fields0 is None:
            fields0 = fields
        elif fields != fields0:
            raise ValueError(f"Field mismatch in {fp}: {fields} != {fields0}")

        if pnames0 is None:
            pnames0 = pnames
            if tuple(pnames0) != EXPECTED_CONDITIONING_PARAM_NAMES:
                raise ValueError(
                    "Raw dataset parameter schema is incompatible with current user-facing conditioning contract. "
                    f"Expected {list(EXPECTED_CONDITIONING_PARAM_NAMES)}, got {pnames0}. "
                    "Regenerate raw data with current config and rerun preprocessing."
                )
        elif pnames != pnames0:
            raise ValueError(f"Param-name mismatch in {fp}: {pnames} != {pnames0}")

        if st_in.ndim != 4 or st_out.ndim != 4:
            raise ValueError(f"Raw transition tensors must be [T,C,H,W], got {st_in.shape} and {st_out.shape}")
        if st_in.shape != st_out.shape:
            raise ValueError(f"Raw transition shape mismatch in {fp}: {st_in.shape} vs {st_out.shape}")

        st_pair = np.concatenate([st_in, st_out], axis=0)  # [2T,C,H,W]
        st_tr = apply_state_transforms(st_pair, fields0, state_norm_cfg).astype(np.float64, copy=False)
        s = st_tr.sum(axis=(0, 2, 3))
        s2 = (st_tr * st_tr).sum(axis=(0, 2, 3))
        n = int(st_tr.shape[0] * st_tr.shape[2] * st_tr.shape[3])

        if state_sum is None:
            state_sum = s
            state_sum2 = s2
        else:
            state_sum += s
            state_sum2 += s2
        state_count += n

        p64 = p.astype(np.float64, copy=False)
        if param_mean_accum is None:
            param_mean_accum = p64.copy()
            param_m2 = np.zeros_like(p64, dtype=np.float64)
            param_count = 1
        else:
            param_count += 1
            delta = p64 - param_mean_accum
            param_mean_accum += delta / float(param_count)
            delta2 = p64 - param_mean_accum
            param_m2 += delta * delta2

    if fields0 is None or pnames0 is None:
        raise RuntimeError("Could not infer field/parameter names from training files")
    if state_sum is None or state_sum2 is None:
        raise RuntimeError("Failed to accumulate state normalization moments")
    if param_mean_accum is None or param_m2 is None:
        raise RuntimeError("Failed to accumulate parameter normalization moments")

    state_mean = state_sum / float(state_count)
    state_var = np.maximum(state_sum2 / float(state_count) - state_mean * state_mean, 0.0)
    state_std = np.maximum(np.sqrt(state_var), STD_FLOOR)

    if param_norm_cfg.mode == "zscore":
        param_mean = param_mean_accum
        if param_count > 1:
            param_var = np.maximum(param_m2 / float(param_count), 0.0)
        else:
            param_var = np.zeros_like(param_mean, dtype=np.float64)
        param_std_raw = np.sqrt(param_var)
        param_is_constant = param_std_raw <= STD_FLOOR
        param_std = np.where(param_is_constant, 1.0, np.maximum(param_std_raw, STD_FLOOR))
    elif param_norm_cfg.mode == "none":
        param_mean = np.zeros_like(param_mean_accum)
        param_std = np.ones_like(param_mean_accum)
        param_is_constant = np.zeros_like(param_mean_accum, dtype=bool)
    else:
        raise ValueError(f"Unsupported param normalization mode: {param_norm_cfg.mode}")

    return NormalizationStats(
        field_names=tuple(fields0),
        param_names=tuple(pnames0),
        field_transforms=dict(state_norm_cfg.field_transforms),
        state_mean=state_mean.astype(np.float64),
        state_std=state_std.astype(np.float64),
        param_mean=param_mean.astype(np.float64),
        param_std=param_std.astype(np.float64),
        param_is_constant=np.asarray(param_is_constant, dtype=bool),
        state_zscore_eps=float(state_norm_cfg.zscore_eps),
        param_zscore_eps=float(param_norm_cfg.eps),
        log10_eps=float(state_norm_cfg.log10_eps),
        signed_log1p_scale=float(state_norm_cfg.signed_log1p_scale),
    )


def _fit_transition_days_norm_stats(
    *,
    train_files: Sequence[Path],
    zscore_eps: float,
) -> Dict[str, Any]:
    """Fit z-score statistics for transition duration conditioning feature."""
    transition_sum = 0.0
    transition_sum2 = 0.0
    transition_count = 0

    for fp in train_files:
        _st_in, _st_out, transition_days, _params, _time_days, _fields, _param_names = _load_raw_transitions_and_params(
            fp
        )
        transition_vals = np.asarray(transition_days, dtype=np.float64)
        transition_sum += float(np.sum(transition_vals))
        transition_sum2 += float(np.sum(transition_vals * transition_vals))
        transition_count += int(transition_vals.size)

    if transition_count < 1:
        raise RuntimeError("Failed to accumulate transition_days normalization moments")

    transition_mean = transition_sum / float(transition_count)
    transition_var = max(transition_sum2 / float(transition_count) - transition_mean * transition_mean, 0.0)
    transition_std_raw = math.sqrt(transition_var)
    transition_is_constant = bool(transition_std_raw <= STD_FLOOR)
    transition_std = 1.0 if transition_is_constant else max(float(transition_std_raw), STD_FLOOR)

    return {
        "name": TRANSITION_DAYS_CONDITIONING_NAME,
        "mean": float(transition_mean),
        "std": float(transition_std),
        "is_constant": bool(transition_is_constant),
        "zscore_eps": float(zscore_eps),
    }


def _write_processed_file(
    *,
    src_file: Path,
    dst_dir: Path,
    split_prefix: str,
    stats: NormalizationStats,
    transition_days_norm: Dict[str, Any],
) -> List[str]:
    """Normalize one raw trajectory file and write per-transition processed files."""
    st_in, st_out, transition_days, p, _time_days, _fields, _pnames = _load_raw_transitions_and_params(src_file)
    if st_in.ndim != 4 or st_out.ndim != 4:
        raise ValueError(f"Raw transitions must be [T,C,H,W], got {st_in.shape} and {st_out.shape}")
    if st_in.shape != st_out.shape:
        raise ValueError(f"Raw transition shape mismatch in {src_file}: {st_in.shape} vs {st_out.shape}")
    if transition_days.ndim != 1 or int(transition_days.shape[0]) != int(st_in.shape[0]):
        raise ValueError(
            f"transition_days shape mismatch in {src_file}: {transition_days.shape} for T={st_in.shape[0]}"
        )

    st_in_norm = normalize_states(st_in, stats).astype(np.float32)
    st_out_norm = normalize_states(st_out, stats).astype(np.float32)
    params_norm = normalize_params(p[None, ...], stats)[0].astype(np.float32)
    transition_days_mean = float(transition_days_norm["mean"])
    transition_days_std = float(transition_days_norm["std"])
    transition_days_eps = float(transition_days_norm["zscore_eps"])
    transition_days_is_constant = bool(transition_days_norm["is_constant"])

    names: List[str] = []
    t_count = int(st_in.shape[0])
    for ti in range(t_count):
        transition_days_value = float(transition_days[ti])
        if transition_days_is_constant:
            transition_days_value_norm = 0.0
        else:
            transition_days_value_norm = (
                (transition_days_value - transition_days_mean) / (transition_days_std + transition_days_eps)
            )
        conditioning_norm = np.concatenate(
            [
                params_norm.astype(np.float32, copy=False),
                np.asarray([transition_days_value_norm], dtype=np.float32),
            ],
            axis=0,
        )

        name = f"{split_prefix}_tr{ti:04d}.npy"
        payload = {
            "state_input_norm": st_in_norm[ti].astype(np.float32),
            "state_target_norm": st_out_norm[ti].astype(np.float32),
            "params_norm": params_norm.astype(np.float32),
            "conditioning_norm": conditioning_norm.astype(np.float32),
            "transition_days": np.asarray(transition_days_value, dtype=np.float64),
            "transition_days_norm": np.asarray(transition_days_value_norm, dtype=np.float32),
        }
        np.save(dst_dir / name, payload, allow_pickle=True)
        names.append(name)
    return names


def preprocess_dataset(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Build processed dataset split and write ``processed_meta.json``."""
    dataset_dir = resolve_path(config_path, cfg.paths.dataset_dir)
    processed_dir = resolve_path(config_path, cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = _list_raw_dataset_files(dataset_dir)
    fingerprint = _build_preprocess_fingerprint(cfg=cfg, files=files)
    meta_path = processed_dir / "processed_meta.json"
    if meta_path.is_file():
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            if _processed_cache_is_valid(meta=cached, fingerprint=fingerprint, processed_dir=processed_dir):
                LOGGER.info("Reusing cached processed dataset at %s", processed_dir)
                return cached
        except Exception:
            # Fallback to full rebuild if metadata is malformed or stale.
            pass

    if processed_dir.exists():
        for p in processed_dir.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    geometry_meta = _validate_raw_geometry(files, cfg=cfg)
    train_files, val_files, test_files = _split_files(
        files,
        seed=cfg.training.split_seed,
        val_fraction=cfg.training.val_fraction,
        test_fraction=cfg.training.test_fraction,
    )

    stats = _fit_stats_streaming(
        train_files=train_files,
        state_norm_cfg=cfg.normalization.state,
        param_norm_cfg=cfg.normalization.params,
    )
    transition_days_norm = _fit_transition_days_norm_stats(
        train_files=train_files,
        zscore_eps=float(cfg.normalization.params.eps),
    )
    fields = list(stats.field_names)
    param_names = list(stats.param_names)
    conditioning_names = list(param_names) + [TRANSITION_DAYS_CONDITIONING_NAME]
    const_param_names = [
        str(param_names[i]) for i, is_const in enumerate(np.asarray(stats.param_is_constant, dtype=bool)) if is_const
    ]
    const_conditioning_names = list(const_param_names)
    if bool(transition_days_norm["is_constant"]):
        const_conditioning_names.append(TRANSITION_DAYS_CONDITIONING_NAME)
    if const_param_names:
        LOGGER.warning(
            "Constant parameters detected and zeroed in normalization (%d/%d): %s",
            len(const_param_names),
            len(param_names),
            const_param_names,
        )
    if len(const_param_names) == len(param_names):
        LOGGER.info(
            "All user-facing parameters are constant in train split; transition model remains valid via state input."
        )
    if bool(transition_days_norm["is_constant"]):
        LOGGER.info(
            "transition_days is constant in train split (mean=%.6e days); transition conditioning feature is zero.",
            float(transition_days_norm["mean"]),
        )

    written_train: List[str] = []
    written_val: List[str] = []
    written_test: List[str] = []
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for fp in train_files:
        names = _write_processed_file(
            src_file=fp,
            dst_dir=train_dir,
            split_prefix=fp.stem,
            stats=stats,
            transition_days_norm=transition_days_norm,
        )
        written_train.extend(str(Path("train") / n) for n in names)
    for fp in val_files:
        names = _write_processed_file(
            src_file=fp,
            dst_dir=val_dir,
            split_prefix=fp.stem,
            stats=stats,
            transition_days_norm=transition_days_norm,
        )
        written_val.extend(str(Path("val") / n) for n in names)
    for fp in test_files:
        names = _write_processed_file(
            src_file=fp,
            dst_dir=test_dir,
            split_prefix=fp.stem,
            stats=stats,
            transition_days_norm=transition_days_norm,
        )
        written_test.extend(str(Path("test") / n) for n in names)

    z0 = _load_npy_payload_dict(processed_dir / written_train[0])
    st0 = np.asarray(z0["state_input_norm"], dtype=np.float32)

    meta: Dict[str, Any] = {
        "fields": fields,
        "param_names": param_names,
        "conditioning_names": conditioning_names,
        "shape": {"C": int(st0.shape[0]), "H": int(st0.shape[1]), "W": int(st0.shape[2])},
        "task": "trajectory_transition",
        "splits": {"train": written_train, "val": written_val, "test": written_test},
        "normalization": stats_to_json(stats),
        "constant_param_names": const_param_names,
        "constant_conditioning_names": const_conditioning_names,
        "transition_days_norm": transition_days_norm,
        "geometry": {
            "lat_order": str(geometry_meta["lat_order"]),
            "lon_origin": str(geometry_meta["lon_origin"]),
            "lon_shift": int(geometry_meta["lon_shift"]),
        },
        "solver": {
            "M": int(cfg.solver.M),
            "dt_seconds": float(cfg.solver.dt_seconds),
            "default_time_days": float(cfg.solver.default_time_days),
        },
        "model_time_mapping": {
            "default_time_days": float(cfg.solver.default_time_days),
            "transition_jump_steps": int(cfg.model.transition_jump_steps),
            "rollout_steps_at_default_time": int(cfg.model.rollout_steps_at_default_time),
        },
        "build_fingerprint": fingerprint,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TransitionProcessedDataset(Dataset):
    """PyTorch dataset wrapper over processed transition files."""

    def __init__(self, *, processed_dir: Path, file_names: Sequence[str]) -> None:
        self.processed_dir = processed_dir
        self.file_names = list(file_names)

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        fp = self.processed_dir / self.file_names[idx]
        if fp.suffix.lower() == ".npy":
            z = _load_npy_payload_dict(fp)
            state_in = np.asarray(z["state_input_norm"], dtype=np.float32)
            state_out = np.asarray(z["state_target_norm"], dtype=np.float32)
            if "conditioning_norm" in z:
                conditioning = np.asarray(z["conditioning_norm"], dtype=np.float32)
            else:
                conditioning = np.asarray(z["params_norm"], dtype=np.float32)
            transition_days = float(np.asarray(z["transition_days"]).item())
        else:
            with np.load(fp, allow_pickle=True) as z:
                state_in = np.asarray(z["state_input_norm"], dtype=np.float32)
                state_out = np.asarray(z["state_target_norm"], dtype=np.float32)
                if "conditioning_norm" in z:
                    conditioning = np.asarray(z["conditioning_norm"], dtype=np.float32)
                else:
                    conditioning = np.asarray(z["params_norm"], dtype=np.float32)
                transition_days = float(np.asarray(z["transition_days"]).item())
        return (
            torch.from_numpy(conditioning),
            torch.from_numpy(state_in),
            torch.from_numpy(state_out),
            torch.tensor(transition_days, dtype=torch.float32),
        )


class PreloadedTransitionDataset(Dataset):
    """Dataset backed by already-loaded transition tensors (typically on GPU)."""

    def __init__(
        self,
        *,
        conditioning: torch.Tensor,
        state_inputs: torch.Tensor,
        state_targets: torch.Tensor,
        transition_days: torch.Tensor,
    ) -> None:
        if conditioning.ndim != 2:
            raise ValueError(f"conditioning must have shape [N,P], got {tuple(conditioning.shape)}")
        if state_inputs.ndim != 4 or state_targets.ndim != 4:
            raise ValueError(
                "state_inputs/state_targets must have shape [N,C,H,W], got "
                f"{tuple(state_inputs.shape)} and {tuple(state_targets.shape)}"
            )
        if transition_days.ndim != 1:
            raise ValueError(f"transition_days must have shape [N], got {tuple(transition_days.shape)}")
        n = int(conditioning.shape[0])
        if int(state_inputs.shape[0]) != n or int(state_targets.shape[0]) != n or int(transition_days.shape[0]) != n:
            raise ValueError(
                "Preloaded tensor batch sizes must match: "
                f"conditioning={tuple(conditioning.shape)}, state_inputs={tuple(state_inputs.shape)}, "
                f"state_targets={tuple(state_targets.shape)}, transition_days={tuple(transition_days.shape)}"
            )
        self.conditioning = conditioning
        self.state_inputs = state_inputs
        self.state_targets = state_targets
        self.transition_days = transition_days

    def __len__(self) -> int:
        return int(self.conditioning.shape[0])

    def __getitem__(self, idx: int):
        return self.conditioning[idx], self.state_inputs[idx], self.state_targets[idx], self.transition_days[idx]


def _preload_split_to_device(
    *,
    processed_dir: Path,
    file_names: Sequence[str],
    device: torch.device,
) -> PreloadedTransitionDataset:
    """Load one split into contiguous tensors and move them to ``device``."""
    conditioning_rows: List[np.ndarray] = []
    state_input_rows: List[np.ndarray] = []
    state_target_rows: List[np.ndarray] = []
    transition_rows: List[float] = []
    for name in file_names:
        fp = processed_dir / str(name)
        if fp.suffix.lower() == ".npy":
            z = _load_npy_payload_dict(fp)
            state_input_rows.append(np.asarray(z["state_input_norm"], dtype=np.float32))
            state_target_rows.append(np.asarray(z["state_target_norm"], dtype=np.float32))
            if "conditioning_norm" in z:
                conditioning_rows.append(np.asarray(z["conditioning_norm"], dtype=np.float32))
            else:
                conditioning_rows.append(np.asarray(z["params_norm"], dtype=np.float32))
            transition_rows.append(float(np.asarray(z["transition_days"]).item()))
        else:
            with np.load(fp, allow_pickle=True) as z:
                state_input_rows.append(np.asarray(z["state_input_norm"], dtype=np.float32))
                state_target_rows.append(np.asarray(z["state_target_norm"], dtype=np.float32))
                if "conditioning_norm" in z:
                    conditioning_rows.append(np.asarray(z["conditioning_norm"], dtype=np.float32))
                else:
                    conditioning_rows.append(np.asarray(z["params_norm"], dtype=np.float32))
                transition_rows.append(float(np.asarray(z["transition_days"]).item()))

    conditioning_t = torch.from_numpy(np.stack(conditioning_rows, axis=0)).to(device=device)
    state_inputs_t = torch.from_numpy(np.stack(state_input_rows, axis=0)).to(device=device)
    state_targets_t = torch.from_numpy(np.stack(state_target_rows, axis=0)).to(device=device)
    transition_t = torch.tensor(transition_rows, dtype=torch.float32, device=device)
    return PreloadedTransitionDataset(
        conditioning=conditioning_t,
        state_inputs=state_inputs_t,
        state_targets=state_targets_t,
        transition_days=transition_t,
    )


def _collect_validation_predictions(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run transition model on a split loader and collect predictions/targets."""
    preds: List[np.ndarray] = []
    tars: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for conditioning_batch, state_input_batch, state_target_batch, _transition_days_batch in loader:
            conditioning_batch = conditioning_batch.to(device=device)
            state_input_batch = state_input_batch.to(device=device)
            state_target_batch = state_target_batch.to(device=device)
            with autocast_context(device, amp_mode):
                pred_batch = model(state_input_batch, conditioning_batch, steps=1)
            preds.append(pred_batch.detach().cpu().numpy())
            tars.append(state_target_batch.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(tars, axis=0)


def _named_scalar_map(values: np.ndarray, field_names: Sequence[str]) -> Dict[str, float]:
    """Map channel-aligned scalar array to ``{field_name: value}``."""
    return {str(field_names[i]): float(values[i]) for i in range(len(field_names))}


def _spectral_relative_power_rmse(
    *,
    pred_phys: np.ndarray,
    tar_phys: np.ndarray,
    field_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute per-channel relative RMSE between mean 2D power spectra."""
    pred_fft = np.fft.rfft2(pred_phys.astype(np.float64, copy=False), axes=(-2, -1))
    tar_fft = np.fft.rfft2(tar_phys.astype(np.float64, copy=False), axes=(-2, -1))
    pred_power = np.mean(np.abs(pred_fft) ** 2, axis=0)  # [C,H,Wf]
    tar_power = np.mean(np.abs(tar_fft) ** 2, axis=0)  # [C,H,Wf]
    per_channel_abs = np.sqrt(np.mean((pred_power - tar_power) ** 2, axis=(1, 2)))
    per_channel_den = np.sqrt(np.mean(tar_power**2, axis=(1, 2))) + STD_FLOOR
    per_channel_rel = per_channel_abs / per_channel_den
    return {
        "mean_relative_power_rmse": float(np.mean(per_channel_rel)),
        "per_channel_relative_power_rmse": _named_scalar_map(per_channel_rel, field_names),
    }


def _physics_summary_metrics(*, pred_phys: np.ndarray, tar_phys: np.ndarray, field_names: Sequence[str]) -> Dict[str, float]:
    """Compute lightweight physical-summary diagnostics from denormalized fields."""
    metrics: Dict[str, float] = {}
    field_to_idx = {str(name): i for i, name in enumerate(field_names)}

    if "Phi" in field_to_idx:
        phi_idx = field_to_idx["Phi"]
        phi_diff = pred_phys[:, phi_idx] - tar_phys[:, phi_idx]
        metrics["phi_mean_bias"] = float(np.mean(phi_diff))
        metrics["phi_rmse"] = float(np.sqrt(np.mean(phi_diff**2)))

    if "U" in field_to_idx and "V" in field_to_idx:
        u_idx = field_to_idx["U"]
        v_idx = field_to_idx["V"]
        pred_ke = 0.5 * (pred_phys[:, u_idx] ** 2 + pred_phys[:, v_idx] ** 2)
        tar_ke = 0.5 * (tar_phys[:, u_idx] ** 2 + tar_phys[:, v_idx] ** 2)
        mean_pred_ke = float(np.mean(pred_ke))
        mean_tar_ke = float(np.mean(tar_ke))
        metrics["kinetic_energy_mean_rel_error"] = float(
            abs(mean_pred_ke - mean_tar_ke) / (abs(mean_tar_ke) + STD_FLOOR)
        )

    if "eta" in field_to_idx:
        eta_idx = field_to_idx["eta"]
        pred_enstrophy = 0.5 * pred_phys[:, eta_idx] ** 2
        tar_enstrophy = 0.5 * tar_phys[:, eta_idx] ** 2
        mean_pred_enstrophy = float(np.mean(pred_enstrophy))
        mean_tar_enstrophy = float(np.mean(tar_enstrophy))
        metrics["enstrophy_like_mean_rel_error"] = float(
            abs(mean_pred_enstrophy - mean_tar_enstrophy) / (abs(mean_tar_enstrophy) + STD_FLOOR)
        )

    if "delta" in field_to_idx:
        delta_idx = field_to_idx["delta"]
        pred_div_activity = pred_phys[:, delta_idx] ** 2
        tar_div_activity = tar_phys[:, delta_idx] ** 2
        mean_pred_div_activity = float(np.mean(pred_div_activity))
        mean_tar_div_activity = float(np.mean(tar_div_activity))
        metrics["divergence_activity_mean_rel_error"] = float(
            abs(mean_pred_div_activity - mean_tar_div_activity) / (abs(mean_tar_div_activity) + STD_FLOOR)
        )

    return metrics


def _compute_gate_metrics(
    *,
    pred_norm: np.ndarray,
    tar_norm: np.ndarray,
    field_names: Sequence[str],
    stats: NormalizationStats,
) -> Dict[str, Any]:
    """Compute normalized and physical-space transition evaluation metrics."""
    diff = pred_norm - tar_norm
    global_rmse_norm = float(np.sqrt(np.mean(diff**2)))
    per_channel_rmse_norm = np.sqrt(np.mean(diff**2, axis=(0, 2, 3)))

    pred_phys = denormalize_states(pred_norm, stats=stats).astype(np.float64, copy=False)
    tar_phys = denormalize_states(tar_norm, stats=stats).astype(np.float64, copy=False)
    diff_phys = pred_phys - tar_phys
    global_rmse_phys = float(np.sqrt(np.mean(diff_phys**2)))
    per_channel_rmse_phys = np.sqrt(np.mean(diff_phys**2, axis=(0, 2, 3)))
    per_channel_mae_phys = np.mean(np.abs(diff_phys), axis=(0, 2, 3))

    spectral_metrics = _spectral_relative_power_rmse(
        pred_phys=pred_phys,
        tar_phys=tar_phys,
        field_names=field_names,
    )
    physics_summary = _physics_summary_metrics(
        pred_phys=pred_phys,
        tar_phys=tar_phys,
        field_names=field_names,
    )

    return {
        "normalized": {
            "global_rmse": global_rmse_norm,
            "per_channel_rmse": _named_scalar_map(per_channel_rmse_norm, field_names),
        },
        "physical": {
            "global_rmse": global_rmse_phys,
            "per_channel_rmse": _named_scalar_map(per_channel_rmse_phys, field_names),
            "per_channel_mae": _named_scalar_map(per_channel_mae_phys, field_names),
        },
        "spectral": spectral_metrics,
        "physics_summary": physics_summary,
    }


def _write_training_history_csv(*, history: Sequence[Dict[str, float]], csv_path: Path) -> None:
    """Write per-epoch training history rows to CSV."""
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(round(float(row["epoch"]))),
                    "train_loss": f"{float(row['train_loss']):.8e}",
                    "val_loss": f"{float(row['val_loss']):.8e}",
                    "lr": f"{float(row['lr']):.8e}",
                }
            )


def _check_finite_tensor(t: torch.Tensor, *, name: str) -> None:
    """Raise if tensor contains NaN/Inf values."""
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} contains non-finite values")


def _should_run_finite_check(step_idx: int) -> bool:
    """Return True when finite-value checks should run for this batch index."""
    idx = int(step_idx)
    return idx == 0 or (FINITE_CHECK_INTERVAL > 0 and (idx % FINITE_CHECK_INTERVAL) == 0)


def _abs_max_tensor(t: torch.Tensor) -> float:
    """Return absolute max value from tensor as Python float."""
    return float(t.detach().abs().max().item())


def _fmt_sci(value: float, *, width: int = 11, sig_figs: int = 3) -> str:
    """Format scalar in compact scientific notation with fixed field width."""
    if sig_figs < 1:
        raise ValueError(f"sig_figs must be >= 1, got {sig_figs}")
    decimals = max(0, int(sig_figs) - 1)
    return f"{float(value):{int(width)}.{decimals}e}"


def _is_power_of_two(v: int) -> bool:
    """Return ``True`` when ``v`` is a positive power of two."""
    iv = int(v)
    return iv > 0 and (iv & (iv - 1)) == 0


def _resolve_runtime_amp_mode(*, requested_amp_mode: str, nlat: int, nlon: int) -> str:
    """Resolve runtime AMP mode with shape safety checks for harmonic layers."""
    mode = str(requested_amp_mode).lower()
    if mode in {"bf16", "fp16"} and (not _is_power_of_two(nlat) or not _is_power_of_two(nlon)):
        LOGGER.warning(
            "Disabling AMP (%s) for grid %dx%d because transformed dimensions are not powers of two.",
            mode,
            int(nlat),
            int(nlon),
        )
        return "none"
    return mode


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all optimizer parameter groups."""
    lr_val = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr_val


def _cosine_warmup_lr(
    *,
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    """Compute cosine schedule with optional linear warmup."""
    if total_epochs < 1:
        raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0, got {base_lr}")
    if min_lr < 0:
        raise ValueError(f"min_lr must be >= 0, got {min_lr}")
    if min_lr > base_lr:
        raise ValueError(f"min_lr must be <= base_lr, got min_lr={min_lr}, base_lr={base_lr}")

    warmup = min(int(warmup_epochs), int(total_epochs))
    e = max(1, int(epoch))
    if warmup > 0 and e <= warmup:
        return float(base_lr) * (float(e) / float(warmup))

    if total_epochs == warmup:
        return float(base_lr)

    steps_after_warmup = int(total_epochs - warmup)
    denom = max(1, steps_after_warmup - 1)
    progress = float(e - warmup - 1) / float(denom)
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def _warn_if_raw_dataset_count_mismatch(cfg: GCMulatorConfig, *, config_path: Path) -> None:
    """Warn when fewer raw files exist than ``sampling.n_sims`` expects."""
    dataset_dir = resolve_path(config_path, cfg.paths.dataset_dir)
    raw_files = sorted(list(dataset_dir.glob("sim_*.npy")) + list(dataset_dir.glob("sim_*.npz")))
    if not raw_files:
        return

    n_found = len(raw_files)
    n_expected = int(cfg.sampling.n_sims)
    if n_found < n_expected:
        LOGGER.warning(
            "Raw dataset has %4d files but config.sampling.n_sims=%4d. "
            "Training will proceed on existing files only.",
            n_found,
            n_expected,
        )


def train_emulator(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Train transition rollout emulator end-to-end and persist artifacts."""
    ensure_torch_harmonics_importable(config_path.parent)
    _warn_if_raw_dataset_count_mismatch(cfg, config_path=config_path)

    processed_meta = preprocess_dataset(cfg, config_path=config_path)
    if str(processed_meta.get("task", "")) != "trajectory_transition":
        raise RuntimeError(
            "Processed dataset task is not trajectory_transition. "
            "Regenerate/preprocess dataset with current trajectory pipeline."
        )
    processed_dir = resolve_path(config_path, cfg.paths.processed_dir)
    model_dir = resolve_path(config_path, cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_dict = asdict(cfg)

    # Persist exact run configuration for reproducibility and future inspection.
    resolved_cfg_path = model_dir / "config_used.resolved.json"
    resolved_cfg_path.write_text(json.dumps(resolved_cfg_dict, indent=2), encoding="utf-8")

    cfg_suffix = config_path.suffix.lower()
    if cfg_suffix not in {".json", ".yaml", ".yml"}:
        cfg_suffix = ".txt"
    original_cfg_path = model_dir / f"config_used.original{cfg_suffix}"
    original_cfg_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    stats = stats_from_json(processed_meta["normalization"])
    field_names = list(processed_meta["fields"])
    processed_geometry = dict(processed_meta.get("geometry", {}))
    lat_order = str(processed_geometry.get("lat_order", "north_to_south"))
    lon_origin = str(processed_geometry.get("lon_origin", "0_to_2pi"))

    device = choose_device(cfg.training.device)
    preload_to_gpu = bool(cfg.training.preload_to_gpu)
    if preload_to_gpu and device.type != "cuda":
        raise RuntimeError(
            "training.preload_to_gpu=true requires CUDA device. "
            f"Resolved device is '{device}'."
        )

    if preload_to_gpu:
        train_ds = _preload_split_to_device(
            processed_dir=processed_dir,
            file_names=processed_meta["splits"]["train"],
            device=device,
        )
        val_ds = _preload_split_to_device(
            processed_dir=processed_dir,
            file_names=processed_meta["splits"]["val"],
            device=device,
        )
        test_ds = _preload_split_to_device(
            processed_dir=processed_dir,
            file_names=processed_meta["splits"]["test"],
            device=device,
        )
        LOGGER.info(
            "Preloaded processed splits to %s | n_train=%d | n_val=%d | n_test=%d",
            device,
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )
    else:
        train_ds = TransitionProcessedDataset(processed_dir=processed_dir, file_names=processed_meta["splits"]["train"])
        val_ds = TransitionProcessedDataset(processed_dir=processed_dir, file_names=processed_meta["splits"]["val"])
        test_ds = TransitionProcessedDataset(processed_dir=processed_dir, file_names=processed_meta["splits"]["test"])

    if len(train_ds) < int(cfg.training.batch_size):
        raise ValueError(
            "Training split size is smaller than batch_size while training loader uses drop_last=True: "
            f"n_train={len(train_ds)}, batch_size={cfg.training.batch_size}"
        )

    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    if preload_to_gpu:
        loader_common: Dict[str, Any] = {
            "num_workers": 0,
            "pin_memory": False,
        }
    else:
        loader_common = {
            "num_workers": cfg.training.num_workers,
            "pin_memory": cfg.training.pin_memory,
        }
        if int(cfg.training.num_workers) > 0:
            # Keep workers alive across epochs and prefetch small batches to reduce loader stalls.
            loader_common["persistent_workers"] = True
            loader_common["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        **loader_common,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        **loader_common,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        **loader_common,
        drop_last=False,
    )

    use_non_blocking = bool(cfg.training.pin_memory) and device.type == "cuda" and not preload_to_gpu

    sample_conditioning, sample_state_in, _sample_state_out, _sample_dt_days = train_ds[0]
    state_chans = int(sample_state_in.shape[0])
    h = int(sample_state_in.shape[1])
    w = int(sample_state_in.shape[2])
    conditioning_dim = int(sample_conditioning.shape[0])

    model = build_state_conditioned_rollout_model(
        img_size=(h, w),
        state_chans=state_chans,
        param_dim=conditioning_dim,
        cfg_model=cfg.model,
        lat_order=lat_order,
        lon_origin=lon_origin,
    ).to(device)
    runtime_amp_mode = _resolve_runtime_amp_mode(
        requested_amp_mode=cfg.training.amp_mode,
        nlat=h,
        nlon=w,
    )

    loss_fn = SphereLoss(nlat=h, nlon=w, grid=cfg.model.grid).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler_type = str(cfg.training.scheduler.type)
    base_learning_rate = float(cfg.training.learning_rate)

    scheduler = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.training.scheduler.factor,
            patience=cfg.training.scheduler.patience,
            min_lr=cfg.training.scheduler.min_lr,
        )

    scaler = None
    if runtime_amp_mode == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    epochs = int(cfg.training.epochs)
    best_val = float("inf")

    best_path = model_dir / "best.pt"
    last_path = model_dir / "last.pt"

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        if scheduler_type == "cosine_warmup":
            lr_epoch = _cosine_warmup_lr(
                epoch=epoch,
                total_epochs=epochs,
                base_lr=base_learning_rate,
                min_lr=float(cfg.training.scheduler.min_lr),
                warmup_epochs=int(cfg.training.scheduler.warmup_epochs),
            )
            _set_optimizer_lr(optimizer, lr_epoch)

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for conditioning_batch, state_input_batch, state_target_batch, transition_days_batch in train_loader:
            conditioning_batch = conditioning_batch.to(device=device, non_blocking=use_non_blocking)
            state_input_batch = state_input_batch.to(device=device, non_blocking=use_non_blocking)
            state_target_batch = state_target_batch.to(device=device, non_blocking=use_non_blocking)
            transition_days_batch = transition_days_batch.to(device=device, non_blocking=use_non_blocking)
            do_finite_check = _should_run_finite_check(train_count)
            if do_finite_check:
                _check_finite_tensor(conditioning_batch, name="train conditioning batch")
                _check_finite_tensor(state_input_batch, name="train state_input batch")
                _check_finite_tensor(state_target_batch, name="train target batch")
                _check_finite_tensor(transition_days_batch, name="train transition_days batch")

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, runtime_amp_mode):
                yhat = model(state_input_batch, conditioning_batch, steps=1)
                if do_finite_check:
                    _check_finite_tensor(yhat, name="train prediction batch")
                loss = loss_fn(yhat, state_target_batch)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite")

            if epoch == 1 and train_count == 0:
                conditioning_abs_max = _abs_max_tensor(conditioning_batch)
                input_abs_max = _abs_max_tensor(state_input_batch)
                target_abs_max = _abs_max_tensor(state_target_batch)
                pred_abs_max = _abs_max_tensor(yhat)
                LOGGER.info(
                    "ScaleCheck | conditioning_abs_max=%s | input_abs_max=%s | target_abs_max=%s | pred_abs_max=%s",
                    _fmt_sci(conditioning_abs_max),
                    _fmt_sci(input_abs_max),
                    _fmt_sci(target_abs_max),
                    _fmt_sci(pred_abs_max),
                )
                if pred_abs_max > 1.0e8 and target_abs_max < 1.0e4:
                    raise RuntimeError(
                        "Detected exploding first-batch predictions in normalized space. "
                        f"pred_abs_max={pred_abs_max:.3e}, target_abs_max={target_abs_max:.3e}. "
                        "Consider reducing model.rollout_steps_at_default_time and/or reducing "
                        "model.residual_init_scale (or disabling residual_prediction)."
                    )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_sum += float(loss.detach().item())
            train_count += 1

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for conditioning_batch, state_input_batch, state_target_batch, transition_days_batch in val_loader:
                conditioning_batch = conditioning_batch.to(device=device, non_blocking=use_non_blocking)
                state_input_batch = state_input_batch.to(device=device, non_blocking=use_non_blocking)
                state_target_batch = state_target_batch.to(device=device, non_blocking=use_non_blocking)
                transition_days_batch = transition_days_batch.to(device=device, non_blocking=use_non_blocking)
                do_finite_check = _should_run_finite_check(val_count)
                with autocast_context(device, runtime_amp_mode):
                    yhat = model(state_input_batch, conditioning_batch, steps=1)
                    if do_finite_check:
                        _check_finite_tensor(yhat, name="val prediction batch")
                        _check_finite_tensor(transition_days_batch, name="val transition_days batch")
                    vloss = loss_fn(yhat, state_target_batch)
                if not torch.isfinite(vloss).item():
                    raise RuntimeError("Validation loss became non-finite")
                val_loss_sum += float(vloss.detach().item())
                val_count += 1

        if train_count == 0:
            raise RuntimeError("No training batches produced. Check batch_size vs dataset size.")
        if val_count == 0:
            raise RuntimeError("No validation batches produced. Check val_fraction vs dataset size.")
        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count

        if scheduler is not None:
            scheduler.step(val_loss)

        lr_now = float(optimizer.param_groups[0]["lr"])
        LOGGER.info(
            "Epoch %4d/%4d | train=%s | val=%s | lr=%s",
            epoch,
            epochs,
            _fmt_sci(train_loss),
            _fmt_sci(val_loss),
            _fmt_sci(lr_now, width=10, sig_figs=2),
        )

        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})

        ckpt = {
            "mode": "state_conditioned_trajectory_transition",
            "model_state": model.state_dict(),
            "fields": field_names,
            "param_names": list(processed_meta["param_names"]),
            "conditioning_names": list(
                processed_meta.get("conditioning_names", processed_meta["param_names"])
            ),
            "shape": {"C": state_chans, "H": h, "W": w},
            "geometry": {
                "lat_order": lat_order,
                "lon_origin": lon_origin,
            },
            "normalization": stats_to_json(stats),
            "transition_days_norm": dict(processed_meta.get("transition_days_norm", {})),
            "solver": asdict(cfg.solver),
            "model_config": asdict(cfg.model),
            "training_config": asdict(cfg.training),
            "runtime_amp_mode": runtime_amp_mode,
            "resolved_config": resolved_cfg_dict,
            "source_config_path": str(config_path),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "learning_rate": float(lr_now),
        }

        torch.save(ckpt, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_path)

    # Validation/test metrics on best checkpoint
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    pred_norm, tar_norm = _collect_validation_predictions(
        model=model,
        loader=val_loader,
        device=device,
        amp_mode=runtime_amp_mode,
    )

    val_metrics = _compute_gate_metrics(
        pred_norm=pred_norm,
        tar_norm=tar_norm,
        field_names=field_names,
        stats=stats,
    )
    test_pred_norm, test_tar_norm = _collect_validation_predictions(
        model=model,
        loader=test_loader,
        device=device,
        amp_mode=runtime_amp_mode,
    )
    test_metrics = _compute_gate_metrics(
        pred_norm=test_pred_norm,
        tar_norm=test_tar_norm,
        field_names=field_names,
        stats=stats,
    )

    (model_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _write_training_history_csv(history=history, csv_path=model_dir / "training_history.csv")
    (model_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (model_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_val_loss": float(best_val),
        "history_path": str(model_dir / "training_history.json"),
        "history_csv_path": str(model_dir / "training_history.csv"),
        "val_metrics_path": str(model_dir / "val_metrics.json"),
        "test_metrics_path": str(model_dir / "test_metrics.json"),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "processed_meta": str(processed_dir / "processed_meta.json"),
        "resolved_config_path": str(resolved_cfg_path),
        "original_config_copy_path": str(original_cfg_path),
        "runtime_amp_mode": runtime_amp_mode,
    }
    return summary
