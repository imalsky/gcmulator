"""Preprocessing and training pipeline for direct-jump visible-state transition learning.

This module covers three stages:

1. validate raw trajectory-window files
2. preprocess them into split-level normalized shards
3. train and evaluate the direct-jump transition emulator
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict
import csv
import json
import logging
import math
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    CONDITIONING_PARAM_NAMES,
    GCMulatorConfig,
    PHYSICAL_STATE_FIELDS,
    PROGNOSTIC_TARGET_FIELDS,
    TRANSITION_TIME_NAME,
    resolve_path,
)
from geometry import geometry_shift_for_nlon
from modeling import (
    SphereLoss,
    autocast_context,
    build_state_conditioned_transition_model,
    choose_device,
    ensure_torch_harmonics_importable,
)
from normalization import (
    NormalizationStats,
    ParamNormalizationStats,
    STD_FLOOR,
    StateNormalizationStats,
    apply_state_transforms,
    denormalize_state_tensor,
    normalize_conditioning,
    normalize_params,
    normalize_state_tensor,
    stats_from_json,
    stats_to_json,
    subset_state_stats,
)


LOGGER = logging.getLogger("train")

PREPROCESS_FINGERPRINT_VERSION = 13
RAW_REQUIRED_KEYS = (
    "state_inputs",
    "state_targets",
    "transition_days",
    "anchor_steps",
    "input_fields",
    "target_fields",
    "params",
    "param_names",
    "default_time_days",
    "burn_in_days",
    "dt_seconds",
    "starttime_index",
    "transition_jump_days_min",
    "transition_jump_days_max",
    "n_transitions",
    "M",
    "nlat",
    "nlon",
    "lat_order",
    "lon_origin",
    "lon_shift",
)
EARLY_STOPPING_PATIENCE_MULTIPLIER = 3


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


def _expected_geometry(cfg: GCMulatorConfig, *, nlon: int) -> Dict[str, Any]:
    """Return the expected geometry metadata for a grid width."""
    return {
        "lat_order": (
            "north_to_south"
            if cfg.geometry.flip_latitude_to_north_south
            else "south_to_north"
        ),
        "lon_origin": "0_to_2pi" if cfg.geometry.roll_longitude_to_0_2pi else "minus_pi_to_pi",
        "lon_shift": int(geometry_shift_for_nlon(int(nlon), cfg.geometry.roll_longitude_to_0_2pi)),
    }


def _list_raw_dataset_files(dataset_dir: Path) -> List[Path]:
    """List raw simulation files and reject unsupported legacy leftovers."""
    legacy_files = sorted(dataset_dir.glob("sim_*.npz"))
    if legacy_files:
        raise RuntimeError(
            "Unsupported legacy raw files were found in the dataset directory. "
            "Remove "
            f"{len(legacy_files)} sim_*.npz files or regenerate with "
            f"overwrite_dataset=true: {dataset_dir}"
        )
    files = sorted(dataset_dir.glob("sim_*.npy"))
    if files:
        return files
    raise FileNotFoundError(f"No sim_*.npy files found in {dataset_dir}")


def _validated_raw_payload(file_path: Path, *, cfg: GCMulatorConfig) -> Dict[str, Any]:
    """Load and validate one raw transition payload against the config contract."""
    payload = _load_npy_payload_dict(file_path)
    missing = [key for key in RAW_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Raw file {file_path} is missing required keys: {missing}")

    input_fields = [
        str(value)
        for value in np.asarray(payload["input_fields"], dtype=object).tolist()
    ]
    target_fields = [
        str(value)
        for value in np.asarray(payload["target_fields"], dtype=object).tolist()
    ]
    if tuple(input_fields) != tuple(PHYSICAL_STATE_FIELDS):
        raise ValueError(
            f"Raw file {file_path} has input_fields={input_fields}, "
            f"expected {list(PHYSICAL_STATE_FIELDS)}"
        )
    if tuple(target_fields) != tuple(PROGNOSTIC_TARGET_FIELDS):
        raise ValueError(
            f"Raw file {file_path} has target_fields={target_fields}, "
            f"expected {list(PROGNOSTIC_TARGET_FIELDS)}"
        )

    param_names = [
        str(value)
        for value in np.asarray(payload["param_names"], dtype=object).tolist()
    ]
    if tuple(param_names) != tuple(CONDITIONING_PARAM_NAMES):
        raise ValueError(
            f"Raw file {file_path} has param_names={param_names}, "
            f"expected {list(CONDITIONING_PARAM_NAMES)}"
        )

    state_inputs = np.asarray(payload["state_inputs"], dtype=np.float32)
    state_targets = np.asarray(payload["state_targets"], dtype=np.float32)
    transition_days = np.asarray(payload["transition_days"], dtype=np.float64)
    anchor_steps = np.asarray(payload["anchor_steps"], dtype=np.int64)
    params = np.asarray(payload["params"], dtype=np.float64)
    default_time_days = float(np.asarray(payload["default_time_days"], dtype=np.float64).item())
    burn_in_days = float(np.asarray(payload["burn_in_days"], dtype=np.float64).item())
    dt_seconds = float(np.asarray(payload["dt_seconds"], dtype=np.float64).item())
    starttime_index = int(np.asarray(payload["starttime_index"], dtype=np.int64).item())
    transition_jump_days_min = float(
        np.asarray(payload["transition_jump_days_min"], dtype=np.float64).item()
    )
    transition_jump_days_max = float(
        np.asarray(payload["transition_jump_days_max"], dtype=np.float64).item()
    )
    n_transitions = int(np.asarray(payload["n_transitions"], dtype=np.int64).item())
    M = int(np.asarray(payload["M"], dtype=np.int64).item())
    nlat = int(np.asarray(payload["nlat"], dtype=np.int64).item())
    nlon = int(np.asarray(payload["nlon"], dtype=np.int64).item())
    lat_order = str(np.asarray(payload["lat_order"], dtype=object).item())
    lon_origin = str(np.asarray(payload["lon_origin"], dtype=object).item())
    lon_shift = int(np.asarray(payload["lon_shift"], dtype=np.int64).item())

    if state_inputs.ndim != 4 or state_targets.ndim != 4:
        raise ValueError(f"Raw states in {file_path} must be [T,C,H,W]")
    if state_inputs.shape[0] != state_targets.shape[0]:
        raise ValueError(f"Raw transition count mismatch in {file_path}")
    if state_inputs.shape[1] != len(PHYSICAL_STATE_FIELDS):
        raise ValueError(f"Input channel count mismatch in {file_path}")
    if state_targets.shape[1] != len(PROGNOSTIC_TARGET_FIELDS):
        raise ValueError(f"Target channel count mismatch in {file_path}")
    if transition_days.shape != (state_inputs.shape[0],):
        raise ValueError(f"transition_days shape mismatch in {file_path}")
    if anchor_steps.shape != (state_inputs.shape[0],):
        raise ValueError(f"anchor_steps shape mismatch in {file_path}")
    if params.shape != (len(CONDITIONING_PARAM_NAMES),):
        raise ValueError(f"params shape mismatch in {file_path}: {params.shape}")
    if int(state_inputs.shape[0]) != n_transitions:
        raise ValueError(f"n_transitions mismatch in {file_path}")
    if int(state_inputs.shape[2]) != nlat or int(state_inputs.shape[3]) != nlon:
        raise ValueError(f"Input spatial metadata mismatch in {file_path}")
    if int(state_targets.shape[2]) != nlat or int(state_targets.shape[3]) != nlon:
        raise ValueError(f"Target spatial metadata mismatch in {file_path}")

    if M != int(cfg.solver.M):
        raise ValueError(f"Raw file {file_path} has M={M}, expected {cfg.solver.M}")
    if not math.isclose(dt_seconds, float(cfg.solver.dt_seconds), rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            f"Raw file {file_path} has dt_seconds={dt_seconds}, expected {cfg.solver.dt_seconds}"
        )
    if not math.isclose(
        default_time_days,
        float(cfg.solver.default_time_days),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"Raw file {file_path} has default_time_days={default_time_days}, "
            f"expected {cfg.solver.default_time_days}"
        )
    if not math.isclose(
        burn_in_days,
        float(cfg.sampling.burn_in_days),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"Raw file {file_path} has burn_in_days={burn_in_days}, "
            f"expected {cfg.sampling.burn_in_days}"
        )
    if starttime_index != int(cfg.solver.starttime_index):
        raise ValueError(
            f"Raw file {file_path} has starttime_index={starttime_index}, "
            f"expected {cfg.solver.starttime_index}"
        )
    if not math.isclose(
        transition_jump_days_min,
        float(cfg.sampling.transition_jump_days_min),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"Raw file {file_path} has transition_jump_days_min={transition_jump_days_min}, "
            f"expected {cfg.sampling.transition_jump_days_min}"
        )
    if not math.isclose(
        transition_jump_days_max,
        float(cfg.sampling.transition_jump_days_max),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"Raw file {file_path} has transition_jump_days_max={transition_jump_days_max}, "
            f"expected {cfg.sampling.transition_jump_days_max}"
        )
    if n_transitions != int(cfg.sampling.transitions_per_simulation):
        raise ValueError(
            f"Raw file {file_path} has n_transitions={n_transitions}, "
            f"expected {cfg.sampling.transitions_per_simulation}"
        )
    if np.any(transition_days <= 0.0):
        raise ValueError(f"Raw file {file_path} contains non-positive transition_days")

    expected_geometry = _expected_geometry(cfg, nlon=nlon)
    if lat_order != expected_geometry["lat_order"] or lon_origin != expected_geometry["lon_origin"]:
        raise ValueError(
            f"Geometry mismatch in {file_path}: "
            f"lat_order={lat_order}, lon_origin={lon_origin}, expected {expected_geometry}"
        )
    if lon_shift != int(expected_geometry["lon_shift"]):
        raise ValueError(
            f"Geometry mismatch in {file_path}: lon_shift={lon_shift}, "
            f"expected {expected_geometry['lon_shift']}"
        )

    return {
        "state_inputs": state_inputs,
        "state_targets": state_targets,
        "transition_days": transition_days,
        "anchor_steps": anchor_steps,
        "input_fields": input_fields,
        "target_fields": target_fields,
        "params": params,
        "param_names": param_names,
        "default_time_days": default_time_days,
        "burn_in_days": burn_in_days,
        "dt_seconds": dt_seconds,
        "starttime_index": starttime_index,
        "transition_jump_days_min": transition_jump_days_min,
        "transition_jump_days_max": transition_jump_days_max,
        "n_transitions": n_transitions,
        "M": M,
        "nlat": nlat,
        "nlon": nlon,
        "lat_order": lat_order,
        "lon_origin": lon_origin,
        "lon_shift": lon_shift,
    }


def _raw_file_signature(file_path: Path, *, cfg: GCMulatorConfig) -> Dict[str, Any]:
    """Return a validated signature used to detect raw-dataset changes."""
    metadata = _validated_raw_payload(file_path, cfg=cfg)
    stat = file_path.stat()
    return {
        "name": file_path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "input_fields": list(metadata["input_fields"]),
        "target_fields": list(metadata["target_fields"]),
        "param_names": list(metadata["param_names"]),
        "default_time_days": float(metadata["default_time_days"]),
        "burn_in_days": float(metadata["burn_in_days"]),
        "dt_seconds": float(metadata["dt_seconds"]),
        "starttime_index": int(metadata["starttime_index"]),
        "transition_jump_days_min": float(metadata["transition_jump_days_min"]),
        "transition_jump_days_max": float(metadata["transition_jump_days_max"]),
        "n_transitions": int(metadata["n_transitions"]),
        "nlat": int(metadata["nlat"]),
        "nlon": int(metadata["nlon"]),
        "lat_order": str(metadata["lat_order"]),
        "lon_origin": str(metadata["lon_origin"]),
        "lon_shift": int(metadata["lon_shift"]),
    }


def _build_preprocess_fingerprint(*, cfg: GCMulatorConfig, files: Sequence[Path]) -> Dict[str, Any]:
    """Build reproducibility fingerprint for processed-data cache reuse."""
    sampling_fingerprint = asdict(cfg.sampling)
    sampling_fingerprint.pop("generation_workers", None)
    return {
        "version": PREPROCESS_FINGERPRINT_VERSION,
        "input_fields": list(PHYSICAL_STATE_FIELDS),
        "target_fields": list(PROGNOSTIC_TARGET_FIELDS),
        "split_seed": int(cfg.training.split_seed),
        "val_fraction": float(cfg.training.val_fraction),
        "test_fraction": float(cfg.training.test_fraction),
        "normalization": asdict(cfg.normalization),
        "solver": asdict(cfg.solver),
        "sampling": sampling_fingerprint,
        "geometry": asdict(cfg.geometry),
        "raw_files": [_raw_file_signature(file_path, cfg=cfg) for file_path in files],
    }


def _processed_cache_is_valid(
    *,
    meta: Dict[str, Any],
    fingerprint: Dict[str, Any],
    processed_dir: Path,
) -> bool:
    """Return True when processed data matches the current preprocessing fingerprint."""
    if str(meta.get("task", "")) != "trajectory_transition":
        return False
    if meta.get("build_fingerprint") != fingerprint:
        return False
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        return False
    for split_name in ("train", "val", "test"):
        entries = splits.get(split_name)
        if not isinstance(entries, list) or not entries:
            return False
        for entry in entries:
            if not isinstance(entry, dict):
                return False
            rel_path = entry.get("file")
            n_samples = entry.get("n_samples")
            if not isinstance(rel_path, str) or not isinstance(n_samples, int):
                return False
            if n_samples < 1:
                return False
            if not (processed_dir / rel_path).is_file():
                return False
    return True


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
        raise ValueError("training.val_fraction + training.test_fraction must be < 1")

    indices = np.arange(len(files))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    n_total = len(files)
    n_val = max(1, int(round(float(val_fraction) * n_total)))
    n_test = max(1, int(round(float(test_fraction) * n_total)))
    if (n_val + n_test) >= n_total:
        raise ValueError(
            "Split sizes leave no train files; increase dataset size or reduce "
            "val/test fractions"
        )

    test_ids = set(indices[:n_test].tolist())
    val_ids = set(indices[n_test : n_test + n_val].tolist())
    train = [files[i] for i in range(n_total) if i not in val_ids and i not in test_ids]
    val = [files[i] for i in range(n_total) if i in val_ids]
    test = [files[i] for i in range(n_total) if i in test_ids]
    if not train or not val or not test:
        raise ValueError("Train/val/test splits must all be non-empty")
    return train, val, test


def _fit_stats_streaming(
    *,
    train_files: Sequence[Path],
    cfg: GCMulatorConfig,
) -> NormalizationStats:
    """Compute normalization statistics from the train split using streaming moments."""
    state_sum: np.ndarray | None = None
    state_sum2: np.ndarray | None = None
    state_count = 0

    param_mean: np.ndarray | None = None
    param_m2: np.ndarray | None = None
    param_count = 0

    transition_time_sum: np.ndarray | None = None
    transition_time_sum2: np.ndarray | None = None
    transition_time_count = 0

    for file_path in train_files:
        payload = _validated_raw_payload(file_path, cfg=cfg)
        transformed = apply_state_transforms(
            payload["state_inputs"],
            payload["input_fields"],
            cfg.normalization.state,
        ).astype(np.float64, copy=False)
        # Count every grid point in every transition frame because state
        # normalization is applied channel-wise over the full training corpus.
        state_chunk_sum = transformed.sum(axis=(0, 2, 3))
        state_chunk_sum2 = (transformed * transformed).sum(axis=(0, 2, 3))
        n_samples = int(transformed.shape[0] * transformed.shape[2] * transformed.shape[3])
        if state_sum is None:
            state_sum = state_chunk_sum
            state_sum2 = state_chunk_sum2
        else:
            state_sum += state_chunk_sum
            state_sum2 += state_chunk_sum2
        state_count += n_samples

        params = payload["params"].astype(np.float64, copy=False)
        if param_mean is None:
            param_mean = params.copy()
            param_m2 = np.zeros_like(params, dtype=np.float64)
            param_count = 1
        else:
            param_count += 1
            delta = params - param_mean
            param_mean += delta / float(param_count)
            delta2 = params - param_mean
            param_m2 += delta * delta2

        log10_transition_days = np.log10(
            np.maximum(payload["transition_days"].astype(np.float64, copy=False), 1.0e-30)
        ).reshape(-1, 1)
        transition_chunk_sum = log10_transition_days.sum(axis=0)
        transition_chunk_sum2 = (log10_transition_days * log10_transition_days).sum(axis=0)
        if transition_time_sum is None:
            transition_time_sum = transition_chunk_sum
            transition_time_sum2 = transition_chunk_sum2
        else:
            transition_time_sum += transition_chunk_sum
            transition_time_sum2 += transition_chunk_sum2
        transition_time_count += int(log10_transition_days.shape[0])

    if (
        state_sum is None
        or state_sum2 is None
        or param_mean is None
        or param_m2 is None
        or transition_time_sum is None
        or transition_time_sum2 is None
    ):
        raise RuntimeError("Could not infer normalization statistics from the training split")

    state_mean = state_sum / float(state_count)
    state_var = np.maximum(state_sum2 / float(state_count) - state_mean * state_mean, 0.0)
    state_std = np.maximum(np.sqrt(state_var), STD_FLOOR)

    if cfg.normalization.params.mode == "zscore":
        if param_count > 1:
            param_var = np.maximum(param_m2 / float(param_count), 0.0)
        else:
            param_var = np.zeros_like(param_mean, dtype=np.float64)
        param_std_raw = np.sqrt(param_var)
        param_is_constant = param_std_raw <= STD_FLOOR
        param_std = np.where(param_is_constant, 1.0, np.maximum(param_std_raw, STD_FLOOR))
        param_mean_out = param_mean
    elif cfg.normalization.params.mode == "none":
        param_is_constant = np.zeros_like(param_mean, dtype=bool)
        param_std = np.ones_like(param_mean, dtype=np.float64)
        param_mean_out = np.zeros_like(param_mean, dtype=np.float64)
        transition_time_is_constant = np.zeros((1,), dtype=bool)
        transition_time_std = np.ones((1,), dtype=np.float64)
        transition_time_mean_out = np.zeros((1,), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported param normalization mode: {cfg.normalization.params.mode}")

    if cfg.normalization.params.mode == "zscore":
        transition_time_mean = transition_time_sum / float(transition_time_count)
        transition_time_var = np.maximum(
            transition_time_sum2 / float(transition_time_count)
            - transition_time_mean * transition_time_mean,
            0.0,
        )
        transition_time_std_raw = np.sqrt(transition_time_var)
        transition_time_is_constant = transition_time_std_raw <= STD_FLOOR
        transition_time_std = np.where(
            transition_time_is_constant,
            1.0,
            np.maximum(transition_time_std_raw, STD_FLOOR),
        )
        transition_time_mean_out = transition_time_mean

    input_state_stats = StateNormalizationStats(
        field_names=tuple(PHYSICAL_STATE_FIELDS),
        field_transforms=dict(cfg.normalization.state.field_transforms),
        mean=state_mean.astype(np.float64),
        std=state_std.astype(np.float64),
        zscore_eps=float(cfg.normalization.state.zscore_eps),
        log10_eps=float(cfg.normalization.state.log10_eps),
        signed_log1p_scale=float(cfg.normalization.state.signed_log1p_scale),
    )
    target_state_stats = subset_state_stats(input_state_stats, PROGNOSTIC_TARGET_FIELDS)
    param_stats = ParamNormalizationStats(
        param_names=tuple(CONDITIONING_PARAM_NAMES),
        mean=param_mean_out.astype(np.float64),
        std=param_std.astype(np.float64),
        is_constant=np.asarray(param_is_constant, dtype=bool),
        zscore_eps=float(cfg.normalization.params.eps),
    )
    return NormalizationStats(
        input_state=input_state_stats,
        target_state=target_state_stats,
        params=param_stats,
        transition_time=ParamNormalizationStats(
            param_names=(TRANSITION_TIME_NAME,),
            mean=transition_time_mean_out.astype(np.float64),
            std=transition_time_std.astype(np.float64),
            is_constant=np.asarray(transition_time_is_constant, dtype=bool),
            zscore_eps=float(cfg.normalization.params.eps),
        ),
    )


def _write_processed_shard(
    *,
    src_file: Path,
    dst_dir: Path,
    stats: NormalizationStats,
    cfg: GCMulatorConfig,
) -> Dict[str, Any]:
    """Normalize one raw trajectory file and write one shard file."""
    payload = _validated_raw_payload(src_file, cfg=cfg)
    state_inputs_norm = normalize_state_tensor(
        payload["state_inputs"],
        stats.input_state,
    ).astype(np.float32)
    state_targets_norm = normalize_state_tensor(
        payload["state_targets"],
        stats.target_state,
    ).astype(np.float32)
    params_norm = normalize_params(payload["params"][None, ...], stats.params)[0].astype(np.float32)
    conditioning_norm = normalize_conditioning(
        payload["params"],
        payload["transition_days"],
        param_stats=stats.params,
        transition_time_stats=stats.transition_time,
    ).astype(np.float32)

    shard_name = f"{src_file.stem}.npz"
    np.savez(
        dst_dir / shard_name,
        state_inputs_norm=state_inputs_norm,
        state_targets_norm=state_targets_norm,
        params_norm=params_norm,
        conditioning_norm=conditioning_norm,
        transition_days=payload["transition_days"].astype(np.float64),
        anchor_steps=payload["anchor_steps"].astype(np.int64),
    )
    return {
        "file": str(Path(dst_dir.name) / shard_name),
        "n_samples": int(payload["transition_days"].shape[0]),
    }


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
            if _processed_cache_is_valid(
                meta=cached,
                fingerprint=fingerprint,
                processed_dir=processed_dir,
            ):
                LOGGER.info("Reusing cached processed dataset at %s", processed_dir)
                return cached
        except Exception:
            pass

    # A full rebuild is simpler and easier to reason about than partial shard
    # invalidation, so stale processed data is cleared eagerly.
    if processed_dir.exists():
        for path in processed_dir.glob("*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    train_files, val_files, test_files = _split_files(
        files,
        seed=int(cfg.training.split_seed),
        val_fraction=float(cfg.training.val_fraction),
        test_fraction=float(cfg.training.test_fraction),
    )
    stats = _fit_stats_streaming(train_files=train_files, cfg=cfg)

    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_shards = [
        _write_processed_shard(src_file=file_path, dst_dir=train_dir, stats=stats, cfg=cfg)
        for file_path in train_files
    ]
    val_shards = [
        _write_processed_shard(src_file=file_path, dst_dir=val_dir, stats=stats, cfg=cfg)
        for file_path in val_files
    ]
    test_shards = [
        _write_processed_shard(src_file=file_path, dst_dir=test_dir, stats=stats, cfg=cfg)
        for file_path in test_files
    ]

    with np.load(
        processed_dir / train_shards[0]["file"],
        allow_pickle=False,
    ) as first_train_payload:
        sample_input = np.asarray(first_train_payload["state_inputs_norm"], dtype=np.float32)[0]
        sample_target = np.asarray(first_train_payload["state_targets_norm"], dtype=np.float32)[0]

    meta: Dict[str, Any] = {
        "task": "trajectory_transition",
        "input_fields": list(stats.input_state.field_names),
        "target_fields": list(stats.target_state.field_names),
        "param_names": list(stats.params.param_names),
        "conditioning_names": list(stats.params.param_names) + list(
            stats.transition_time.param_names
        ),
        "input_shape": {
            "C": int(sample_input.shape[0]),
            "H": int(sample_input.shape[1]),
            "W": int(sample_input.shape[2]),
        },
        "target_shape": {
            "C": int(sample_target.shape[0]),
            "H": int(sample_target.shape[1]),
            "W": int(sample_target.shape[2]),
        },
        "splits": {"train": train_shards, "val": val_shards, "test": test_shards},
        "normalization": stats_to_json(stats),
        "solver": asdict(cfg.solver),
        "sampling": asdict(cfg.sampling),
        "geometry": {
            **asdict(cfg.geometry),
            "lat_order": "north_to_south",
            "lon_origin": "0_to_2pi",
        },
        "build_fingerprint": fingerprint,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TransitionShardDataset(Dataset):
    """PyTorch dataset wrapper over processed NPZ shards."""

    def __init__(
        self,
        *,
        processed_dir: Path,
        shard_entries: Sequence[Dict[str, Any]],
    ) -> None:
        """Index processed shard metadata and prepare a one-shard read cache."""
        self.processed_dir = processed_dir
        self.shard_entries = list(shard_entries)
        if not self.shard_entries:
            raise ValueError("TransitionShardDataset requires at least one shard entry")

        self.cumulative_counts: List[int] = []
        running = 0
        for entry in self.shard_entries:
            n_samples = int(entry["n_samples"])
            if n_samples < 1:
                raise ValueError("Shard n_samples must be >= 1")
            running += n_samples
            self.cumulative_counts.append(running)

        self._cached_path: Path | None = None
        self._cached_arrays: Dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        """Return the total number of transition samples across all shards."""
        return int(self.cumulative_counts[-1])

    def _load_shard(self, shard_path: Path) -> Dict[str, np.ndarray]:
        """Load one processed shard and reuse it while adjacent indices are read."""
        if self._cached_path == shard_path and self._cached_arrays is not None:
            return self._cached_arrays
        # Each dataset instance keeps a single-shard cache because dataloaders
        # tend to walk consecutive indices within the same simulation window.
        with np.load(shard_path, allow_pickle=False) as npz:
            arrays = {
                "conditioning_norm": np.asarray(npz["conditioning_norm"], dtype=np.float32),
                "state_inputs_norm": np.asarray(npz["state_inputs_norm"], dtype=np.float32),
                "state_targets_norm": np.asarray(npz["state_targets_norm"], dtype=np.float32),
            }
        self._cached_path = shard_path
        self._cached_arrays = arrays
        return arrays

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one normalized transition sample from the processed shards."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_counts, int(index))
        shard_start = 0 if shard_index == 0 else self.cumulative_counts[shard_index - 1]
        local_index = int(index) - int(shard_start)
        shard_path = self.processed_dir / str(self.shard_entries[shard_index]["file"])
        arrays = self._load_shard(shard_path)
        return (
            torch.from_numpy(arrays["conditioning_norm"][local_index]),
            torch.from_numpy(arrays["state_inputs_norm"][local_index]),
            torch.from_numpy(arrays["state_targets_norm"][local_index]),
        )


class PreloadedTransitionDataset(Dataset):
    """Dataset backed by already-loaded transition tensors."""

    def __init__(
        self,
        *,
        conditioning: torch.Tensor,
        state_inputs: torch.Tensor,
        state_targets: torch.Tensor,
    ) -> None:
        """Validate and store already-loaded transition tensors."""
        if conditioning.ndim != 2:
            raise ValueError(f"conditioning must be [N,P], got {tuple(conditioning.shape)}")
        if state_inputs.ndim != 4 or state_targets.ndim != 4:
            raise ValueError("state_inputs/state_targets must be [N,C,H,W]")
        n_samples = int(conditioning.shape[0])
        if any(int(tensor.shape[0]) != n_samples for tensor in (state_inputs, state_targets)):
            raise ValueError("Preloaded tensor batch sizes must match")
        self.conditioning = conditioning
        self.state_inputs = state_inputs
        self.state_targets = state_targets

    def __len__(self) -> int:
        """Return the number of preloaded transition samples."""
        return int(self.conditioning.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one preloaded transition sample."""
        return (
            self.conditioning[index],
            self.state_inputs[index],
            self.state_targets[index],
        )


def _preload_split_to_device(
    *,
    processed_dir: Path,
    shard_entries: Sequence[Dict[str, Any]],
    device: torch.device,
) -> PreloadedTransitionDataset:
    """Load one split into contiguous tensors and move them to ``device``."""
    conditioning_rows: List[np.ndarray] = []
    state_input_rows: List[np.ndarray] = []
    state_target_rows: List[np.ndarray] = []

    for entry in shard_entries:
        with np.load(processed_dir / str(entry["file"]), allow_pickle=False) as npz:
            conditioning = np.asarray(npz["conditioning_norm"], dtype=np.float32)
            state_inputs = np.asarray(npz["state_inputs_norm"], dtype=np.float32)
            state_targets = np.asarray(npz["state_targets_norm"], dtype=np.float32)
            conditioning_rows.append(conditioning)
            state_input_rows.append(state_inputs)
            state_target_rows.append(state_targets)

    conditioning = torch.from_numpy(np.concatenate(conditioning_rows, axis=0)).to(device=device)
    state_inputs = torch.from_numpy(np.concatenate(state_input_rows, axis=0)).to(device=device)
    state_targets = torch.from_numpy(np.concatenate(state_target_rows, axis=0)).to(device=device)
    return PreloadedTransitionDataset(
        conditioning=conditioning,
        state_inputs=state_inputs,
        state_targets=state_targets,
    )


def _collect_predictions(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the transition model on a split loader and collect predictions/targets."""
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for conditioning_batch, state_input_batch, state_target_batch in loader:
            conditioning_batch = conditioning_batch.to(device=device)
            state_input_batch = state_input_batch.to(device=device)
            with autocast_context(device, amp_mode):
                pred_batch = model(state_input_batch, conditioning_batch)
            predictions.append(pred_batch.detach().cpu().numpy())
            targets.append(state_target_batch.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def _named_scalar_map(values: np.ndarray, field_names: Sequence[str]) -> Dict[str, float]:
    """Map channel-aligned scalars to ``{field_name: value}``."""
    return {str(field_names[index]): float(values[index]) for index in range(len(field_names))}


def _compute_spherical_power_spectrum_metrics(
    *,
    pred_phys: np.ndarray,
    target_phys: np.ndarray,
    field_names: Sequence[str],
    grid: str,
) -> Dict[str, Any]:
    """Compute a true spherical harmonic power-spectrum mismatch."""
    from torch_harmonics import RealSHT

    pred_tensor = torch.from_numpy(np.asarray(pred_phys, dtype=np.float32))
    target_tensor = torch.from_numpy(np.asarray(target_phys, dtype=np.float32))
    nlat = int(pred_tensor.shape[-2])
    nlon = int(pred_tensor.shape[-1])
    sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid, csphase=False).float()

    with torch.no_grad():
        pred_coeffs = sht(pred_tensor).abs().pow(2)
        target_coeffs = sht(target_tensor).abs().pow(2)
        pred_coeffs[..., 1:] *= 2.0
        target_coeffs[..., 1:] *= 2.0
        pred_ps = pred_coeffs.sum(dim=-1).mean(dim=0)
        target_ps = target_coeffs.sum(dim=-1).mean(dim=0)

    eps = torch.tensor(1.0e-12, dtype=pred_ps.dtype)
    rel_l1 = torch.sum(torch.abs(pred_ps - target_ps), dim=-1) / torch.clamp(
        torch.sum(torch.abs(target_ps), dim=-1),
        min=eps,
    )
    return {
        "global_relative_l1": float(torch.mean(rel_l1).item()),
        "per_channel_relative_l1": _named_scalar_map(rel_l1.cpu().numpy(), field_names),
        "degree_count": int(pred_ps.shape[-1]),
    }


def _compute_one_step_metrics(
    *,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    field_names: Sequence[str],
    state_stats: StateNormalizationStats,
    grid: str,
) -> Dict[str, Any]:
    """Compute normalized, physical, and spherical-spectrum single-call metrics."""
    diff_norm = pred_norm - target_norm
    per_channel_rmse_norm = np.sqrt(np.mean(diff_norm**2, axis=(0, 2, 3)))

    pred_phys = denormalize_state_tensor(
        pred_norm,
        stats=state_stats,
    ).astype(np.float64, copy=False)
    target_phys = denormalize_state_tensor(
        target_norm,
        stats=state_stats,
    ).astype(np.float64, copy=False)
    diff_phys = pred_phys - target_phys
    per_channel_rmse_phys = np.sqrt(np.mean(diff_phys**2, axis=(0, 2, 3)))
    per_channel_mae_phys = np.mean(np.abs(diff_phys), axis=(0, 2, 3))

    return {
        "normalized": {
            "global_rmse": float(np.sqrt(np.mean(diff_norm**2))),
            "per_channel_rmse": _named_scalar_map(per_channel_rmse_norm, field_names),
        },
        "physical": {
            "global_rmse": float(np.sqrt(np.mean(diff_phys**2))),
            "per_channel_rmse": _named_scalar_map(per_channel_rmse_phys, field_names),
            "per_channel_mae": _named_scalar_map(per_channel_mae_phys, field_names),
        },
        "spherical_spectrum": _compute_spherical_power_spectrum_metrics(
            pred_phys=pred_phys,
            target_phys=target_phys,
            field_names=field_names,
            grid=grid,
        ),
    }


def _write_training_history_csv(*, history: Sequence[Dict[str, float]], csv_path: Path) -> None:
    """Write per-epoch training history rows to CSV."""
    field_names = ["epoch", "train_loss", "val_loss", "lr", "epoch_seconds"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(round(float(row["epoch"]))),
                    "train_loss": _format_sigfig(float(row["train_loss"])),
                    "val_loss": _format_sigfig(float(row["val_loss"])),
                    "lr": _format_sigfig(float(row["lr"])),
                    "epoch_seconds": f"{float(row['epoch_seconds']):.2f}",
                }
            )


def _format_sigfig(value: float, *, digits: int = 4) -> str:
    """Format one scalar with a fixed number of significant figures."""
    if digits < 1:
        raise ValueError("digits must be >= 1")
    return f"{float(value):.{int(digits)}g}"


def _check_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Raise if tensor contains NaN/Inf values."""
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")


def _is_power_of_two(value: int) -> bool:
    """Return True when the value is a positive power of two."""
    integer = int(value)
    return integer > 0 and (integer & (integer - 1)) == 0


def _resolve_runtime_amp_mode(*, requested_amp_mode: str, nlat: int, nlon: int) -> str:
    """Resolve runtime AMP mode with shape safety checks for harmonic layers."""
    mode = str(requested_amp_mode).lower()
    if mode in {"bf16", "fp16"} and (
        not _is_power_of_two(nlat) or not _is_power_of_two(nlon)
    ):
        LOGGER.warning(
            "Disabling AMP (%s) for grid %dx%d because transformed "
            "dimensions are not powers of two",
            mode,
            int(nlat),
            int(nlon),
        )
        return "none"
    return mode


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all optimizer parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _linear_warmup_lr(*, epoch: int, base_lr: float, warmup_epochs: int) -> float:
    """Compute a linear warmup that reaches the base LR at the last warmup epoch."""
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    if base_lr <= 0:
        raise ValueError("base_lr must be > 0")
    if warmup_epochs == 0:
        return float(base_lr)
    e = max(1, int(epoch))
    progress = float(min(e, warmup_epochs)) / float(warmup_epochs)
    return float(base_lr) * progress


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
        raise ValueError("total_epochs must be >= 1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    if base_lr <= 0:
        raise ValueError("base_lr must be > 0")
    if min_lr < 0 or min_lr > base_lr:
        raise ValueError("min_lr must satisfy 0 <= min_lr <= base_lr")

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


def _loss_improved(*, current: float, best: float, min_delta: float) -> bool:
    """Return whether the monitored loss improved beyond the absolute tolerance."""
    if min_delta <= 0:
        raise ValueError("min_delta must be > 0")
    return float(current) < float(best) - float(min_delta)


def _early_stopping_patience(*, scheduler_patience: int, warmup_epochs: int) -> int:
    """Choose a conservative patience that still leaves room for LR reductions."""
    if scheduler_patience < 0:
        raise ValueError("scheduler_patience must be >= 0")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    return max(
        1,
        int(warmup_epochs) + int(scheduler_patience),
        int(scheduler_patience) * EARLY_STOPPING_PATIENCE_MULTIPLIER,
    )


def _set_determinism(seed: int) -> None:
    """Apply reproducible random-seed and deterministic-kernel settings."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_emulator(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Train the transition emulator end-to-end and persist artifacts."""
    ensure_torch_harmonics_importable()
    processed_meta = preprocess_dataset(cfg, config_path=config_path)
    processed_dir = resolve_path(config_path, cfg.paths.processed_dir)
    model_dir = resolve_path(config_path, cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_dict = asdict(cfg)
    (model_dir / "config_used.resolved.json").write_text(
        json.dumps(resolved_cfg_dict, indent=2),
        encoding="utf-8",
    )
    cfg_suffix = config_path.suffix.lower()
    if cfg_suffix not in {".json", ".yaml", ".yml"}:
        cfg_suffix = ".txt"
    (model_dir / f"config_used.original{cfg_suffix}").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    stats = stats_from_json(processed_meta["normalization"])
    input_field_names = list(processed_meta["input_fields"])
    target_field_names = list(processed_meta["target_fields"])
    device = choose_device(cfg.training.device)
    preload_to_gpu = bool(cfg.training.preload_to_gpu)
    if preload_to_gpu and device.type != "cuda":
        raise RuntimeError("training.preload_to_gpu=true requires CUDA")

    if preload_to_gpu:
        train_dataset = _preload_split_to_device(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["train"],
            device=device,
        )
        val_dataset = _preload_split_to_device(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["val"],
            device=device,
        )
        test_dataset = _preload_split_to_device(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["test"],
            device=device,
        )
        LOGGER.info(
            "Preloaded processed splits to %s | n_train=%d | n_val=%d | n_test=%d",
            device,
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )
    else:
        train_dataset = TransitionShardDataset(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["train"],
        )
        val_dataset = TransitionShardDataset(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["val"],
        )
        test_dataset = TransitionShardDataset(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["test"],
        )

    if len(train_dataset) < int(cfg.training.batch_size):
        raise ValueError("Training split size is smaller than batch_size while drop_last=True")

    _set_determinism(int(cfg.training.seed))

    if preload_to_gpu:
        loader_common: Dict[str, Any] = {"num_workers": 0, "pin_memory": False}
    else:
        loader_common = {
            "num_workers": int(cfg.training.num_workers),
            "pin_memory": bool(cfg.training.pin_memory),
        }
        if int(cfg.training.num_workers) > 0:
            loader_common["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=bool(cfg.training.shuffle),
        drop_last=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        drop_last=False,
        **loader_common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    use_non_blocking = (
        bool(cfg.training.pin_memory)
        and device.type == "cuda"
        and not preload_to_gpu
    )
    (
        sample_conditioning,
        sample_state_input,
        sample_state_target,
    ) = train_dataset[0]
    input_state_chans = int(sample_state_input.shape[0])
    target_state_chans = int(sample_state_target.shape[0])
    nlat = int(sample_state_input.shape[1])
    nlon = int(sample_state_input.shape[2])
    conditioning_dim = int(sample_conditioning.shape[0])
    residual_input_indices = [
        input_field_names.index(field_name)
        for field_name in target_field_names
    ]

    model = build_state_conditioned_transition_model(
        img_size=(nlat, nlon),
        input_state_chans=input_state_chans,
        target_state_chans=target_state_chans,
        param_dim=conditioning_dim,
        residual_input_indices=residual_input_indices,
        cfg_model=cfg.model,
        lat_order=str(processed_meta["geometry"]["lat_order"]),
        lon_origin=str(processed_meta["geometry"]["lon_origin"]),
    ).to(device)
    runtime_amp_mode = _resolve_runtime_amp_mode(
        requested_amp_mode=cfg.training.amp_mode,
        nlat=nlat,
        nlon=nlon,
    )

    loss_fn = SphereLoss(nlat=nlat, nlon=nlon, grid=cfg.model.grid).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler_type = str(cfg.training.scheduler.type)
    scheduler_patience = int(cfg.training.scheduler.patience)
    scheduler_min_lr = float(cfg.training.scheduler.min_lr)
    scheduler_eps = float(cfg.training.scheduler.eps)
    warmup_epochs = int(cfg.training.scheduler.warmup_epochs)
    scheduler = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.training.scheduler.factor),
            patience=scheduler_patience,
            threshold=scheduler_eps,
            threshold_mode="abs",
            min_lr=scheduler_min_lr,
            eps=scheduler_eps,
        )

    scaler = None
    if runtime_amp_mode == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    epochs = int(cfg.training.epochs)
    best_val = float("inf")
    best_path = model_dir / "best.pt"
    last_path = model_dir / "last.pt"
    history: List[Dict[str, float]] = []
    base_learning_rate = float(cfg.training.learning_rate)
    epochs_without_improvement = 0
    early_stop_patience = _early_stopping_patience(
        scheduler_patience=scheduler_patience,
        warmup_epochs=warmup_epochs,
    )

    # ------------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        if scheduler_type == "cosine_warmup":
            _set_optimizer_lr(
                optimizer,
                _cosine_warmup_lr(
                    epoch=epoch,
                    total_epochs=epochs,
                    base_lr=base_learning_rate,
                    min_lr=scheduler_min_lr,
                    warmup_epochs=warmup_epochs,
                ),
            )
        elif scheduler_type == "plateau" and warmup_epochs > 0 and epoch <= warmup_epochs:
            _set_optimizer_lr(
                optimizer,
                _linear_warmup_lr(
                    epoch=epoch,
                    base_lr=base_learning_rate,
                    warmup_epochs=warmup_epochs,
                ),
            )

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for conditioning_batch, state_input_batch, state_target_batch in train_loader:
            conditioning_batch = conditioning_batch.to(device=device, non_blocking=use_non_blocking)
            state_input_batch = state_input_batch.to(device=device, non_blocking=use_non_blocking)
            state_target_batch = state_target_batch.to(device=device, non_blocking=use_non_blocking)

            _check_finite_tensor(conditioning_batch, name="train conditioning batch")
            _check_finite_tensor(state_input_batch, name="train state_input batch")
            _check_finite_tensor(state_target_batch, name="train target batch")

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, runtime_amp_mode):
                prediction = model(state_input_batch, conditioning_batch)
                _check_finite_tensor(prediction, name="train prediction batch")
                loss = loss_fn(prediction, state_target_batch)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite")

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
            for conditioning_batch, state_input_batch, state_target_batch in val_loader:
                conditioning_batch = conditioning_batch.to(
                    device=device,
                    non_blocking=use_non_blocking,
                )
                state_input_batch = state_input_batch.to(
                    device=device,
                    non_blocking=use_non_blocking,
                )
                state_target_batch = state_target_batch.to(
                    device=device,
                    non_blocking=use_non_blocking,
                )

                _check_finite_tensor(conditioning_batch, name="val conditioning batch")
                _check_finite_tensor(state_input_batch, name="val state_input batch")
                _check_finite_tensor(state_target_batch, name="val target batch")

                with autocast_context(device, runtime_amp_mode):
                    prediction = model(state_input_batch, conditioning_batch)
                    _check_finite_tensor(prediction, name="val prediction batch")
                    vloss = loss_fn(prediction, state_target_batch)
                if not torch.isfinite(vloss).item():
                    raise RuntimeError("Validation loss became non-finite")
                val_loss_sum += float(vloss.detach().item())
                val_count += 1

        if train_count == 0 or val_count == 0:
            raise RuntimeError("No training or validation batches were produced")
        train_loss = train_loss_sum / float(train_count)
        val_loss = val_loss_sum / float(val_count)
        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_start
        LOGGER.info(
            "Epoch %4d/%4d | train=%s | val=%s | lr=%s | time=%.2fs",
            epoch,
            epochs,
            _format_sigfig(train_loss),
            _format_sigfig(val_loss),
            _format_sigfig(current_lr),
            epoch_seconds,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "epoch_seconds": epoch_seconds,
            }
        )

        # Checkpoints include enough metadata for export and downstream tools to
        # reconstruct shapes, normalization, and dataset conventions.
        checkpoint = {
            "mode": "state_conditioned_prognostic_transition",
            "model_state": model.state_dict(),
            "input_fields": input_field_names,
            "target_fields": target_field_names,
            "param_names": list(processed_meta["param_names"]),
            "conditioning_names": list(processed_meta["conditioning_names"]),
            "shape": {
                "input_C": input_state_chans,
                "target_C": target_state_chans,
                "H": nlat,
                "W": nlon,
            },
            "geometry": dict(processed_meta["geometry"]),
            "normalization": stats_to_json(stats),
            "solver": asdict(cfg.solver),
            "sampling": asdict(cfg.sampling),
            "model_config": asdict(cfg.model),
            "training_config": asdict(cfg.training),
            "runtime_amp_mode": runtime_amp_mode,
            "resolved_config": resolved_cfg_dict,
            "source_config_path": str(config_path),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "learning_rate": float(current_lr),
            "epoch_seconds": float(epoch_seconds),
        }
        torch.save(checkpoint, last_path)
        if _loss_improved(current=val_loss, best=best_val, min_delta=scheduler_eps):
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        if epoch < epochs and epochs_without_improvement >= early_stop_patience:
            LOGGER.info(
                "Early stopping at epoch %4d/%4d after %d epochs without "
                "validation improvement larger than %.1e",
                epoch,
                epochs,
                epochs_without_improvement,
                scheduler_eps,
            )
            break

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    val_pred, val_target = _collect_predictions(
        model=model,
        loader=val_loader,
        device=device,
        amp_mode=runtime_amp_mode,
    )
    test_pred, test_target = _collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
        amp_mode=runtime_amp_mode,
    )
    val_metrics = _compute_one_step_metrics(
        pred_norm=val_pred,
        target_norm=val_target,
        field_names=target_field_names,
        state_stats=stats.target_state,
        grid=cfg.model.grid,
    )
    test_metrics = _compute_one_step_metrics(
        pred_norm=test_pred,
        target_norm=test_target,
        field_names=target_field_names,
        state_stats=stats.target_state,
        grid=cfg.model.grid,
    )

    (model_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    _write_training_history_csv(
        history=history,
        csv_path=model_dir / "training_history.csv",
    )
    (model_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (model_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_val_loss": float(best_val),
        "history_path": str(model_dir / "training_history.json"),
        "history_csv_path": str(model_dir / "training_history.csv"),
        "val_metrics_path": str(model_dir / "val_metrics.json"),
        "test_metrics_path": str(model_dir / "test_metrics.json"),
        "processed_meta": str(processed_dir / "processed_meta.json"),
    }
