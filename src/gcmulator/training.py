"""Preprocessing and training pipeline for direct-jump visible-state transition learning.

This module covers three stages:

1. validate raw trajectory-window files
2. preprocess them into split-level normalized shards
3. train and evaluate the direct-jump transition emulator
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import csv
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch

from .config import (
    CONDITIONING_PARAM_NAMES,
    GCMulatorConfig,
    PHYSICAL_STATE_FIELDS,
    TRANSITION_TIME_NAME,
    resolve_path,
)
from .geometry import geometry_shift_for_nlon
from .modeling import (
    SphereLoss,
    autocast_context,
    build_state_conditioned_transition_model,
    choose_device,
    ensure_torch_harmonics_importable,
)
from .normalization import (
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
)
from .sampling import (
    LiveTransitionCatalog,
    build_live_transition_catalog,
    build_uniform_checkpoint_schedule,
    checkpoint_schedule_kwargs,
    valid_anchor_counts_for_catalog,
    weighted_log10_transition_stats,
)


LOGGER = logging.getLogger("train")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESS_FINGERPRINT_VERSION = 16
RAW_REQUIRED_KEYS = (
    "checkpoint_states",
    "checkpoint_steps",
    "checkpoint_days",
    "state_fields",
    "params",
    "param_names",
    "default_time_days",
    "burn_in_days",
    "dt_seconds",
    "starttime_index",
    "saved_checkpoint_interval_days",
    "n_saved_checkpoints",
    "M",
    "nlat",
    "nlon",
    "lat_order",
    "lon_origin",
    "lon_shift",
)
EARLY_STOPPING_PATIENCE_MULTIPLIER = 3
# Seed offsets ensure validation and test pair tables are deterministic but
# independent from the per-epoch training pair tables.
VAL_PAIR_SEED_OFFSET = 100_000
TEST_PAIR_SEED_OFFSET = 200_000
# Multiplicative headroom applied when checking whether a processed split fits
# in GPU VRAM.  The 20% margin covers model parameters, optimizer state, and
# temporary activations that live alongside the preloaded data.
VRAM_HEADROOM_FACTOR = 1.2


def _display_repo_path(path: Path) -> str:
    """Return a repository-relative path string for stored metadata."""
    return str(Path(os.path.relpath(Path(path).resolve(), start=PROJECT_ROOT)))


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
    """Load and validate one raw checkpoint-sequence payload against the config contract."""
    payload = _load_npy_payload_dict(file_path)
    missing = [key for key in RAW_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Raw file {file_path} is missing required keys: {missing}")

    state_fields = [
        str(value)
        for value in np.asarray(payload["state_fields"], dtype=object).tolist()
    ]
    if tuple(state_fields) != tuple(PHYSICAL_STATE_FIELDS):
        raise ValueError(
            f"Raw file {file_path} has state_fields={state_fields}, "
            f"expected {list(PHYSICAL_STATE_FIELDS)}"
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

    checkpoint_states = np.asarray(payload["checkpoint_states"], dtype=np.float32)
    checkpoint_steps = np.asarray(payload["checkpoint_steps"], dtype=np.int64)
    checkpoint_days = np.asarray(payload["checkpoint_days"], dtype=np.float64)
    params = np.asarray(payload["params"], dtype=np.float64)
    default_time_days = float(np.asarray(payload["default_time_days"], dtype=np.float64).item())
    burn_in_days = float(np.asarray(payload["burn_in_days"], dtype=np.float64).item())
    dt_seconds = float(np.asarray(payload["dt_seconds"], dtype=np.float64).item())
    starttime_index = int(np.asarray(payload["starttime_index"], dtype=np.int64).item())
    saved_checkpoint_interval_days = float(
        np.asarray(payload["saved_checkpoint_interval_days"], dtype=np.float64).item()
    )
    n_saved_checkpoints = int(np.asarray(payload["n_saved_checkpoints"], dtype=np.int64).item())
    M = int(np.asarray(payload["M"], dtype=np.int64).item())
    nlat = int(np.asarray(payload["nlat"], dtype=np.int64).item())
    nlon = int(np.asarray(payload["nlon"], dtype=np.int64).item())
    lat_order = str(np.asarray(payload["lat_order"], dtype=object).item())
    lon_origin = str(np.asarray(payload["lon_origin"], dtype=object).item())
    lon_shift = int(np.asarray(payload["lon_shift"], dtype=np.int64).item())

    if checkpoint_states.ndim != 4:
        raise ValueError(f"Raw checkpoint states in {file_path} must be [S,C,H,W]")
    if checkpoint_states.shape[1] != len(PHYSICAL_STATE_FIELDS):
        raise ValueError(f"Checkpoint channel count mismatch in {file_path}")
    if checkpoint_steps.shape != (checkpoint_states.shape[0],):
        raise ValueError(f"checkpoint_steps shape mismatch in {file_path}")
    if checkpoint_days.shape != (checkpoint_states.shape[0],):
        raise ValueError(f"checkpoint_days shape mismatch in {file_path}")
    if params.shape != (len(CONDITIONING_PARAM_NAMES),):
        raise ValueError(f"params shape mismatch in {file_path}: {params.shape}")
    if int(checkpoint_states.shape[0]) != n_saved_checkpoints:
        raise ValueError(f"n_saved_checkpoints mismatch in {file_path}")
    if int(checkpoint_states.shape[2]) != nlat or int(checkpoint_states.shape[3]) != nlon:
        raise ValueError(f"Checkpoint spatial metadata mismatch in {file_path}")
    if np.any(np.diff(checkpoint_steps) <= 0):
        raise ValueError(f"checkpoint_steps must be strictly increasing in {file_path}")
    if np.any(np.diff(checkpoint_days) <= 0.0):
        raise ValueError(f"checkpoint_days must be strictly increasing in {file_path}")

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

    expected_schedule = build_uniform_checkpoint_schedule(
        time_days=float(cfg.solver.default_time_days),
        dt_seconds=float(cfg.solver.dt_seconds),
        **checkpoint_schedule_kwargs(
            saved_checkpoint_interval_days=float(cfg.sampling.saved_checkpoint_interval_days),
            saved_snapshots_per_sim=cfg.sampling.saved_snapshots_per_sim,
        ),
    )
    if not math.isclose(
        saved_checkpoint_interval_days,
        float(expected_schedule.interval_days),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"Raw file {file_path} has saved_checkpoint_interval_days="
            f"{saved_checkpoint_interval_days}, expected {expected_schedule.interval_days}"
        )
    if not np.array_equal(checkpoint_steps, expected_schedule.checkpoint_steps):
        raise ValueError(
            f"Raw file {file_path} checkpoint_steps do not match the configured cadence"
        )
    if not np.allclose(
        checkpoint_days,
        expected_schedule.checkpoint_days,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"Raw file {file_path} checkpoint_days do not match the configured cadence"
        )

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
        "checkpoint_states": checkpoint_states,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_days": checkpoint_days,
        "state_fields": state_fields,
        "params": params,
        "param_names": param_names,
        "default_time_days": default_time_days,
        "burn_in_days": burn_in_days,
        "dt_seconds": dt_seconds,
        "starttime_index": starttime_index,
        "saved_checkpoint_interval_days": saved_checkpoint_interval_days,
        "n_saved_checkpoints": n_saved_checkpoints,
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
        "state_fields": list(metadata["state_fields"]),
        "param_names": list(metadata["param_names"]),
        "default_time_days": float(metadata["default_time_days"]),
        "burn_in_days": float(metadata["burn_in_days"]),
        "dt_seconds": float(metadata["dt_seconds"]),
        "starttime_index": int(metadata["starttime_index"]),
        "saved_checkpoint_interval_days": float(metadata["saved_checkpoint_interval_days"]),
        "n_saved_checkpoints": int(metadata["n_saved_checkpoints"]),
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
        "state_fields": list(PHYSICAL_STATE_FIELDS),
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
    if str(meta.get("task", "")) != "checkpoint_sequence_transition":
        return False
    if meta.get("build_fingerprint") != fingerprint:
        return False
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        return False
    sequence_length = meta.get("sequence_length")
    split_sequence_counts = meta.get("split_sequence_counts")
    live_transition_catalog = meta.get("live_transition_catalog")
    if not isinstance(sequence_length, int) or sequence_length < 2:
        return False
    if not isinstance(split_sequence_counts, dict):
        return False
    if not isinstance(live_transition_catalog, dict):
        return False
    for split_name in ("train", "val", "test"):
        entries = splits.get(split_name)
        if not isinstance(entries, list) or not entries:
            return False
        total_sequences = 0
        for entry in entries:
            if not isinstance(entry, dict):
                return False
            rel_path = entry.get("file")
            n_sequences = entry.get("n_sequences")
            entry_sequence_length = entry.get("sequence_length")
            if (
                not isinstance(rel_path, str)
                or not isinstance(n_sequences, int)
                or not isinstance(entry_sequence_length, int)
            ):
                return False
            if n_sequences < 1 or entry_sequence_length != sequence_length:
                return False
            if not (processed_dir / rel_path).is_file():
                return False
            total_sequences += int(n_sequences)
        if int(split_sequence_counts.get(split_name, -1)) != total_sequences:
            return False
    return True


def _expected_checkpoint_schedule_and_catalog(
    cfg: GCMulatorConfig,
) -> tuple[Any, LiveTransitionCatalog]:
    """Build the checkpoint schedule and discrete live jump catalog from config."""
    schedule = build_uniform_checkpoint_schedule(
        time_days=float(cfg.solver.default_time_days),
        dt_seconds=float(cfg.solver.dt_seconds),
        **checkpoint_schedule_kwargs(
            saved_checkpoint_interval_days=float(cfg.sampling.saved_checkpoint_interval_days),
            saved_snapshots_per_sim=cfg.sampling.saved_snapshots_per_sim,
        ),
    )
    catalog = build_live_transition_catalog(
        checkpoint_days=schedule.checkpoint_days,
        burn_in_days=float(cfg.sampling.burn_in_days),
        transition_days_min=float(cfg.sampling.live_transition_days_min),
        transition_days_max=float(cfg.sampling.live_transition_days_max),
        tolerance_fraction=float(cfg.sampling.live_transition_tolerance_fraction),
        pair_sampling_policy=str(cfg.sampling.pair_sampling_policy),
    )
    return schedule, catalog


def _live_transition_catalog_to_json(catalog: LiveTransitionCatalog) -> Dict[str, Any]:
    """Serialize one discrete live transition catalog for metadata storage."""
    return {
        "gap_offsets": catalog.gap_offsets.astype(np.int64).tolist(),
        "transition_days": catalog.transition_days.astype(np.float64).tolist(),
        "probabilities": catalog.probabilities.astype(np.float64).tolist(),
        "burn_in_start_index": int(catalog.burn_in_start_index),
    }


def _live_transition_catalog_from_json(payload: Dict[str, Any]) -> LiveTransitionCatalog:
    """Reconstruct a discrete live transition catalog from stored metadata."""
    return LiveTransitionCatalog(
        gap_offsets=np.asarray(payload["gap_offsets"], dtype=np.int64),
        transition_days=np.asarray(payload["transition_days"], dtype=np.float64),
        probabilities=np.asarray(payload["probabilities"], dtype=np.float64),
        burn_in_start_index=int(payload["burn_in_start_index"]),
    )


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
    live_catalog: LiveTransitionCatalog,
) -> NormalizationStats:
    """Compute normalization statistics from the train split using streaming moments."""
    state_sum: np.ndarray | None = None
    state_sum2: np.ndarray | None = None
    state_count = 0

    param_mean: np.ndarray | None = None
    param_m2: np.ndarray | None = None
    param_count = 0

    for file_path in train_files:
        payload = _validated_raw_payload(file_path, cfg=cfg)
        transformed = apply_state_transforms(
            payload["checkpoint_states"],
            PHYSICAL_STATE_FIELDS,
            cfg.normalization.state,
        ).astype(np.float64, copy=False)

        chunk_sum = transformed.sum(axis=(0, 2, 3))
        chunk_sum2 = (transformed * transformed).sum(axis=(0, 2, 3))
        n_samples = int(
            transformed.shape[0] * transformed.shape[2] * transformed.shape[3]
        )
        if state_sum is None:
            state_sum = chunk_sum
            state_sum2 = chunk_sum2
        else:
            state_sum += chunk_sum
            state_sum2 += chunk_sum2
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

    if (
        state_sum is None
        or state_sum2 is None
        or param_mean is None
        or param_m2 is None
    ):
        raise RuntimeError("Could not infer normalization statistics from the training split")

    state_mean = state_sum / float(state_count)
    state_var = np.maximum(
        state_sum2 / float(state_count) - state_mean * state_mean,
        0.0,
    )
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
    else:
        raise ValueError(f"Unsupported param normalization mode: {cfg.normalization.params.mode}")

    if cfg.normalization.params.mode == "zscore":
        (
            transition_time_mean_out,
            transition_time_std,
            transition_time_is_constant,
        ) = weighted_log10_transition_stats(live_catalog)
    else:
        transition_time_mean_out = np.zeros((1,), dtype=np.float64)
        transition_time_std = np.ones((1,), dtype=np.float64)
        transition_time_is_constant = np.zeros((1,), dtype=bool)

    state_stats = StateNormalizationStats(
        field_names=tuple(PHYSICAL_STATE_FIELDS),
        field_transforms=dict(cfg.normalization.state.field_transforms),
        mean=state_mean.astype(np.float64),
        std=state_std.astype(np.float64),
        zscore_eps=float(cfg.normalization.state.zscore_eps),
        log10_eps=float(cfg.normalization.state.log10_eps),
        signed_log1p_scale=float(cfg.normalization.state.signed_log1p_scale),
    )
    param_stats = ParamNormalizationStats(
        param_names=tuple(CONDITIONING_PARAM_NAMES),
        mean=param_mean_out.astype(np.float64),
        std=param_std.astype(np.float64),
        is_constant=np.asarray(param_is_constant, dtype=bool),
        zscore_eps=float(cfg.normalization.params.eps),
    )
    return NormalizationStats(
        state=state_stats,
        params=param_stats,
        transition_time=ParamNormalizationStats(
            param_names=(TRANSITION_TIME_NAME,),
            mean=transition_time_mean_out.astype(np.float64),
            std=transition_time_std.astype(np.float64),
            is_constant=np.asarray(transition_time_is_constant, dtype=bool),
            zscore_eps=float(cfg.normalization.params.eps),
        )
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
    states_norm = normalize_state_tensor(
        payload["checkpoint_states"],
        stats.state,
    ).astype(np.float32)
    params_norm = normalize_params(payload["params"][None, ...], stats.params)[0].astype(np.float32)

    shard_name = f"{src_file.stem}.npz"
    np.savez(
        dst_dir / shard_name,
        states_norm=states_norm,
        params_norm=params_norm,
        checkpoint_days=payload["checkpoint_days"].astype(np.float64),
    )
    return {
        "file": str(Path(dst_dir.name) / shard_name),
        "n_sequences": 1,
        "sequence_length": int(payload["checkpoint_days"].shape[0]),
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
                LOGGER.info(
                    "Reusing cached processed dataset at %s",
                    _display_repo_path(processed_dir),
                )
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
    expected_schedule, live_catalog = _expected_checkpoint_schedule_and_catalog(cfg)
    stats = _fit_stats_streaming(
        train_files=train_files,
        cfg=cfg,
        live_catalog=live_catalog,
    )

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
        state_sequence = np.asarray(first_train_payload["states_norm"], dtype=np.float32)
        checkpoint_days = np.asarray(first_train_payload["checkpoint_days"], dtype=np.float64)

    sample_state = state_sequence[0]
    geometry_meta = _expected_geometry(cfg, nlon=int(sample_state.shape[2]))
    split_sequence_counts = {
        "train": int(sum(entry["n_sequences"] for entry in train_shards)),
        "val": int(sum(entry["n_sequences"] for entry in val_shards)),
        "test": int(sum(entry["n_sequences"] for entry in test_shards)),
    }

    meta: Dict[str, Any] = {
        "task": "checkpoint_sequence_transition",
        "state_fields": list(stats.state.field_names),
        "param_names": list(stats.params.param_names),
        "conditioning_names": list(stats.params.param_names) + list(
            stats.transition_time.param_names
        ),
        "state_shape": {
            "C": int(sample_state.shape[0]),
            "H": int(sample_state.shape[1]),
            "W": int(sample_state.shape[2]),
        },
        "sequence_length": int(state_sequence.shape[0]),
        "checkpoint_days": checkpoint_days.astype(np.float64).tolist(),
        "split_sequence_counts": split_sequence_counts,
        "splits": {"train": train_shards, "val": val_shards, "test": test_shards},
        "live_transition_catalog": _live_transition_catalog_to_json(live_catalog),
        "normalization": stats_to_json(stats),
        "solver": asdict(cfg.solver),
        "sampling": asdict(cfg.sampling),
        "geometry": {
            **asdict(cfg.geometry),
            **geometry_meta,
        },
        "saved_checkpoint_interval_days": float(expected_schedule.interval_days),
        "build_fingerprint": fingerprint,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@dataclass(frozen=True)
class PreloadedSequenceSplit:
    """One processed split materialized as GPU-resident sequence tensors."""

    states: torch.Tensor
    params: torch.Tensor

    @property
    def n_sequences(self) -> int:
        """Return the number of stored simulation sequences."""
        return int(self.states.shape[0])

    @property
    def sequence_length(self) -> int:
        """Return the number of saved checkpoints per sequence."""
        return int(self.states.shape[1])


@dataclass(frozen=True)
class DeviceLiveTransitionCatalog:
    """GPU-resident discrete jump catalog and normalized transition feature."""

    gap_offsets: torch.Tensor
    transition_days: torch.Tensor
    probabilities: torch.Tensor
    transition_time_norm: torch.Tensor
    burn_in_start_index: int


@dataclass(frozen=True)
class LivePairTable:
    """GPU index table describing one epoch of live-sampled transition pairs."""

    sequence_indices: torch.Tensor
    anchor_indices: torch.Tensor
    target_indices: torch.Tensor
    transition_time_norm: torch.Tensor

    @property
    def n_pairs(self) -> int:
        """Return the number of sampled live transition pairs."""
        return int(self.sequence_indices.shape[0])


@dataclass(frozen=True)
class SequencePairSelection:
    """One sampled set of pair indices for a single stored sequence shard."""

    anchor_indices: np.ndarray
    target_indices: np.ndarray
    transition_days: np.ndarray
    transition_time_norm: np.ndarray

    @property
    def n_pairs(self) -> int:
        """Return the number of sampled pairs."""
        return int(self.anchor_indices.shape[0])


@dataclass(frozen=True)
class ResampledSplitPlan:
    """Per-split shard order and sampled pair selections for CPU-side iteration."""

    sequence_order: np.ndarray
    selections: tuple[SequencePairSelection, ...]

    @property
    def n_pairs(self) -> int:
        """Return the number of pairs across all sequences in the split."""
        return int(sum(selection.n_pairs for selection in self.selections))


def _normalize_transition_days_feature(
    transition_days: np.ndarray,
    stats: ParamNormalizationStats,
) -> np.ndarray:
    """Normalize physical transition days into the model's log-time feature."""
    log10_transition_days = np.log10(np.maximum(transition_days.astype(np.float64), 1.0e-30))
    return normalize_params(log10_transition_days.reshape(-1, 1), stats).astype(np.float32)


def _candidate_pair_weights(
    *,
    counts: np.ndarray,
    catalog: LiveTransitionCatalog,
    pair_sampling_policy: str,
) -> np.ndarray:
    """Return per-candidate probabilities for weighted no-replacement sampling."""
    if pair_sampling_policy == "uniform_pairs":
        return np.ones((int(np.sum(counts)),), dtype=np.float64)

    if pair_sampling_policy == "uniform_gaps":
        gap_weights = np.ones_like(catalog.transition_days, dtype=np.float64)
    elif pair_sampling_policy == "inverse_time":
        gap_weights = catalog.probabilities.astype(np.float64, copy=False)
    else:
        raise ValueError(f"Unsupported pair_sampling_policy: {pair_sampling_policy}")

    weights = np.concatenate(
        [
            np.full(int(count), float(gap_weight) / float(count), dtype=np.float64)
            for count, gap_weight in zip(counts.tolist(), gap_weights.tolist(), strict=True)
        ],
        axis=0,
    )
    weights /= np.sum(weights)
    return weights


def _sample_sequence_pair_selection(
    *,
    sequence_length: int,
    catalog: LiveTransitionCatalog,
    pairs_per_sim: int,
    pair_sampling_policy: str,
    transition_time_stats: ParamNormalizationStats,
    seed: int,
) -> SequencePairSelection:
    """Sample one per-sequence pair set from the valid candidate pair space."""
    counts = valid_anchor_counts_for_catalog(
        sequence_length=int(sequence_length),
        catalog=catalog,
    )
    total_candidates = int(np.sum(counts))
    if pairs_per_sim > total_candidates:
        raise ValueError(
            "pairs_per_sim exceeds the number of valid candidate pairs for one sequence: "
            f"pairs_per_sim={pairs_per_sim}, total_candidates={total_candidates}"
        )

    rng = np.random.default_rng(int(seed))
    cumulative_counts = np.cumsum(counts, dtype=np.int64)
    if pair_sampling_policy == "uniform_pairs":
        sampled_ids = np.asarray(
            rng.choice(total_candidates, size=int(pairs_per_sim), replace=False),
            dtype=np.int64,
        )
    else:
        weights = _candidate_pair_weights(
            counts=counts,
            catalog=catalog,
            pair_sampling_policy=str(pair_sampling_policy),
        )
        sampled_ids = np.asarray(
            rng.choice(
                total_candidates,
                size=int(pairs_per_sim),
                replace=False,
                p=weights,
            ),
            dtype=np.int64,
        )

    gap_indices = np.searchsorted(cumulative_counts, sampled_ids, side="right").astype(np.int64)
    prior_counts = np.zeros_like(sampled_ids, dtype=np.int64)
    nonzero_mask = gap_indices > 0
    if np.any(nonzero_mask):
        prior_counts[nonzero_mask] = cumulative_counts[gap_indices[nonzero_mask] - 1]
    anchor_offsets = sampled_ids - prior_counts
    anchor_indices = anchor_offsets + int(catalog.burn_in_start_index)
    target_indices = anchor_indices + catalog.gap_offsets[gap_indices].astype(np.int64)
    transition_days = catalog.transition_days[gap_indices].astype(np.float64)
    transition_time_norm = _normalize_transition_days_feature(
        transition_days.astype(np.float64, copy=False),
        transition_time_stats,
    )
    return SequencePairSelection(
        anchor_indices=anchor_indices.astype(np.int64),
        target_indices=target_indices.astype(np.int64),
        transition_days=transition_days.astype(np.float64),
        transition_time_norm=transition_time_norm.astype(np.float32),
    )


def _build_resampled_split_plan(
    *,
    n_sequences: int,
    sequence_length: int,
    catalog: LiveTransitionCatalog,
    pairs_per_sim: int,
    pair_sampling_policy: str,
    transition_time_stats: ParamNormalizationStats,
    seed: int,
    shuffle_sequences: bool,
) -> ResampledSplitPlan:
    """Build one split plan by resampling a fixed number of pairs per sequence."""
    sequence_order = np.arange(int(n_sequences), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if shuffle_sequences:
        rng.shuffle(sequence_order)

    selections = tuple(
        _sample_sequence_pair_selection(
            sequence_length=int(sequence_length),
            catalog=catalog,
            pairs_per_sim=int(pairs_per_sim),
            pair_sampling_policy=str(pair_sampling_policy),
            transition_time_stats=transition_time_stats,
            seed=int(seed) + 1_000_003 * int(sequence_index),
        )
        for sequence_index in range(int(n_sequences))
    )
    return ResampledSplitPlan(
        sequence_order=sequence_order,
        selections=selections,
    )


def _catalog_to_device(
    *,
    catalog: LiveTransitionCatalog,
    transition_time_stats: ParamNormalizationStats,
    device: torch.device,
) -> DeviceLiveTransitionCatalog:
    """Move the discrete live jump catalog and normalized time feature to GPU."""
    transition_time_norm = _normalize_transition_days_feature(
        catalog.transition_days.astype(np.float64),
        transition_time_stats,
    )
    return DeviceLiveTransitionCatalog(
        gap_offsets=torch.from_numpy(catalog.gap_offsets.astype(np.int64)).to(device=device),
        transition_days=torch.from_numpy(catalog.transition_days.astype(np.float32)).to(device=device),
        probabilities=torch.from_numpy(catalog.probabilities.astype(np.float32)).to(device=device),
        transition_time_norm=torch.from_numpy(transition_time_norm).to(device=device),
        burn_in_start_index=int(catalog.burn_in_start_index),
    )


def _estimate_split_gpu_bytes(
    *,
    processed_meta: Dict[str, Any],
    split_name: str,
    resident_pairs_per_sequence: int,
) -> int:
    """Estimate GPU bytes required for one active processed split and optional pair table."""
    split_counts = dict(processed_meta["split_sequence_counts"])
    n_sequences = int(split_counts[split_name])
    sequence_length = int(processed_meta["sequence_length"])
    state_shape = dict(processed_meta["state_shape"])
    param_dim = int(len(processed_meta["param_names"]))
    state_bytes = (
        n_sequences
        * sequence_length
        * int(state_shape["C"])
        * int(state_shape["H"])
        * int(state_shape["W"])
        * 4
    )
    params_bytes = n_sequences * param_dim * 4
    pair_count = n_sequences * int(resident_pairs_per_sequence)
    pair_table_bytes = pair_count * (8 + 8 + 8 + 4)
    return int(state_bytes + params_bytes + pair_table_bytes)


def _assert_split_fits_gpu(
    *,
    processed_meta: Dict[str, Any],
    split_name: str,
    resident_pairs_per_sequence: int,
    device: torch.device,
) -> None:
    """Fail early when one active split plus fixed headroom cannot fit on the GPU."""
    if device.type != "cuda":
        raise RuntimeError("GPU split residency requires a CUDA device")
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    estimated_bytes = _estimate_split_gpu_bytes(
        processed_meta=processed_meta,
        split_name=split_name,
        resident_pairs_per_sequence=resident_pairs_per_sequence,
    )
    required_bytes = int(math.ceil(float(estimated_bytes) * VRAM_HEADROOM_FACTOR))
    if required_bytes > total_memory:
        gib = 1024.0 ** 3
        raise RuntimeError(
            f"Processed split `{split_name}` cannot fit on {device} with the required "
            f"{VRAM_HEADROOM_FACTOR:.0%} VRAM headroom: estimated={estimated_bytes / gib:.2f} GiB, "
            f"required_with_headroom={required_bytes / gib:.2f} GiB, "
            f"device_total={total_memory / gib:.2f} GiB"
        )


def _load_sequence_split_to_device(
    *,
    processed_dir: Path,
    shard_entries: Sequence[Dict[str, Any]],
    device: torch.device,
) -> PreloadedSequenceSplit:
    """Load one processed split into contiguous GPU tensors."""
    state_rows: List[np.ndarray] = []
    params_rows: List[np.ndarray] = []

    for entry in shard_entries:
        shard_path = processed_dir / str(entry["file"])
        with np.load(shard_path, allow_pickle=False) as npz:
            states = np.asarray(npz["states_norm"], dtype=np.float32)
            params_norm = np.asarray(npz["params_norm"], dtype=np.float32)
        if states.ndim != 4:
            raise ValueError(f"Processed shard {shard_path} must store rank-4 state sequences")
        if params_norm.ndim != 1:
            raise ValueError(f"Processed shard {shard_path} params_norm must be rank-1")
        state_rows.append(states)
        params_rows.append(params_norm)

    if not state_rows:
        raise RuntimeError("Cannot preload an empty sequence split")

    states_tensor = torch.from_numpy(np.stack(state_rows, axis=0)).to(device=device)
    params_tensor = torch.from_numpy(np.stack(params_rows, axis=0)).to(device=device)
    return PreloadedSequenceSplit(
        states=states_tensor,
        params=params_tensor,
    )


def _release_preloaded_sequence_split(
    split: PreloadedSequenceSplit | None,
) -> PreloadedSequenceSplit | None:
    """Drop one active split reference before another split is loaded onto the GPU."""
    if split is None:
        return None
    del split
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def _sample_live_pair_table(
    *,
    split: PreloadedSequenceSplit,
    catalog: DeviceLiveTransitionCatalog,
    live_pairs_per_sequence: int,
    seed: int,
    shuffle_pairs: bool,
) -> LivePairTable:
    """Sample one deterministic GPU-resident live pair table for a split."""
    if split.n_sequences < 1:
        raise ValueError("Split must contain at least one sequence")
    if live_pairs_per_sequence < 1:
        raise ValueError("live_pairs_per_sequence must be >= 1")

    device = split.states.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    sequence_indices = torch.arange(split.n_sequences, device=device, dtype=torch.long)
    sequence_indices = sequence_indices.repeat_interleave(int(live_pairs_per_sequence))
    n_pairs = int(sequence_indices.shape[0])

    catalog_indices = torch.multinomial(
        catalog.probabilities,
        num_samples=n_pairs,
        replacement=True,
        generator=generator,
    )
    gap_offsets = torch.index_select(catalog.gap_offsets, 0, catalog_indices)
    transition_time_norm = torch.index_select(
        catalog.transition_time_norm,
        0,
        catalog_indices,
    )

    min_anchor = int(catalog.burn_in_start_index)
    max_anchor = int(split.sequence_length) - 1 - gap_offsets
    valid_anchor_counts = max_anchor - int(min_anchor) + 1
    if torch.any(valid_anchor_counts <= 0):
        raise RuntimeError("Live jump catalog produced an invalid anchor range")
    anchor_uniform = torch.rand((n_pairs,), generator=generator, device=device)
    anchor_indices = (
        torch.floor(anchor_uniform * valid_anchor_counts.to(dtype=torch.float32)).to(torch.long)
        + int(min_anchor)
    )
    target_indices = anchor_indices + gap_offsets

    if shuffle_pairs:
        permutation = torch.randperm(n_pairs, generator=generator, device=device)
        sequence_indices = torch.index_select(sequence_indices, 0, permutation)
        anchor_indices = torch.index_select(anchor_indices, 0, permutation)
        target_indices = torch.index_select(target_indices, 0, permutation)
        transition_time_norm = torch.index_select(transition_time_norm, 0, permutation)

    return LivePairTable(
        sequence_indices=sequence_indices,
        anchor_indices=anchor_indices,
        target_indices=target_indices,
        transition_time_norm=transition_time_norm,
    )


def _iter_live_pair_batches(
    *,
    split: PreloadedSequenceSplit,
    pair_table: LivePairTable,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield GPU batches gathered directly from one sequence split and pair table."""
    total_pairs = int(pair_table.n_pairs)
    for start in range(0, total_pairs, int(batch_size)):
        end = min(start + int(batch_size), total_pairs)
        sequence_indices = pair_table.sequence_indices[start:end]
        anchor_indices = pair_table.anchor_indices[start:end]
        target_indices = pair_table.target_indices[start:end]
        state_inputs = split.states[sequence_indices, anchor_indices]
        state_targets = split.states[sequence_indices, target_indices]
        params_norm = torch.index_select(split.params, 0, sequence_indices)  # shape: (B, P)
        conditioning = torch.cat(
            (params_norm, pair_table.transition_time_norm[start:end]),
            dim=1,
        )  # shape: (B, P + 1)
        yield conditioning, state_inputs, state_targets


def _iter_resampled_pair_batches(
    *,
    processed_dir: Path,
    shard_entries: Sequence[Dict[str, Any]],
    plan: ResampledSplitPlan,
    batch_size: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield GPU batches by resampling CPU-side pairs from stored sequence shards."""
    if len(shard_entries) != len(plan.selections):
        raise ValueError("shard_entries and plan.selections must have the same length")

    for sequence_index in plan.sequence_order.tolist():
        shard_path = processed_dir / str(shard_entries[int(sequence_index)]["file"])
        selection = plan.selections[int(sequence_index)]
        with np.load(shard_path, allow_pickle=False) as npz:
            states = np.asarray(npz["states_norm"], dtype=np.float32)
            params_norm = np.asarray(npz["params_norm"], dtype=np.float32)

        state_inputs = states[selection.anchor_indices]
        state_targets = states[selection.target_indices]
        params_batch = np.repeat(params_norm[None, :], int(selection.n_pairs), axis=0)
        conditioning = np.concatenate(
            [params_batch, selection.transition_time_norm.astype(np.float32, copy=False)],
            axis=1,
        ).astype(np.float32, copy=False)

        for start in range(0, int(selection.n_pairs), int(batch_size)):
            end = min(start + int(batch_size), int(selection.n_pairs))
            conditioning_batch = torch.from_numpy(conditioning[start:end]).to(device=device)
            state_input_batch = torch.from_numpy(state_inputs[start:end]).to(device=device)
            state_target_batch = torch.from_numpy(state_targets[start:end]).to(device=device)
            yield conditioning_batch, state_input_batch, state_target_batch


def _iter_resampled_pair_batches_preloaded(
    *,
    split: PreloadedSequenceSplit,
    plan: ResampledSplitPlan,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield resampled pair batches directly from one GPU-resident sequence split."""
    if int(split.n_sequences) != len(plan.selections):
        raise ValueError("split.n_sequences and plan.selections must have the same length")

    device = split.states.device
    for sequence_index in plan.sequence_order.tolist():
        selection = plan.selections[int(sequence_index)]
        anchor_indices = torch.from_numpy(selection.anchor_indices).to(device=device, dtype=torch.long)
        target_indices = torch.from_numpy(selection.target_indices).to(device=device, dtype=torch.long)
        transition_time_norm = torch.from_numpy(
            selection.transition_time_norm.astype(np.float32, copy=False)
        ).to(device=device)
        params_norm = split.params[int(sequence_index)].unsqueeze(0)

        for start in range(0, int(selection.n_pairs), int(batch_size)):
            end = min(start + int(batch_size), int(selection.n_pairs))
            anchor_batch = anchor_indices[start:end]
            target_batch = target_indices[start:end]
            state_input_batch = split.states[int(sequence_index), anchor_batch]
            state_target_batch = split.states[int(sequence_index), target_batch]
            params_batch = params_norm.expand(end - start, -1)
            conditioning_batch = torch.cat(
                (params_batch, transition_time_norm[start:end]),
                dim=1,
            )
            yield conditioning_batch, state_input_batch, state_target_batch


def _collect_predictions(
    *,
    model: torch.nn.Module,
    split: PreloadedSequenceSplit,
    pair_table: LivePairTable,
    batch_size: int,
    device: torch.device,
    amp_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the transition model on one live-sampled split and collect predictions/targets."""
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for conditioning_batch, state_input_batch, state_target_batch in _iter_live_pair_batches(
            split=split,
            pair_table=pair_table,
            batch_size=batch_size,
        ):
            with autocast_context(device, amp_mode):
                pred_batch = model(state_input_batch, conditioning_batch)
            predictions.append(pred_batch.detach().float().cpu().numpy())
            targets.append(state_target_batch.detach().float().cpu().numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def _collect_predictions_resampled(
    *,
    model: torch.nn.Module,
    processed_dir: Path,
    shard_entries: Sequence[Dict[str, Any]],
    plan: ResampledSplitPlan,
    batch_size: int,
    device: torch.device,
    amp_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the transition model on one CPU-side resampled split and collect outputs."""
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for conditioning_batch, state_input_batch, state_target_batch in _iter_resampled_pair_batches(
            processed_dir=processed_dir,
            shard_entries=shard_entries,
            plan=plan,
            batch_size=batch_size,
            device=device,
        ):
            with autocast_context(device, amp_mode):
                pred_batch = model(state_input_batch, conditioning_batch)
            predictions.append(pred_batch.detach().float().cpu().numpy())
            targets.append(state_target_batch.detach().float().cpu().numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def _collect_predictions_resampled_preloaded(
    *,
    model: torch.nn.Module,
    split: PreloadedSequenceSplit,
    plan: ResampledSplitPlan,
    batch_size: int,
    device: torch.device,
    amp_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the transition model on one GPU-preloaded resampled split and collect outputs."""
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for conditioning_batch, state_input_batch, state_target_batch in _iter_resampled_pair_batches_preloaded(
            split=split,
            plan=plan,
            batch_size=batch_size,
        ):
            with autocast_context(device, amp_mode):
                pred_batch = model(state_input_batch, conditioning_batch)
            predictions.append(pred_batch.detach().float().cpu().numpy())
            targets.append(state_target_batch.detach().float().cpu().numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def _named_scalar_map(values: np.ndarray, field_names: Sequence[str]) -> Dict[str, float]:
    """Map channel-aligned scalars to ``{field_name: value}``."""
    return {str(field_names[index]): float(values[index]) for index in range(len(field_names))}


def _ordered_channel_loss_weights(
    *,
    field_names: Sequence[str],
    configured_weights: Dict[str, float] | None,
) -> np.ndarray:
    """Return channel loss weights in the model's stable field order."""
    if configured_weights is None:
        return np.ones((len(field_names),), dtype=np.float32)
    return np.asarray(
        [float(configured_weights[str(field_name)]) for field_name in field_names],
        dtype=np.float32,
    )


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
    """Compute normalized, physical, and spherical-spectrum single-call metrics.

    Args:
        pred_norm:
            Normalized predictions with shape ``[N, C, H, W]``.
        target_norm:
            Normalized targets with shape ``[N, C, H, W]`` for the same ``C``
            prognostic channels.
    """
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
    if not history:
        raise ValueError("history must contain at least one row")

    field_names: list[str] = []
    for row in history:
        for key in row:
            if key not in field_names:
                field_names.append(str(key))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in history:
            formatted_row: Dict[str, str] = {}
            for key in field_names:
                value = row[key]
                if key == "epoch":
                    formatted_row[key] = str(int(round(float(value))))
                else:
                    formatted_row[key] = _format_scientific(float(value))
            writer.writerow(formatted_row)


def _format_scientific(value: float, *, digits: int = 4) -> str:
    """Format one scalar with fixed significant figures in scientific notation."""
    if digits < 1:
        raise ValueError("digits must be >= 1")
    precision = int(digits) - 1
    return f"{float(value):.{precision}e}"


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


def _update_loss_tracking(
    *,
    current: float,
    best: float,
    bad_epochs: int,
    min_delta: float,
) -> Tuple[float, int, bool]:
    """Update the best loss and consecutive bad-epoch count for one metric."""
    if bad_epochs < 0:
        raise ValueError("bad_epochs must be >= 0")
    improved = _loss_improved(current=current, best=best, min_delta=min_delta)
    if improved:
        return float(current), 0, True
    return float(best), int(bad_epochs) + 1, False


def _reduce_plateau_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float,
    min_lr: float,
    eps: float,
) -> bool:
    """Apply one plateau LR reduction step across all optimizer parameter groups."""
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError("factor must be in (0,1)")
    if min_lr < 0.0:
        raise ValueError("min_lr must be >= 0")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    reduced = False
    for group in optimizer.param_groups:
        current_lr = float(group["lr"])
        new_lr = max(float(min_lr), current_lr * float(factor))
        if (current_lr - new_lr) > float(eps):
            group["lr"] = float(new_lr)
            reduced = True
    return reduced


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


def _plateau_reduce_threshold(*, scheduler_patience: int) -> int:
    """Return the bad-epoch count required for one plateau LR reduction.

    ``scheduler_patience=0`` is supported and means "reduce after the first
    non-improving epoch", not "reduce immediately on an improving epoch."
    """
    if scheduler_patience < 0:
        raise ValueError("scheduler_patience must be >= 0")
    return max(1, int(scheduler_patience))


def _set_determinism(*, seed: int, deterministic: bool) -> None:
    """Apply reproducible random seeds and optional deterministic-kernel settings."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def train_emulator(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Train the transition emulator end-to-end and persist artifacts."""
    device = choose_device(cfg.training.device)
    pair_iteration_mode = str(cfg.training.pair_iteration_mode)
    preload_to_gpu = bool(cfg.training.preload_to_gpu)
    if pair_iteration_mode == "live_sampled_gpu" and not preload_to_gpu:
        raise RuntimeError(
            "This training path requires training.preload_to_gpu=true so live pair sampling stays on GPU"
        )
    if device.type != "cuda":
        raise RuntimeError("training requires CUDA")

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
    state_field_names = list(processed_meta["state_fields"])

    _set_determinism(
        seed=int(cfg.training.seed),
        deterministic=bool(cfg.training.deterministic),
    )
    batch_size = int(cfg.training.batch_size)
    state_shape = dict(processed_meta["state_shape"])
    state_chans = int(state_shape["C"])
    nlat = int(state_shape["H"])
    nlon = int(state_shape["W"])
    conditioning_dim = int(len(processed_meta["conditioning_names"]))
    channel_loss_weights = _ordered_channel_loss_weights(
        field_names=state_field_names,
        configured_weights=cfg.training.channel_loss_weights,
    )
    live_catalog = _live_transition_catalog_from_json(processed_meta["live_transition_catalog"])
    live_pairs_per_sequence = int(cfg.sampling.live_pairs_per_sequence)
    pairs_per_sim = int(cfg.sampling.pairs_per_sim)
    pair_sampling_policy = str(cfg.sampling.pair_sampling_policy)
    sequence_batch_size = (
        batch_size // live_pairs_per_sequence
        if pair_iteration_mode == "live_sampled_gpu"
        else None
    )
    device_catalog = None
    if pair_iteration_mode == "live_sampled_gpu":
        device_catalog = _catalog_to_device(
            catalog=live_catalog,
            transition_time_stats=stats.transition_time,
            device=device,
        )
    if preload_to_gpu:
        resident_pairs_per_sequence = (
            live_pairs_per_sequence if pair_iteration_mode == "live_sampled_gpu" else 0
        )
        for split_name in ("train", "val", "test"):
            _assert_split_fits_gpu(
                processed_meta=processed_meta,
                split_name=split_name,
                resident_pairs_per_sequence=resident_pairs_per_sequence,
                device=device,
            )

    model = build_state_conditioned_transition_model(
        img_size=(nlat, nlon),
        input_state_chans=state_chans,
        target_state_chans=state_chans,
        param_dim=conditioning_dim,
        cfg_model=cfg.model,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(
        "Model parameters | total=%s | trainable=%s",
        f"{n_params:,}",
        f"{n_trainable:,}",
    )
    runtime_amp_mode = _resolve_runtime_amp_mode(
        requested_amp_mode=cfg.training.amp_mode,
        nlat=nlat,
        nlon=nlon,
    )
    if pair_iteration_mode == "live_sampled_gpu":
        LOGGER.info(
            "Training runtime | mode=%s | device=%s | amp=%s | deterministic=%s | "
            "pair_batch=%s | sequence_batch=%s | preload_to_gpu=%s | split_sequence_counts=%s",
            pair_iteration_mode,
            device,
            runtime_amp_mode,
            str(bool(cfg.training.deterministic)).lower(),
            _format_scientific(float(batch_size)),
            _format_scientific(float(sequence_batch_size if sequence_batch_size is not None else 0)),
            "true",
            json.dumps(processed_meta["split_sequence_counts"], sort_keys=True),
        )
    else:
        LOGGER.info(
            "Training runtime | mode=%s | device=%s | amp=%s | deterministic=%s | "
            "pair_batch=%s | pairs_per_sim=%s | pair_sampling_policy=%s | preload_to_gpu=%s | "
            "split_sequence_counts=%s",
            pair_iteration_mode,
            device,
            runtime_amp_mode,
            str(bool(cfg.training.deterministic)).lower(),
            _format_scientific(float(batch_size)),
            _format_scientific(float(pairs_per_sim)),
            pair_sampling_policy,
            str(bool(preload_to_gpu)).lower(),
            json.dumps(processed_meta["split_sequence_counts"], sort_keys=True),
        )

    grad_clip_norm = float(cfg.training.grad_clip_norm)
    loss_fn = SphereLoss(
        nlat=nlat,
        nlon=nlon,
        grid=cfg.model.grid,
        channel_weights=channel_loss_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler_type = str(cfg.training.scheduler.type)
    scheduler_patience = int(cfg.training.scheduler.patience)
    scheduler_min_lr = float(cfg.training.scheduler.min_lr)
    scheduler_eps = float(cfg.training.scheduler.eps)
    scheduler_factor = float(cfg.training.scheduler.factor)
    warmup_epochs = int(cfg.training.scheduler.warmup_epochs)
    plateau_reduce_threshold = _plateau_reduce_threshold(
        scheduler_patience=scheduler_patience,
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
    plateau_bad_epochs = 0
    early_stop_patience = _early_stopping_patience(
        scheduler_patience=scheduler_patience,
        warmup_epochs=warmup_epochs,
    )
    val_pair_seed = int(cfg.training.seed) + VAL_PAIR_SEED_OFFSET
    test_pair_seed = int(cfg.training.seed) + TEST_PAIR_SEED_OFFSET
    val_pair_table: LivePairTable | None = None
    test_pair_table: LivePairTable | None = None
    val_resampled_plan: ResampledSplitPlan | None = None
    test_resampled_plan: ResampledSplitPlan | None = None

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
        train_channel_loss_sum = np.zeros((state_chans,), dtype=np.float64)
        train_count = 0
        train_phase_start = time.perf_counter()
        if pair_iteration_mode == "live_sampled_gpu":
            train_split = _load_sequence_split_to_device(
                processed_dir=processed_dir,
                shard_entries=processed_meta["splits"]["train"],
                device=device,
            )
            train_pair_table = _sample_live_pair_table(
                split=train_split,
                catalog=device_catalog,
                live_pairs_per_sequence=live_pairs_per_sequence,
                seed=int(cfg.training.seed) + int(epoch),
                shuffle_pairs=bool(cfg.training.shuffle),
            )
            train_samples = int(train_pair_table.n_pairs)
            train_batches = _iter_live_pair_batches(
                split=train_split,
                pair_table=train_pair_table,
                batch_size=batch_size,
            )
        else:
            train_split = (
                _load_sequence_split_to_device(
                    processed_dir=processed_dir,
                    shard_entries=processed_meta["splits"]["train"],
                    device=device,
                )
                if preload_to_gpu
                else None
            )
            train_pair_table = None
            train_plan = _build_resampled_split_plan(
                n_sequences=int(processed_meta["split_sequence_counts"]["train"]),
                sequence_length=int(processed_meta["sequence_length"]),
                catalog=live_catalog,
                pairs_per_sim=pairs_per_sim,
                pair_sampling_policy=pair_sampling_policy,
                transition_time_stats=stats.transition_time,
                seed=int(cfg.training.seed) + int(epoch),
                shuffle_sequences=bool(cfg.training.shuffle),
            )
            train_samples = int(train_plan.n_pairs)
            if preload_to_gpu and train_split is not None:
                train_batches = _iter_resampled_pair_batches_preloaded(
                    split=train_split,
                    plan=train_plan,
                    batch_size=batch_size,
                )
            else:
                train_batches = _iter_resampled_pair_batches(
                    processed_dir=processed_dir,
                    shard_entries=processed_meta["splits"]["train"],
                    plan=train_plan,
                    batch_size=batch_size,
                    device=device,
                )

        for conditioning_batch, state_input_batch, state_target_batch in train_batches:
            _check_finite_tensor(conditioning_batch, name="train conditioning batch")
            _check_finite_tensor(state_input_batch, name="train state_input batch")
            _check_finite_tensor(state_target_batch, name="train target batch")

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, runtime_amp_mode):
                prediction = model(state_input_batch, conditioning_batch)
                _check_finite_tensor(prediction, name="train prediction batch")
                loss, per_channel_loss = loss_fn.loss_with_channels(
                    prediction,
                    state_target_batch,
                )
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite")

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            train_loss_sum += float(loss.detach().item())
            train_channel_loss_sum += per_channel_loss.detach().cpu().numpy().astype(np.float64)
            train_count += 1
        train_seconds = time.perf_counter() - train_phase_start
        if train_pair_table is not None:
            del train_pair_table
        if train_split is not None:
            train_split = _release_preloaded_sequence_split(train_split)

        model.eval()
        val_loss_sum = 0.0
        val_channel_loss_sum = np.zeros((state_chans,), dtype=np.float64)
        val_count = 0
        val_phase_start = time.perf_counter()
        if pair_iteration_mode == "live_sampled_gpu":
            val_split = _load_sequence_split_to_device(
                processed_dir=processed_dir,
                shard_entries=processed_meta["splits"]["val"],
                device=device,
            )
            if val_pair_table is None:
                val_pair_table = _sample_live_pair_table(
                    split=val_split,
                    catalog=device_catalog,
                    live_pairs_per_sequence=live_pairs_per_sequence,
                    seed=val_pair_seed,
                    shuffle_pairs=False,
                )
            val_samples = int(val_pair_table.n_pairs)
            val_batches = _iter_live_pair_batches(
                split=val_split,
                pair_table=val_pair_table,
                batch_size=batch_size,
            )
        else:
            val_split = (
                _load_sequence_split_to_device(
                    processed_dir=processed_dir,
                    shard_entries=processed_meta["splits"]["val"],
                    device=device,
                )
                if preload_to_gpu
                else None
            )
            if val_resampled_plan is None:
                val_resampled_plan = _build_resampled_split_plan(
                    n_sequences=int(processed_meta["split_sequence_counts"]["val"]),
                    sequence_length=int(processed_meta["sequence_length"]),
                    catalog=live_catalog,
                    pairs_per_sim=pairs_per_sim,
                    pair_sampling_policy=pair_sampling_policy,
                    transition_time_stats=stats.transition_time,
                    seed=val_pair_seed,
                    shuffle_sequences=False,
                )
            val_samples = int(val_resampled_plan.n_pairs)
            if preload_to_gpu and val_split is not None:
                val_batches = _iter_resampled_pair_batches_preloaded(
                    split=val_split,
                    plan=val_resampled_plan,
                    batch_size=batch_size,
                )
            else:
                val_batches = _iter_resampled_pair_batches(
                    processed_dir=processed_dir,
                    shard_entries=processed_meta["splits"]["val"],
                    plan=val_resampled_plan,
                    batch_size=batch_size,
                    device=device,
                )

        with torch.no_grad():
            for conditioning_batch, state_input_batch, state_target_batch in val_batches:
                _check_finite_tensor(conditioning_batch, name="val conditioning batch")
                _check_finite_tensor(state_input_batch, name="val state_input batch")
                _check_finite_tensor(state_target_batch, name="val target batch")

                with autocast_context(device, runtime_amp_mode):
                    prediction = model(state_input_batch, conditioning_batch)
                    _check_finite_tensor(prediction, name="val prediction batch")
                    vloss, val_per_channel_loss = loss_fn.loss_with_channels(
                        prediction,
                        state_target_batch,
                    )
                if not torch.isfinite(vloss).item():
                    raise RuntimeError("Validation loss became non-finite")
                val_loss_sum += float(vloss.detach().item())
                val_channel_loss_sum += (
                    val_per_channel_loss.detach().cpu().numpy().astype(np.float64)
                )
                val_count += 1
        val_seconds = time.perf_counter() - val_phase_start
        if val_split is not None:
            val_split = _release_preloaded_sequence_split(val_split)

        if train_count == 0 or val_count == 0:
            raise RuntimeError("No training or validation batches were produced")
        train_loss = train_loss_sum / float(train_count)
        val_loss = val_loss_sum / float(val_count)
        train_channel_loss = train_channel_loss_sum / float(train_count)
        val_channel_loss = val_channel_loss_sum / float(val_count)
        best_val, epochs_without_improvement, val_improved = _update_loss_tracking(
            current=val_loss,
            best=best_val,
            bad_epochs=epochs_without_improvement,
            min_delta=scheduler_eps,
        )
        if scheduler_type == "plateau":
            if val_improved:
                plateau_bad_epochs = 0
            else:
                plateau_bad_epochs += 1
            if epoch >= warmup_epochs and plateau_bad_epochs >= plateau_reduce_threshold:
                if _reduce_plateau_learning_rate(
                    optimizer,
                    factor=scheduler_factor,
                    min_lr=scheduler_min_lr,
                    eps=scheduler_eps,
                ):
                    plateau_bad_epochs = 0

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_start
        train_samples_per_second = float(train_samples) / max(train_seconds, 1.0e-12)
        val_samples_per_second = float(val_samples) / max(val_seconds, 1.0e-12)
        LOGGER.info(
            "Epoch %4d/%4d | train=%s | val=%s | lr=%s | train_t=%s s | "
            "val_t=%s s | epoch_t=%s s | train_rate=%s samples/s | "
            "val_rate=%s samples/s",
            epoch,
            epochs,
            _format_scientific(train_loss),
            _format_scientific(val_loss),
            _format_scientific(current_lr),
            _format_scientific(train_seconds),
            _format_scientific(val_seconds),
            _format_scientific(epoch_seconds),
            _format_scientific(train_samples_per_second),
            _format_scientific(val_samples_per_second),
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{
                    f"train_loss_{field_name.lower()}": float(train_channel_loss[field_index])
                    for field_index, field_name in enumerate(state_field_names)
                },
                **{
                    f"val_loss_{field_name.lower()}": float(val_channel_loss[field_index])
                    for field_index, field_name in enumerate(state_field_names)
                },
                "lr": current_lr,
                "train_seconds": float(train_seconds),
                "val_seconds": float(val_seconds),
                "epoch_seconds": epoch_seconds,
                "train_samples": float(train_samples),
                "val_samples": float(val_samples),
                "train_samples_per_second": float(train_samples_per_second),
                "val_samples_per_second": float(val_samples_per_second),
            }
        )

        # Checkpoints include enough metadata for export and downstream tools to
        # reconstruct shapes, normalization, and dataset conventions.
        checkpoint = {
            "mode": "state_conditioned_prognostic_transition",
            "model_state": model.state_dict(),
            "state_fields": state_field_names,
            "param_names": list(processed_meta["param_names"]),
            "conditioning_names": list(processed_meta["conditioning_names"]),
            "shape": {
                "C": state_chans,
                "H": nlat,
                "W": nlon,
            },
            "geometry": dict(processed_meta["geometry"]),
            "sequence_length": int(processed_meta["sequence_length"]),
            "live_transition_catalog": dict(processed_meta["live_transition_catalog"]),
            "normalization": stats_to_json(stats),
            "solver": asdict(cfg.solver),
            "sampling": asdict(cfg.sampling),
            "model_config": asdict(cfg.model),
            "training_config": asdict(cfg.training),
            "runtime_amp_mode": runtime_amp_mode,
            "resolved_config": resolved_cfg_dict,
            "source_config_path": _display_repo_path(config_path),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_per_channel_loss": _named_scalar_map(train_channel_loss, state_field_names),
            "val_per_channel_loss": _named_scalar_map(val_channel_loss, state_field_names),
            "learning_rate": float(current_lr),
            "epoch_seconds": float(epoch_seconds),
        }
        torch.save(checkpoint, last_path)
        if val_improved:
            torch.save(checkpoint, best_path)

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
    if pair_iteration_mode == "live_sampled_gpu":
        val_split = _load_sequence_split_to_device(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["val"],
            device=device,
        )
        if val_pair_table is None:
            val_pair_table = _sample_live_pair_table(
                split=val_split,
                catalog=device_catalog,
                live_pairs_per_sequence=live_pairs_per_sequence,
                seed=val_pair_seed,
                shuffle_pairs=False,
            )
        val_pred, val_target = _collect_predictions(
            model=model,
            split=val_split,
            pair_table=val_pair_table,
            batch_size=batch_size,
            device=device,
            amp_mode=runtime_amp_mode,
        )
        val_split = _release_preloaded_sequence_split(val_split)
        test_split = _load_sequence_split_to_device(
            processed_dir=processed_dir,
            shard_entries=processed_meta["splits"]["test"],
            device=device,
        )
        if test_pair_table is None:
            test_pair_table = _sample_live_pair_table(
                split=test_split,
                catalog=device_catalog,
                live_pairs_per_sequence=live_pairs_per_sequence,
                seed=test_pair_seed,
                shuffle_pairs=False,
            )
        test_pred, test_target = _collect_predictions(
            model=model,
            split=test_split,
            pair_table=test_pair_table,
            batch_size=batch_size,
            device=device,
            amp_mode=runtime_amp_mode,
        )
        test_split = _release_preloaded_sequence_split(test_split)
    else:
        if val_resampled_plan is None:
            val_resampled_plan = _build_resampled_split_plan(
                n_sequences=int(processed_meta["split_sequence_counts"]["val"]),
                sequence_length=int(processed_meta["sequence_length"]),
                catalog=live_catalog,
                pairs_per_sim=pairs_per_sim,
                pair_sampling_policy=pair_sampling_policy,
                transition_time_stats=stats.transition_time,
                seed=val_pair_seed,
                shuffle_sequences=False,
            )
        if test_resampled_plan is None:
            test_resampled_plan = _build_resampled_split_plan(
                n_sequences=int(processed_meta["split_sequence_counts"]["test"]),
                sequence_length=int(processed_meta["sequence_length"]),
                catalog=live_catalog,
                pairs_per_sim=pairs_per_sim,
                pair_sampling_policy=pair_sampling_policy,
                transition_time_stats=stats.transition_time,
                seed=test_pair_seed,
                shuffle_sequences=False,
            )
        if preload_to_gpu:
            val_split = _load_sequence_split_to_device(
                processed_dir=processed_dir,
                shard_entries=processed_meta["splits"]["val"],
                device=device,
            )
            val_pred, val_target = _collect_predictions_resampled_preloaded(
                model=model,
                split=val_split,
                plan=val_resampled_plan,
                batch_size=batch_size,
                device=device,
                amp_mode=runtime_amp_mode,
            )
            val_split = _release_preloaded_sequence_split(val_split)
            test_split = _load_sequence_split_to_device(
                processed_dir=processed_dir,
                shard_entries=processed_meta["splits"]["test"],
                device=device,
            )
            test_pred, test_target = _collect_predictions_resampled_preloaded(
                model=model,
                split=test_split,
                plan=test_resampled_plan,
                batch_size=batch_size,
                device=device,
                amp_mode=runtime_amp_mode,
            )
            test_split = _release_preloaded_sequence_split(test_split)
        else:
            val_pred, val_target = _collect_predictions_resampled(
                model=model,
                processed_dir=processed_dir,
                shard_entries=processed_meta["splits"]["val"],
                plan=val_resampled_plan,
                batch_size=batch_size,
                device=device,
                amp_mode=runtime_amp_mode,
            )
            test_pred, test_target = _collect_predictions_resampled(
                model=model,
                processed_dir=processed_dir,
                shard_entries=processed_meta["splits"]["test"],
                plan=test_resampled_plan,
                batch_size=batch_size,
                device=device,
                amp_mode=runtime_amp_mode,
            )
    val_metrics = _compute_one_step_metrics(
        pred_norm=val_pred,
        target_norm=val_target,
        field_names=state_field_names,
        state_stats=stats.state,
        grid=cfg.model.grid,
    )
    test_metrics = _compute_one_step_metrics(
        pred_norm=test_pred,
        target_norm=test_target,
        field_names=state_field_names,
        state_stats=stats.state,
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
        "best_checkpoint": _display_repo_path(best_path),
        "last_checkpoint": _display_repo_path(last_path),
        "best_val_loss": float(best_val),
        "history_path": _display_repo_path(model_dir / "training_history.json"),
        "history_csv_path": _display_repo_path(model_dir / "training_history.csv"),
        "val_metrics_path": _display_repo_path(model_dir / "val_metrics.json"),
        "test_metrics_path": _display_repo_path(model_dir / "test_metrics.json"),
        "processed_meta": _display_repo_path(processed_dir / "processed_meta.json"),
    }
