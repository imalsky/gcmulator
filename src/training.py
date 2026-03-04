"""Preprocessing and training pipeline for terminal-state emulator learning."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import GCMulatorConfig, resolve_path, time_days_to_rollout_steps
from .geometry import geometry_shift_for_nlon
from .modeling import (
    SphereLoss,
    autocast_context,
    build_rollout_model,
    choose_device,
    ensure_torch_harmonics_importable,
)
from .normalization import (
    NormalizationStats,
    apply_state_transforms,
    normalize_params,
    normalize_states,
    stats_from_json,
    stats_to_json,
)

LOGGER = logging.getLogger("src.train")

# Lower bound used when computing z-score denominators from empirical variance.
STD_FLOOR = 1.0e-12
FINITE_CHECK_INTERVAL = 25
PREPROCESS_FINGERPRINT_VERSION = 1


def _list_raw_dataset_files(dataset_dir: Path) -> List[Path]:
    """List raw simulation files and fail when dataset is empty."""
    files = sorted(dataset_dir.glob("sim_*.npz"))
    if not files:
        raise FileNotFoundError(f"No sim_*.npz files found in {dataset_dir}")
    return files


def _split_files(files: Sequence[Path], *, seed: int, val_fraction: float) -> Tuple[List[Path], List[Path]]:
    """Create reproducible file-level train/validation split."""
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"training.val_fraction must be in (0,1), got {val_fraction}")
    idx = np.arange(len(files))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(float(val_fraction) * len(files))))
    val_ids = set(idx[:n_val].tolist())
    train = [files[i] for i in range(len(files)) if i not in val_ids]
    val = [files[i] for i in range(len(files)) if i in val_ids]
    if not train:
        raise ValueError(
            f"Train split is empty (n_files={len(files)}, val_fraction={val_fraction}). "
            "Increase dataset size or reduce val_fraction."
        )
    if not val:
        raise ValueError(
            f"Validation split is empty (n_files={len(files)}, val_fraction={val_fraction}). "
            "Increase dataset size or increase val_fraction."
        )
    return train, val


def _load_raw_metadata(file_path: Path) -> Dict[str, Any]:
    """Load metadata from one raw simulation file without touching state arrays."""
    with np.load(file_path, allow_pickle=True) as z:
        try:
            lat_order = str(np.asarray(z["lat_order"], dtype=object).item())
            lon_origin = str(np.asarray(z["lon_origin"], dtype=object).item())
            lon_shift = int(np.asarray(z["lon_shift"], dtype=np.int64).item())
            nlat = int(np.asarray(z["nlat"], dtype=np.int64).item())
            nlon = int(np.asarray(z["nlon"], dtype=np.int64).item())
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
            }
        else:
            current = (
                str(meta["lat_order"]),
                str(meta["lon_origin"]),
                int(meta["lon_shift"]),
                int(meta["nlat"]),
                int(meta["nlon"]),
            )
            baseline = (
                str(ref["lat_order"]),
                str(ref["lon_origin"]),
                int(ref["lon_shift"]),
                int(ref["nlat"]),
                int(ref["nlon"]),
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
        "normalization": asdict(cfg.normalization),
        "geometry": asdict(cfg.geometry),
        "solver": {
            "M": int(cfg.solver.M),
            "dt_seconds": float(cfg.solver.dt_seconds),
            "default_time_days": float(cfg.solver.default_time_days),
        },
        "model_time_mapping": {
            "default_time_days": float(cfg.solver.default_time_days),
            "rollout_steps_at_default_time": int(cfg.model.rollout_steps_at_default_time),
        },
        "raw_files": [_raw_file_signature(fp) for fp in files],
    }


def _processed_cache_is_valid(*, meta: Dict[str, Any], fingerprint: Dict[str, Any], processed_dir: Path) -> bool:
    """Return True when processed data matches the current preprocessing fingerprint."""
    if meta.get("build_fingerprint") != fingerprint:
        return False
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        return False
    for split_name in ("train", "val"):
        names = splits.get(split_name)
        if not isinstance(names, list) or not names:
            return False
        for name in names:
            if not (processed_dir / str(name)).is_file():
                return False
    return True


def _load_raw_state_and_params(file_path: Path) -> Tuple[np.ndarray, np.ndarray, float, List[str], List[str]]:
    """Load one raw simulation file and extract state, params, and metadata."""
    with np.load(file_path, allow_pickle=True) as z:
        state = np.asarray(z["state_final"], dtype=np.float32)
        params = np.asarray(z["params"], dtype=np.float64)
        time_days = float(np.asarray(z["time_days"]).item())
        fields = [str(x) for x in np.asarray(z["fields"], dtype=object).tolist()]
        param_names = [str(x) for x in np.asarray(z["param_names"], dtype=object).tolist()]
    return state, params, time_days, fields, param_names


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
        st, p, _t, fields, pnames = _load_raw_state_and_params(fp)
        if fields0 is None:
            fields0 = fields
        elif fields != fields0:
            raise ValueError(f"Field mismatch in {fp}: {fields} != {fields0}")

        if pnames0 is None:
            pnames0 = pnames
        elif pnames != pnames0:
            raise ValueError(f"Param-name mismatch in {fp}: {pnames} != {pnames0}")

        st_tr = apply_state_transforms(st[None, ...], fields0, state_norm_cfg).astype(np.float64, copy=False)
        s = st_tr.sum(axis=(0, 2, 3))
        s2 = (st_tr * st_tr).sum(axis=(0, 2, 3))
        n = int(st_tr.shape[2] * st_tr.shape[3])

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


def _write_processed_file(
    *,
    src_file: Path,
    dst_file: Path,
    stats: NormalizationStats,
) -> str:
    """Normalize one raw sample and write one processed ``.npz`` file."""
    st, p, time_days, _fields, _pnames = _load_raw_state_and_params(src_file)

    st_norm = normalize_states(st[None, ...], stats)[0]
    p_norm = normalize_params(p[None, ...], stats)[0]

    np.savez_compressed(
        dst_file,
        state_final_norm=st_norm.astype(np.float32),
        params_norm=p_norm.astype(np.float32),
        time_days=np.asarray(time_days, dtype=np.float64),
    )
    return dst_file.name


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

    geometry_meta = _validate_raw_geometry(files, cfg=cfg)
    train_files, val_files = _split_files(files, seed=cfg.training.split_seed, val_fraction=cfg.training.val_fraction)

    stats = _fit_stats_streaming(
        train_files=train_files,
        state_norm_cfg=cfg.normalization.state,
        param_norm_cfg=cfg.normalization.params,
    )
    fields = list(stats.field_names)
    param_names = list(stats.param_names)
    const_param_names = [
        str(param_names[i]) for i, is_const in enumerate(np.asarray(stats.param_is_constant, dtype=bool)) if is_const
    ]
    varying_param_count = len(param_names) - len(const_param_names)
    if varying_param_count <= 0:
        raise ValueError(
            "All parameters are constant in this dataset split. "
            "At least one varying parameter is required for conditional emulation."
        )
    if const_param_names:
        LOGGER.warning(
            "Constant parameters detected and zeroed in normalization (%d/%d): %s",
            len(const_param_names),
            len(param_names),
            const_param_names,
        )

    written_train: List[str] = []
    written_val: List[str] = []

    for fp in train_files:
        dst = processed_dir / f"{fp.stem}_train.npz"
        written_train.append(_write_processed_file(src_file=fp, dst_file=dst, stats=stats))
    for fp in val_files:
        dst = processed_dir / f"{fp.stem}_val.npz"
        written_val.append(_write_processed_file(src_file=fp, dst_file=dst, stats=stats))

    with np.load(processed_dir / written_train[0], allow_pickle=True) as z:
        st0 = np.asarray(z["state_final_norm"], dtype=np.float32)

    meta: Dict[str, Any] = {
        "fields": fields,
        "param_names": param_names,
        "shape": {"C": int(st0.shape[0]), "H": int(st0.shape[1]), "W": int(st0.shape[2])},
        "splits": {"train": written_train, "val": written_val},
        "normalization": stats_to_json(stats),
        "constant_param_names": const_param_names,
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
            "rollout_steps_at_default_time": int(cfg.model.rollout_steps_at_default_time),
        },
        "build_fingerprint": fingerprint,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TerminalProcessedDataset(Dataset):
    """PyTorch dataset wrapper over processed terminal-state files."""

    def __init__(self, *, processed_dir: Path, file_names: Sequence[str]) -> None:
        self.processed_dir = processed_dir
        self.file_names = list(file_names)

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        fp = self.processed_dir / self.file_names[idx]
        with np.load(fp, allow_pickle=True) as z:
            state = np.asarray(z["state_final_norm"], dtype=np.float32)
            params = np.asarray(z["params_norm"], dtype=np.float32)
            time_days = float(np.asarray(z["time_days"]).item())
        return torch.from_numpy(params), torch.from_numpy(state), torch.tensor(time_days, dtype=torch.float32)


def _batch_steps(time_days_tensor: torch.Tensor, *, default_time_days: float, rollout_steps_at_default_time: int) -> int:
    """Convert batch ``time_days`` tensor into one shared integer rollout count."""
    flat = time_days_tensor.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("time_days batch is empty")

    first = flat[0]
    if not torch.allclose(flat, first.expand_as(flat), rtol=0.0, atol=1.0e-8):
        uniq = sorted({float(x.item()) for x in flat})
        raise ValueError(
            f"Mixed time_days values in a single batch: {uniq}. "
            "All simulations must use the same time_days."
        )

    return time_days_to_rollout_steps(
        float(first.item()),
        default_time_days=default_time_days,
        rollout_steps_at_default_time=rollout_steps_at_default_time,
    )


def _collect_validation_predictions(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    default_time_days: float,
    rollout_steps_at_default_time: int,
    amp_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on validation loader and collect predictions/targets as NumPy."""
    preds: List[np.ndarray] = []
    tars: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for pb, yb, tb in loader:
            pb = pb.to(device=device)
            yb = yb.to(device=device)
            steps = _batch_steps(
                tb,
                default_time_days=default_time_days,
                rollout_steps_at_default_time=rollout_steps_at_default_time,
            )
            with autocast_context(device, amp_mode):
                yhat = model(pb, steps=steps)
            preds.append(yhat.detach().cpu().numpy())
            tars.append(yb.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(tars, axis=0)


def _compute_gate_metrics(
    *,
    pred_norm: np.ndarray,
    tar_norm: np.ndarray,
    field_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute global and per-channel RMSE in normalized space."""
    diff = pred_norm - tar_norm
    global_rmse = float(np.sqrt(np.mean(diff**2)))
    per_channel = np.sqrt(np.mean(diff**2, axis=(0, 2, 3)))
    per_channel_named = {str(field_names[i]): float(per_channel[i]) for i in range(len(field_names))}
    return {
        "global_rmse": global_rmse,
        "per_channel_rmse": per_channel_named,
    }


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
    raw_files = sorted(dataset_dir.glob("sim_*.npz"))
    if not raw_files:
        return

    n_found = len(raw_files)
    n_expected = int(cfg.sampling.n_sims)
    if n_found < n_expected:
        LOGGER.warning(
            "Raw dataset has %d files but config.sampling.n_sims=%d. "
            "Training will proceed on existing files only.",
            n_found,
            n_expected,
        )


def train_emulator(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Train rollout emulator end-to-end and persist checkpoints/metrics."""
    ensure_torch_harmonics_importable(config_path.parent)
    _warn_if_raw_dataset_count_mismatch(cfg, config_path=config_path)

    processed_meta = preprocess_dataset(cfg, config_path=config_path)
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

    train_ds = TerminalProcessedDataset(processed_dir=processed_dir, file_names=processed_meta["splits"]["train"])
    val_ds = TerminalProcessedDataset(processed_dir=processed_dir, file_names=processed_meta["splits"]["val"])

    if len(train_ds) < int(cfg.training.batch_size):
        raise ValueError(
            "Training split size is smaller than batch_size while training loader uses drop_last=True: "
            f"n_train={len(train_ds)}, batch_size={cfg.training.batch_size}"
        )

    device = choose_device(cfg.training.device)

    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    loader_common: Dict[str, Any] = {
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

    use_non_blocking = bool(cfg.training.pin_memory) and device.type == "cuda"

    sample_params, sample_state, _sample_time = train_ds[0]
    state_chans = int(sample_state.shape[0])
    h = int(sample_state.shape[1])
    w = int(sample_state.shape[2])
    param_dim = int(sample_params.shape[0])

    model = build_rollout_model(
        img_size=(h, w),
        state_chans=state_chans,
        param_dim=param_dim,
        cfg_model=cfg.model,
        lat_order=lat_order,
        lon_origin=lon_origin,
    ).to(device)

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
    if cfg.training.amp_mode == "fp16" and device.type == "cuda":
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

        for pb, yb, tb in train_loader:
            pb = pb.to(device=device, non_blocking=use_non_blocking)
            yb = yb.to(device=device, non_blocking=use_non_blocking)
            do_finite_check = _should_run_finite_check(train_count)
            if do_finite_check:
                _check_finite_tensor(pb, name="train params batch")
                _check_finite_tensor(yb, name="train target batch")
            steps = _batch_steps(
                tb,
                default_time_days=cfg.solver.default_time_days,
                rollout_steps_at_default_time=cfg.model.rollout_steps_at_default_time,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.training.amp_mode):
                yhat = model(pb, steps=steps)
                if do_finite_check:
                    _check_finite_tensor(yhat, name="train prediction batch")
                loss = loss_fn(yhat, yb)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite")

            if epoch == 1 and train_count == 0:
                param_abs_max = _abs_max_tensor(pb)
                target_abs_max = _abs_max_tensor(yb)
                pred_abs_max = _abs_max_tensor(yhat)
                LOGGER.info(
                    "Epoch 1 batch 1 scales | params_abs_max=%.3e | target_abs_max=%.3e | pred_abs_max=%.3e",
                    param_abs_max,
                    target_abs_max,
                    pred_abs_max,
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
            for pb, yb, tb in val_loader:
                pb = pb.to(device=device, non_blocking=use_non_blocking)
                yb = yb.to(device=device, non_blocking=use_non_blocking)
                do_finite_check = _should_run_finite_check(val_count)
                steps = _batch_steps(
                    tb,
                    default_time_days=cfg.solver.default_time_days,
                    rollout_steps_at_default_time=cfg.model.rollout_steps_at_default_time,
                )
                with autocast_context(device, cfg.training.amp_mode):
                    yhat = model(pb, steps=steps)
                    if do_finite_check:
                        _check_finite_tensor(yhat, name="val prediction batch")
                    vloss = loss_fn(yhat, yb)
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
            "Epoch %d/%d | train=%.6e | val=%.6e | lr=%.3e",
            epoch,
            epochs,
            train_loss,
            val_loss,
            lr_now,
        )

        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})

        ckpt = {
            "mode": "param_rollout_terminal",
            "model_state": model.state_dict(),
            "fields": field_names,
            "param_names": list(processed_meta["param_names"]),
            "shape": {"C": state_chans, "H": h, "W": w},
            "geometry": {
                "lat_order": lat_order,
                "lon_origin": lon_origin,
            },
            "normalization": stats_to_json(stats),
            "solver": asdict(cfg.solver),
            "model_config": asdict(cfg.model),
            "training_config": asdict(cfg.training),
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

    # Validation metrics on best checkpoint
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    pred_norm, tar_norm = _collect_validation_predictions(
        model=model,
        loader=val_loader,
        device=device,
        default_time_days=cfg.solver.default_time_days,
        rollout_steps_at_default_time=cfg.model.rollout_steps_at_default_time,
        amp_mode=cfg.training.amp_mode,
    )

    val_metrics = _compute_gate_metrics(pred_norm=pred_norm, tar_norm=tar_norm, field_names=field_names)

    (model_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (model_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_val_loss": float(best_val),
        "history_path": str(model_dir / "training_history.json"),
        "val_metrics": val_metrics,
        "processed_meta": str(processed_dir / "processed_meta.json"),
        "resolved_config_path": str(resolved_cfg_path),
        "original_config_copy_path": str(original_cfg_path),
    }
    return summary
