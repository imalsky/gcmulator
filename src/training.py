from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import GCMulatorConfig, resolve_path, time_days_to_rollout_steps
from .constants import STD_FLOOR
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


def _list_raw_dataset_files(dataset_dir: Path) -> List[Path]:
    files = sorted(dataset_dir.glob("sim_*.npz"))
    if not files:
        raise FileNotFoundError(f"No sim_*.npz files found in {dataset_dir}")
    return files


def _split_files(files: Sequence[Path], *, seed: int, val_fraction: float) -> Tuple[List[Path], List[Path]]:
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


def _load_raw_state_and_params(file_path: Path) -> Tuple[np.ndarray, np.ndarray, float, List[str], List[str]]:
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
    fields0: List[str] | None = None
    pnames0: List[str] | None = None

    state_sum: np.ndarray | None = None
    state_sum2: np.ndarray | None = None
    state_count = 0

    param_sum: np.ndarray | None = None
    param_sum2: np.ndarray | None = None
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
        if param_sum is None:
            param_sum = p64.copy()
            param_sum2 = p64 * p64
        else:
            param_sum += p64
            param_sum2 += p64 * p64
        param_count += 1

    if fields0 is None or pnames0 is None:
        raise RuntimeError("Could not infer field/parameter names from training files")
    if state_sum is None or state_sum2 is None:
        raise RuntimeError("Failed to accumulate state normalization moments")
    if param_sum is None or param_sum2 is None:
        raise RuntimeError("Failed to accumulate parameter normalization moments")

    state_mean = state_sum / float(state_count)
    state_var = np.maximum(state_sum2 / float(state_count) - state_mean * state_mean, 0.0)
    state_std = np.maximum(np.sqrt(state_var), STD_FLOOR)

    if param_norm_cfg.mode == "zscore":
        param_mean = param_sum / float(param_count)
        param_var = np.maximum(param_sum2 / float(param_count) - param_mean * param_mean, 0.0)
        param_std = np.maximum(np.sqrt(param_var), STD_FLOOR)
    elif param_norm_cfg.mode == "none":
        param_mean = np.zeros_like(param_sum)
        param_std = np.ones_like(param_sum)
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


def _validate_existing_processed_meta(meta: Dict[str, Any], *, processed_dir: Path) -> None:
    required_top = {"fields", "param_names", "shape", "splits", "normalization"}
    missing_top = sorted(required_top.difference(meta))
    if missing_top:
        raise KeyError(f"processed_meta.json missing required keys: {missing_top}")

    splits_obj = meta["splits"]
    if not isinstance(splits_obj, dict):
        raise ValueError("processed_meta.json 'splits' must be an object")

    for split_name in ("train", "val"):
        split_files = splits_obj.get(split_name)
        if not isinstance(split_files, list) or not split_files:
            raise ValueError(f"processed_meta.json splits.{split_name} must be a non-empty list")
        missing_files = [str(name) for name in split_files if not (processed_dir / str(name)).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"processed_meta.json splits.{split_name} references missing files in {processed_dir}: "
                f"{missing_files[:5]}{'...' if len(missing_files) > 5 else ''}"
            )


def preprocess_dataset(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    dataset_dir = resolve_path(config_path, cfg.paths.dataset_dir)
    processed_dir = resolve_path(config_path, cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    meta_path = processed_dir / "processed_meta.json"
    if meta_path.exists() and not cfg.paths.overwrite_processed:
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(cached_meta, dict):
            raise ValueError(f"processed_meta.json must contain an object: {meta_path}")
        _validate_existing_processed_meta(cached_meta, processed_dir=processed_dir)
        return cached_meta

    if cfg.paths.overwrite_processed and processed_dir.exists():
        for p in processed_dir.glob("*"):
            if p.is_file():
                p.unlink()

    files = _list_raw_dataset_files(dataset_dir)
    train_files, val_files = _split_files(files, seed=cfg.training.split_seed, val_fraction=cfg.training.val_fraction)

    stats = _fit_stats_streaming(
        train_files=train_files,
        state_norm_cfg=cfg.normalization.state,
        param_norm_cfg=cfg.normalization.params,
    )
    fields = list(stats.field_names)
    param_names = list(stats.param_names)

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
        "solver": {
            "M": int(cfg.solver.M),
            "dt_seconds": float(cfg.solver.dt_seconds),
            "default_time_days": float(cfg.solver.default_time_days),
        },
        "model_time_mapping": {
            "default_time_days": float(cfg.solver.default_time_days),
            "rollout_steps_at_default_time": int(cfg.model.rollout_steps_at_default_time),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TerminalProcessedDataset(Dataset):
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
    diff = pred_norm - tar_norm
    global_rmse = float(np.sqrt(np.mean(diff**2)))
    per_channel = np.sqrt(np.mean(diff**2, axis=(0, 2, 3)))
    per_channel_named = {str(field_names[i]): float(per_channel[i]) for i in range(len(field_names))}
    return {
        "global_rmse": global_rmse,
        "per_channel_rmse": per_channel_named,
    }


def train_emulator(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    ensure_torch_harmonics_importable(config_path.parent)

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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=False,
    )

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
    ).to(device)

    loss_fn = SphereLoss(nlat=h, nlon=w, grid=cfg.model.grid).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    scheduler = None
    if cfg.training.scheduler.type == "plateau":
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
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for pb, yb, tb in train_loader:
            pb = pb.to(device=device)
            yb = yb.to(device=device)
            steps = _batch_steps(
                tb,
                default_time_days=cfg.solver.default_time_days,
                rollout_steps_at_default_time=cfg.model.rollout_steps_at_default_time,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.training.amp_mode):
                yhat = model(pb, steps=steps)
                loss = loss_fn(yhat, yb)

            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.training.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
                optimizer.step()

            train_loss_sum += float(loss.detach().item())
            train_count += 1

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for pb, yb, tb in val_loader:
                pb = pb.to(device=device)
                yb = yb.to(device=device)
                steps = _batch_steps(
                    tb,
                    default_time_days=cfg.solver.default_time_days,
                    rollout_steps_at_default_time=cfg.model.rollout_steps_at_default_time,
                )
                with autocast_context(device, cfg.training.amp_mode):
                    yhat = model(pb, steps=steps)
                    vloss = loss_fn(yhat, yb)
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
