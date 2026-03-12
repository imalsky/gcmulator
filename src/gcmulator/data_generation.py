"""Raw dataset generation for checkpoint-sequence direct-jump training."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import GCMulatorConfig, PHYSICAL_STATE_FIELDS, resolve_path
from .geometry import apply_geometry_state
from .my_swamp_backend import (
    conditioning_param_names,
    detect_jax_backend,
    ensure_my_swamp_importable,
    params_to_conditioning_vector,
    params_to_public_json_dict,
    run_trajectory_checkpoints,
    run_trajectory_checkpoints_batched,
)
from .sampling import (
    build_uniform_checkpoint_schedule,
    sample_parameter_dict,
    to_extended9,
)


LOGGER = logging.getLogger("generate")
GPU_BACKENDS = {"gpu", "cuda", "rocm", "metal"}


def _list_existing_raw_files(dataset_dir: Path) -> List[Path]:
    """Return generated raw simulation files written by the active contract."""
    return sorted(dataset_dir.glob("sim_*.npy"))


def _list_legacy_raw_files(dataset_dir: Path) -> List[Path]:
    """Return unsupported legacy raw files that should fail fast."""
    return sorted(dataset_dir.glob("sim_*.npz"))


def _clear_dataset_dir(dataset_dir: Path) -> None:
    """Remove existing generated simulation files and manifest."""
    legacy = _list_legacy_raw_files(dataset_dir)
    if legacy:
        raise RuntimeError(
            "Unsupported legacy raw files were found in the dataset directory. "
            "Remove "
            f"{len(legacy)} sim_*.npz files before regenerating the dataset: "
            f"{dataset_dir}"
        )
    for path in _list_existing_raw_files(dataset_dir):
        path.unlink()
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def _write_sim_record(
    *,
    sim_idx: int,
    checkpoint_states: np.ndarray,
    checkpoint_steps: np.ndarray,
    checkpoint_days: np.ndarray,
    params_vector: np.ndarray,
    params_json: Dict[str, float],
    cfg: GCMulatorConfig,
    dataset_dir: Path,
    resolved_checkpoint_interval_days: float,
) -> Dict[str, Any]:
    """Write one checkpoint-sequence simulation into a raw ``sim_XXXXXX.npy`` record."""
    if checkpoint_states.ndim != 4:
        raise ValueError("checkpoint_states must be [S,C,H,W]")
    if checkpoint_states.shape[1] != len(PHYSICAL_STATE_FIELDS):
        raise ValueError("checkpoint_states channel count mismatch")
    if checkpoint_steps.ndim != 1 or checkpoint_days.ndim != 1:
        raise ValueError("checkpoint_steps/checkpoint_days must be rank-1")
    if checkpoint_steps.shape != checkpoint_days.shape:
        raise ValueError("checkpoint_steps and checkpoint_days must align")
    if int(checkpoint_states.shape[0]) != int(checkpoint_steps.shape[0]):
        raise ValueError("checkpoint_states and checkpoint_steps must align")
    if int(checkpoint_states.shape[0]) < 2:
        raise RuntimeError("At least two checkpoints are required per trajectory")

    states_geom, geometry_info = apply_geometry_state(
        checkpoint_states.astype(np.float64, copy=True),
        flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
        roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
    )

    raw_path = dataset_dir / f"sim_{sim_idx:06d}.npy"
    payload = {
        "checkpoint_states": states_geom,
        "checkpoint_steps": checkpoint_steps.astype(np.int64, copy=False),
        "checkpoint_days": checkpoint_days.astype(np.float64, copy=False),
        "state_fields": np.asarray(list(PHYSICAL_STATE_FIELDS), dtype=object),
        "params": params_vector.astype(np.float64, copy=False),
        "param_names": np.asarray(list(conditioning_param_names()), dtype=object),
        "default_time_days": np.asarray(
            float(cfg.solver.default_time_days),
            dtype=np.float64,
        ),
        "burn_in_days": np.asarray(float(cfg.sampling.burn_in_days), dtype=np.float64),
        "dt_seconds": np.asarray(float(cfg.solver.dt_seconds), dtype=np.float64),
        "starttime_index": np.asarray(int(cfg.solver.starttime_index), dtype=np.int64),
        "saved_checkpoint_interval_days": np.asarray(
            float(resolved_checkpoint_interval_days),
            dtype=np.float64,
        ),
        "n_saved_checkpoints": np.asarray(int(states_geom.shape[0]), dtype=np.int64),
        "M": np.asarray(int(cfg.solver.M), dtype=np.int64),
        "nlat": np.asarray(int(states_geom.shape[-2]), dtype=np.int64),
        "nlon": np.asarray(int(states_geom.shape[-1]), dtype=np.int64),
        "lat_order": np.asarray(str(geometry_info["lat_order"]), dtype=object),
        "lon_origin": np.asarray(str(geometry_info["lon_origin"]), dtype=object),
        "lon_shift": np.asarray(int(geometry_info["lon_shift"]), dtype=np.int64),
    }
    np.save(raw_path, payload, allow_pickle=True)

    return {
        "sim_idx": int(sim_idx),
        "file": raw_path.name,
        "state_fields": list(PHYSICAL_STATE_FIELDS),
        "param_names": list(conditioning_param_names()),
        "params": params_json,
        "default_time_days": float(cfg.solver.default_time_days),
        "burn_in_days": float(cfg.sampling.burn_in_days),
        "dt_seconds": float(cfg.solver.dt_seconds),
        "starttime_index": int(cfg.solver.starttime_index),
        "saved_checkpoint_interval_days": float(resolved_checkpoint_interval_days),
        "n_saved_checkpoints": int(states_geom.shape[0]),
        "checkpoint_day_start": float(checkpoint_days[0]),
        "checkpoint_day_end": float(checkpoint_days[-1]),
        "M": int(cfg.solver.M),
        "nlat": int(states_geom.shape[-2]),
        "nlon": int(states_geom.shape[-1]),
        "lat_order": str(geometry_info["lat_order"]),
        "lon_origin": str(geometry_info["lon_origin"]),
        "lon_shift": int(geometry_info["lon_shift"]),
    }


def _log_progress(*, completed: int, total: int, start_time: float) -> None:
    """Log elapsed runtime and ETA for dataset generation."""
    elapsed = time.time() - start_time
    average = elapsed / float(completed)
    remaining = average * float(total - completed)
    LOGGER.info(
        "Generated %4d/%4d trajectory sims | elapsed=%8.1fs | ETA=%8.1fs",
        completed,
        total,
        elapsed,
        remaining,
    )


def _resolve_generation_batch_size(
    *,
    requested_workers: int,
    n_sims: int,
    jax_backend: str,
) -> int:
    """Resolve the vectorized JAX trajectory batch size for generation."""
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    if requested_workers > 0:
        return min(int(requested_workers), int(n_sims))

    raw = os.environ.get("GCMULATOR_JAX_SIM_BATCH", "auto").strip().lower()
    if raw in {"", "auto"}:
        if jax_backend.lower() in GPU_BACKENDS:
            auto_gpu = int(os.environ.get("GCMULATOR_JAX_SIM_BATCH_AUTO_GPU", "8"))
            if auto_gpu < 1:
                raise ValueError(
                    "GCMULATOR_JAX_SIM_BATCH_AUTO_GPU must be >= 1, "
                    f"got {auto_gpu}"
                )
            batch_size = auto_gpu
        else:
            batch_size = 1
    else:
        try:
            batch_size = int(raw)
        except Exception as exc:
            raise ValueError(
                "GCMULATOR_JAX_SIM_BATCH must be 'auto' or an integer >= 1, "
                f"got {raw!r}"
            ) from exc
        if batch_size < 1:
            raise ValueError(f"GCMULATOR_JAX_SIM_BATCH must be >= 1, got {batch_size}")

    return min(int(batch_size), int(n_sims))


def generate_dataset(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Generate the raw checkpoint-sequence dataset using MY_SWAMP as the source of truth."""
    dataset_dir = resolve_path(config_path, cfg.paths.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    legacy = _list_legacy_raw_files(dataset_dir)
    if legacy:
        raise RuntimeError(
            "Unsupported legacy raw files were found in the dataset directory. "
            "Remove "
            f"{len(legacy)} sim_*.npz files before generating a new dataset: "
            f"{dataset_dir}"
        )
    if cfg.paths.overwrite_dataset:
        _clear_dataset_dir(dataset_dir)
    else:
        existing = _list_existing_raw_files(dataset_dir)
        if existing:
            raise FileExistsError(
                "Dataset directory already contains raw simulation files "
                f"({len(existing)} sim_*.npy files) and overwrite_dataset=false: "
                f"{dataset_dir}"
            )

    ensure_my_swamp_importable(config_path.parent)

    n_sims = int(cfg.sampling.n_sims)
    rng = np.random.default_rng(int(cfg.sampling.seed))
    checkpoint_schedule = build_uniform_checkpoint_schedule(
        time_days=float(cfg.solver.default_time_days),
        dt_seconds=float(cfg.solver.dt_seconds),
        saved_checkpoint_interval_days=float(cfg.sampling.saved_checkpoint_interval_days),
        saved_snapshots_per_sim=cfg.sampling.saved_snapshots_per_sim,
    )

    jax_backend = detect_jax_backend()
    generation_batch_size = _resolve_generation_batch_size(
        requested_workers=int(cfg.sampling.generation_workers),
        n_sims=n_sims,
        jax_backend=jax_backend,
    )
    LOGGER.info(
        "Dataset generation | backend=%-6s | n_sims=%d | generation_batch_size=%d "
        "| checkpoints=%d | interval_days=%.6f | state_fields=%s",
        jax_backend,
        n_sims,
        generation_batch_size,
        int(checkpoint_schedule.checkpoint_steps.shape[0]),
        float(checkpoint_schedule.interval_days),
        list(PHYSICAL_STATE_FIELDS),
    )
    if jax_backend.lower() in GPU_BACKENDS and generation_batch_size > 1:
        LOGGER.info(
            "GPU-vectorized trajectory generation is active with batch_size=%d.",
            generation_batch_size,
        )

    items: List[Dict[str, Any]] = []
    start_time = time.time()
    log_every = max(1, n_sims // 20)
    sampled_runs: List[Dict[str, Any]] = []
    for sim_idx in range(n_sims):
        params = to_extended9(sample_parameter_dict(rng, cfg.sampling.parameters))
        sampled_runs.append(
            {
                "sim_idx": int(sim_idx),
                "params": params,
            }
        )

    for batch_start in range(0, n_sims, generation_batch_size):
        batch = sampled_runs[batch_start:batch_start + generation_batch_size]
        checkpoint_steps_batch = np.repeat(
            checkpoint_schedule.checkpoint_steps[None, :],
            len(batch),
            axis=0,
        )
        if len(batch) == 1:
            entry = batch[0]
            checkpoint_states = run_trajectory_checkpoints(
                entry["params"],
                M=int(cfg.solver.M),
                dt_seconds=float(cfg.solver.dt_seconds),
                time_days=float(cfg.solver.default_time_days),
                starttime_index=int(cfg.solver.starttime_index),
                checkpoint_steps=checkpoint_schedule.checkpoint_steps,
            )
            batch_checkpoint_states = np.asarray(checkpoint_states, dtype=np.float64)[None, ...]
        else:
            k6_values = {float(entry["params"].K6) for entry in batch}
            k6phi_values = {entry["params"].K6Phi for entry in batch}
            if len(k6_values) != 1 or len(k6phi_values) != 1:
                raise ValueError(
                    "Batched trajectory generation requires shared internal diffusion controls"
                )
            params_batch = np.stack(
                [
                    params_to_conditioning_vector(entry["params"])
                    for entry in batch
                ],
                axis=0,
            )
            batch_checkpoint_states = run_trajectory_checkpoints_batched(
                params_batch,
                M=int(cfg.solver.M),
                dt_seconds=float(cfg.solver.dt_seconds),
                time_days=float(cfg.solver.default_time_days),
                starttime_index=int(cfg.solver.starttime_index),
                checkpoint_steps_batch=checkpoint_steps_batch,
                k6=float(next(iter(k6_values))),
                k6phi=next(iter(k6phi_values)),
            )

        for batch_index, entry in enumerate(batch):
            item = _write_sim_record(
                sim_idx=int(entry["sim_idx"]),
                checkpoint_states=batch_checkpoint_states[batch_index],
                checkpoint_steps=checkpoint_schedule.checkpoint_steps,
                checkpoint_days=checkpoint_schedule.checkpoint_days,
                params_vector=params_to_conditioning_vector(entry["params"]),
                params_json=params_to_public_json_dict(entry["params"]),
                cfg=cfg,
                dataset_dir=dataset_dir,
                resolved_checkpoint_interval_days=float(checkpoint_schedule.interval_days),
            )
            items.append(item)

        completed = min(batch_start + len(batch), n_sims)
        if (completed % log_every) == 0 or completed == n_sims:
            _log_progress(completed=completed, total=n_sims, start_time=start_time)

    items.sort(key=lambda item: int(item["sim_idx"]))
    manifest = {
        "created_unix": time.time(),
        "n_sims_requested": n_sims,
        "n_sims_written": len(items),
        "state_fields": list(PHYSICAL_STATE_FIELDS),
        "param_names": list(conditioning_param_names()),
        "dataset_dir": str(dataset_dir),
        "solver": {
            "M": int(cfg.solver.M),
            "dt_seconds": float(cfg.solver.dt_seconds),
            "default_time_days": float(cfg.solver.default_time_days),
            "starttime_index": int(cfg.solver.starttime_index),
        },
        "sampling": {
            "seed": int(cfg.sampling.seed),
            "n_sims": int(cfg.sampling.n_sims),
            "generation_workers": int(cfg.sampling.generation_workers),
            "resolved_generation_batch_size": int(generation_batch_size),
            "burn_in_days": float(cfg.sampling.burn_in_days),
            "saved_checkpoint_interval_days": float(checkpoint_schedule.interval_days),
            "saved_snapshots_per_sim": (
                None
                if cfg.sampling.saved_snapshots_per_sim is None
                else int(cfg.sampling.saved_snapshots_per_sim)
            ),
            "live_pairs_per_sequence": int(cfg.sampling.live_pairs_per_sequence),
            "pairs_per_sim": int(cfg.sampling.pairs_per_sim),
            "pair_sampling_policy": str(cfg.sampling.pair_sampling_policy),
            "live_transition_days_min": float(cfg.sampling.live_transition_days_min),
            "live_transition_days_max": float(cfg.sampling.live_transition_days_max),
            "live_transition_tolerance_fraction": float(
                cfg.sampling.live_transition_tolerance_fraction
            ),
            "uses_variable_live_transition": bool(cfg.sampling.uses_variable_live_transition()),
        },
        "geometry": {
            "flip_latitude_to_north_south": bool(cfg.geometry.flip_latitude_to_north_south),
            "roll_longitude_to_0_2pi": bool(cfg.geometry.roll_longitude_to_0_2pi),
        },
        "n_saved_checkpoints": int(checkpoint_schedule.checkpoint_steps.shape[0]),
        "checkpoint_days": checkpoint_schedule.checkpoint_days.tolist(),
        "items": items,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
