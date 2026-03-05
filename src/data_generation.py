"""Raw dataset generation pipeline using MY_SWAMP trajectory transitions."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import Extended9Params, GCMulatorConfig, resolve_path
from .geometry import apply_geometry_state
from .my_swamp_backend import (
    FIELDS_5,
    conditioning_param_names,
    detect_jax_backend,
    ensure_my_swamp_importable,
    params_to_conditioning_vector,
    params_to_public_json_dict,
    run_trajectory_transitions,
)
from .sampling import CONDITIONING_PARAM_NAMES, sample_parameter_dict, to_extended9

LOGGER = logging.getLogger("src.generate")
GPU_BACKENDS = {"gpu", "cuda", "rocm", "metal"}
AUTO_GPU_SIM_BATCH_DEFAULT = 8


def _clear_dataset_dir(dataset_dir: Path) -> None:
    """Remove existing generated simulation files and manifest."""
    for p in dataset_dir.glob("sim_*.npy"):
        p.unlink()
    for p in dataset_dir.glob("sim_*.npz"):
        p.unlink()
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def _resolve_generation_workers(*, requested_workers: int, n_sims: int, jax_backend: str) -> int:
    """Choose safe worker count from config, dataset size, and backend type."""
    if n_sims <= 1:
        return 1
    if requested_workers > 0:
        return min(int(requested_workers), int(n_sims))
    if jax_backend.lower() in GPU_BACKENDS:
        # For one GPU, single simulation worker is usually fastest and most stable.
        return 1
    return 1


def _resolve_sim_batch_size(
    *,
    n_sims: int,
    jax_backend: str,
    generation_workers: int,
) -> int:
    """Resolve per-call JAX batch size for independent trajectory simulations.

    Environment knob:
    - ``GCMULATOR_JAX_SIM_BATCH=auto`` (default): use a conservative GPU batch.
    - ``GCMULATOR_JAX_SIM_BATCH_AUTO_GPU=<int>=1``: auto-mode GPU batch size
      (default: 8).
    - ``GCMULATOR_JAX_SIM_BATCH=<int>=1``: force explicit batch size.
    """
    raw = os.environ.get("GCMULATOR_JAX_SIM_BATCH", "auto").strip().lower()
    if raw in {"", "auto"}:
        if jax_backend.lower() in GPU_BACKENDS and int(generation_workers) == 1:
            auto_gpu_raw = os.environ.get("GCMULATOR_JAX_SIM_BATCH_AUTO_GPU", str(AUTO_GPU_SIM_BATCH_DEFAULT)).strip()
            try:
                batch = int(auto_gpu_raw)
            except Exception as exc:
                raise ValueError(
                    "GCMULATOR_JAX_SIM_BATCH_AUTO_GPU must be an integer >= 1, "
                    f"got {auto_gpu_raw!r}"
                ) from exc
            if batch < 1:
                raise ValueError(f"GCMULATOR_JAX_SIM_BATCH_AUTO_GPU must be >= 1, got {batch}")
        else:
            batch = 1
    else:
        try:
            batch = int(raw)
        except Exception as exc:
            raise ValueError(
                "GCMULATOR_JAX_SIM_BATCH must be 'auto' or an integer >= 1, "
                f"got {raw!r}"
            ) from exc
        if batch < 1:
            raise ValueError(f"GCMULATOR_JAX_SIM_BATCH must be >= 1, got {batch}")

    if int(generation_workers) != 1 and batch > 1:
        LOGGER.warning(
            "Requested GCMULATOR_JAX_SIM_BATCH=%d but generation_workers=%d. "
            "Trajectory-transition generation currently uses scalar solves; falling back to 1.",
            batch,
            int(generation_workers),
        )
        batch = 1
    return min(batch, int(n_sims))


def _write_sim_record(
    *,
    sim_idx: int,
    params: Extended9Params,
    state_inputs: np.ndarray,
    state_targets: np.ndarray,
    transition_days: np.ndarray,
    cfg: GCMulatorConfig,
    dataset_dir: Path,
    param_names: List[str],
) -> Dict[str, Any]:
    """Write one transition trajectory into a ``sim_XXXXXX.npy`` file."""
    if state_inputs.ndim != 4 or state_targets.ndim != 4:
        raise ValueError(
            "state_inputs/state_targets must be [T,C,H,W], got "
            f"{tuple(state_inputs.shape)} and {tuple(state_targets.shape)}"
        )
    if state_inputs.shape != state_targets.shape:
        raise ValueError(f"state_inputs/state_targets shape mismatch: {state_inputs.shape} vs {state_targets.shape}")
    if transition_days.ndim != 1 or int(transition_days.shape[0]) != int(state_inputs.shape[0]):
        raise ValueError(
            "transition_days must be [T] aligned with trajectory transitions, got "
            f"{tuple(transition_days.shape)} for T={state_inputs.shape[0]}"
        )

    t_len = int(state_inputs.shape[0])
    state_inputs_geom = np.empty_like(state_inputs, dtype=np.float32)
    state_targets_geom = np.empty_like(state_targets, dtype=np.float32)
    geom_info: Dict[str, Any] | None = None
    for ti in range(t_len):
        x_in, g_in = apply_geometry_state(
            state_inputs[ti].astype(np.float32, copy=False),
            flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
            roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
        )
        x_out, g_out = apply_geometry_state(
            state_targets[ti].astype(np.float32, copy=False),
            flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
            roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
        )
        if (g_in["lat_order"], g_in["lon_origin"], int(g_in["lon_shift"])) != (
            g_out["lat_order"],
            g_out["lon_origin"],
            int(g_out["lon_shift"]),
        ):
            raise RuntimeError("Geometry conversion mismatch between input and target states.")
        if geom_info is None:
            geom_info = g_in
        else:
            if (geom_info["lat_order"], geom_info["lon_origin"], int(geom_info["lon_shift"])) != (
                g_in["lat_order"],
                g_in["lon_origin"],
                int(g_in["lon_shift"]),
            ):
                raise RuntimeError("Inconsistent geometry metadata across trajectory transitions.")
        state_inputs_geom[ti] = x_in
        state_targets_geom[ti] = x_out
    if geom_info is None:
        raise RuntimeError("Failed to infer geometry metadata for trajectory record.")

    sim_file = dataset_dir / f"sim_{sim_idx:06d}.npy"
    payload = {
        "state_inputs": state_inputs_geom,
        "state_targets": state_targets_geom,
        "transition_days": transition_days.astype(np.float64, copy=False),
        "fields": np.asarray(FIELDS_5, dtype=object),
        "params": params_to_conditioning_vector(params),
        "param_names": np.asarray(param_names, dtype=object),
        "time_days": np.asarray(float(cfg.solver.default_time_days), dtype=np.float64),
        "dt_seconds": np.asarray(float(cfg.solver.dt_seconds), dtype=np.float64),
        "transition_jump_steps": np.asarray(int(cfg.model.transition_jump_steps), dtype=np.int64),
        "n_transitions": np.asarray(int(t_len), dtype=np.int64),
        "M": np.asarray(int(cfg.solver.M), dtype=np.int64),
        "nlat": np.asarray(int(state_inputs_geom.shape[-2]), dtype=np.int64),
        "nlon": np.asarray(int(state_inputs_geom.shape[-1]), dtype=np.int64),
        "lat_order": np.asarray(geom_info["lat_order"], dtype=object),
        "lon_origin": np.asarray(geom_info["lon_origin"], dtype=object),
        "lon_shift": np.asarray(int(geom_info["lon_shift"]), dtype=np.int64),
    }
    np.save(sim_file, payload, allow_pickle=True)

    return {
        "sim_idx": int(sim_idx),
        "file": sim_file.name,
        "fields": list(FIELDS_5),
        "param_names": list(param_names),
        "params": params_to_public_json_dict(params),
        "time_days": float(cfg.solver.default_time_days),
        "transition_jump_steps": int(cfg.model.transition_jump_steps),
        "n_transitions": int(t_len),
        "transition_days_min": float(np.min(transition_days)),
        "transition_days_max": float(np.max(transition_days)),
        "dt_seconds": float(cfg.solver.dt_seconds),
        "M": int(cfg.solver.M),
        "nlat": int(state_inputs_geom.shape[-2]),
        "nlon": int(state_inputs_geom.shape[-1]),
        "lat_order": str(geom_info["lat_order"]),
        "lon_origin": str(geom_info["lon_origin"]),
        "lon_shift": int(geom_info["lon_shift"]),
    }


def _write_one_sim(
    *,
    sim_idx: int,
    params: Extended9Params,
    cfg: GCMulatorConfig,
    dataset_dir: Path,
    param_names: List[str],
) -> Dict[str, Any]:
    """Run one simulation and write trajectory transitions to one file."""
    state_inputs, state_targets, transition_days = run_trajectory_transitions(
        params,
        M=cfg.solver.M,
        dt_seconds=cfg.solver.dt_seconds,
        time_days=cfg.solver.default_time_days,
        starttime_index=cfg.solver.starttime_index,
        n_transitions=cfg.model.rollout_steps_at_default_time,
        transition_jump_steps=cfg.model.transition_jump_steps,
    )
    return _write_sim_record(
        sim_idx=sim_idx,
        params=params,
        state_inputs=state_inputs,
        state_targets=state_targets,
        transition_days=transition_days,
        cfg=cfg,
        dataset_dir=dataset_dir,
        param_names=param_names,
    )


def _log_progress(*, completed: int, total: int, start_t: float) -> None:
    """Log elapsed runtime and ETA for dataset generation."""
    elapsed = time.time() - start_t
    avg = elapsed / float(completed)
    remain = avg * float(total - completed)
    LOGGER.info(
        "Generated %4d/%4d trajectory sims | elapsed=%8.1fs | ETA=%8.1fs",
        completed,
        total,
        elapsed,
        remain,
    )


def generate_dataset(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Generate trajectory-transition dataset using MY_SWAMP truth."""
    ensure_my_swamp_importable(config_path.parent)

    dataset_dir = resolve_path(config_path, cfg.paths.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if cfg.paths.overwrite_dataset:
        _clear_dataset_dir(dataset_dir)
    else:
        existing = sorted(list(dataset_dir.glob("sim_*.npy")) + list(dataset_dir.glob("sim_*.npz")))
        if existing:
            raise FileExistsError(
                f"Dataset directory already contains {len(existing)} simulation files and overwrite_dataset=false: {dataset_dir}"
            )

    n_sims = int(cfg.sampling.n_sims)
    rng = np.random.default_rng(cfg.sampling.seed)
    param_names = list(conditioning_param_names())
    sampled_params = [to_extended9(sample_parameter_dict(rng, cfg.sampling.parameters)) for _ in range(n_sims)]

    jax_backend = detect_jax_backend()
    generation_workers = _resolve_generation_workers(
        requested_workers=int(cfg.sampling.generation_workers),
        n_sims=n_sims,
        jax_backend=jax_backend,
    )
    sim_batch_size = _resolve_sim_batch_size(
        n_sims=n_sims,
        jax_backend=jax_backend,
        generation_workers=generation_workers,
    )
    LOGGER.info(
        "Dataset generation | backend=%-6s | workers=%2d (requested=%2d) | sim_batch=%2d | raw_format=%-3s",
        jax_backend,
        generation_workers,
        int(cfg.sampling.generation_workers),
        sim_batch_size,
        "npy",
    )
    if sim_batch_size > 1:
        LOGGER.warning(
            "Trajectory-transition generation currently runs scalar simulations; ignoring sim_batch=%d.",
            sim_batch_size,
        )
        sim_batch_size = 1
    if jax_backend.lower() in GPU_BACKENDS and generation_workers > 1:
        LOGGER.warning(
            "generation_workers=%d on a single GPU may reduce throughput due to contention. "
            "Consider generation_workers=1.",
            generation_workers,
        )
    require_jax_gpu = os.environ.get("GCMULATOR_REQUIRE_JAX_GPU", "0") == "1"
    if require_jax_gpu and jax_backend.lower() not in GPU_BACKENDS:
        raise RuntimeError(
            "GCMULATOR_REQUIRE_JAX_GPU=1 but JAX backend is "
            f"'{jax_backend}'. Refusing CPU fallback for data generation."
        )

    written_by_idx: Dict[int, Dict[str, Any]] = {}
    start_t = time.time()
    log_every = max(1, n_sims // 20)  # log ~20 times during the run
    completed = 0

    if generation_workers == 1:
        for sim_idx, params in enumerate(sampled_params):
            written_by_idx[sim_idx] = _write_one_sim(
                sim_idx=sim_idx,
                params=params,
                cfg=cfg,
                dataset_dir=dataset_dir,
                param_names=param_names,
            )
            completed += 1
            if (completed % log_every) == 0 or completed == n_sims:
                _log_progress(completed=completed, total=n_sims, start_t=start_t)
    else:
        # Warm-up one simulation before threading so JAX can compile once first.
        written_by_idx[0] = _write_one_sim(
            sim_idx=0,
            params=sampled_params[0],
            cfg=cfg,
            dataset_dir=dataset_dir,
            param_names=param_names,
        )
        completed = 1
        if (completed % log_every) == 0 or completed == n_sims:
            _log_progress(completed=completed, total=n_sims, start_t=start_t)

        with ThreadPoolExecutor(max_workers=generation_workers) as pool:
            futures: List[Future[Dict[str, Any]]] = []
            for sim_idx in range(1, n_sims):
                futures.append(
                    pool.submit(
                        _write_one_sim,
                        sim_idx=sim_idx,
                        params=sampled_params[sim_idx],
                        cfg=cfg,
                        dataset_dir=dataset_dir,
                        param_names=param_names,
                    )
                )

            for future in as_completed(futures):
                item = future.result()
                written_by_idx[int(item["sim_idx"])] = item
                completed += 1
                if (completed % log_every) == 0 or completed == n_sims:
                    _log_progress(completed=completed, total=n_sims, start_t=start_t)

    written = [written_by_idx[i] for i in range(n_sims)]

    manifest = {
        "created_unix": time.time(),
        "n_sims_requested": int(n_sims),
        "n_sims_written": int(len(written)),
        "fields": list(FIELDS_5),
        "param_names": list(param_names),
        "dataset_dir": str(dataset_dir),
        "solver": {
            "M": int(cfg.solver.M),
            "dt_seconds": float(cfg.solver.dt_seconds),
            "default_time_days": float(cfg.solver.default_time_days),
            "starttime_index": int(cfg.solver.starttime_index),
            "transition_jump_steps": int(cfg.model.transition_jump_steps),
        },
        "sampling": {
            "seed": int(cfg.sampling.seed),
            "generation_workers_requested": int(cfg.sampling.generation_workers),
            "generation_workers_used": int(generation_workers),
            "jax_backend": str(jax_backend),
            "parameter_names_config": [x.name for x in cfg.sampling.parameters],
            "conditioning_param_names": list(CONDITIONING_PARAM_NAMES),
            "jax_sim_batch_size": int(sim_batch_size),
        },
        "training_split": {
            "split_seed": int(cfg.training.split_seed),
            "val_fraction": float(cfg.training.val_fraction),
            "test_fraction": float(cfg.training.test_fraction),
        },
        "items": written,
    }

    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Dataset generation complete -> %s", dataset_dir)
    return manifest
