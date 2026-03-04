"""Raw dataset generation pipeline using MY_SWAMP terminal states."""

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
    detect_jax_backend,
    ensure_my_swamp_importable,
    param_names_extended9,
    params_to_json_dict,
    params_to_ordered_vector,
    run_terminal_state_batch,
    run_terminal_state,
)
from .sampling import EXTENDED9_PARAM_NAMES, sample_parameter_dict, to_extended9

LOGGER = logging.getLogger("src.generate")
GPU_BACKENDS = {"gpu", "cuda", "rocm", "metal"}


def _clear_dataset_dir(dataset_dir: Path) -> None:
    """Remove existing generated simulation files and manifest."""
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
    """Resolve per-call JAX batch size for independent terminal simulations.

    Environment knob:
    - ``GCMULATOR_JAX_SIM_BATCH=auto`` (default): use a conservative GPU batch.
    - ``GCMULATOR_JAX_SIM_BATCH=<int>=1``: force explicit batch size.
    """
    raw = os.environ.get("GCMULATOR_JAX_SIM_BATCH", "auto").strip().lower()
    if raw in {"", "auto"}:
        if jax_backend.lower() in GPU_BACKENDS and int(generation_workers) == 1:
            batch = 4
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
            "Batched vmap generation is only used with generation_workers=1; falling back to 1.",
            batch,
            int(generation_workers),
        )
        batch = 1
    return min(batch, int(n_sims))


def _write_sim_record(
    *,
    sim_idx: int,
    params: Extended9Params,
    state: Any,
    cfg: GCMulatorConfig,
    dataset_dir: Path,
    param_names: List[str],
) -> Dict[str, Any]:
    """Write one terminal state into a ``sim_XXXXXX.npz`` file and metadata record."""
    state_chw = state.as_stacked().astype(np.float32, copy=False)
    state_chw, geom_info = apply_geometry_state(
        state_chw,
        flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
        roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
    )

    sim_file = dataset_dir / f"sim_{sim_idx:06d}.npz"
    payload = {
        "state_final": state_chw,
        "fields": np.asarray(FIELDS_5, dtype=object),
        "params": params_to_ordered_vector(params),
        "param_names": np.asarray(param_names, dtype=object),
        "time_days": np.asarray(float(cfg.solver.default_time_days), dtype=np.float64),
        "dt_seconds": np.asarray(float(cfg.solver.dt_seconds), dtype=np.float64),
        "M": np.asarray(int(cfg.solver.M), dtype=np.int64),
        "nlat": np.asarray(int(state_chw.shape[-2]), dtype=np.int64),
        "nlon": np.asarray(int(state_chw.shape[-1]), dtype=np.int64),
        "lat_order": np.asarray(geom_info["lat_order"], dtype=object),
        "lon_origin": np.asarray(geom_info["lon_origin"], dtype=object),
        "lon_shift": np.asarray(int(geom_info["lon_shift"]), dtype=np.int64),
    }
    compress_raw = os.environ.get("GCMULATOR_COMPRESS_RAW", "0") == "1"
    if compress_raw:
        np.savez_compressed(sim_file, **payload)
    else:
        np.savez(sim_file, **payload)

    return {
        "sim_idx": int(sim_idx),
        "file": sim_file.name,
        "fields": list(FIELDS_5),
        "param_names": list(param_names),
        "params": params_to_json_dict(params),
        "time_days": float(cfg.solver.default_time_days),
        "dt_seconds": float(cfg.solver.dt_seconds),
        "M": int(cfg.solver.M),
        "nlat": int(state_chw.shape[-2]),
        "nlon": int(state_chw.shape[-1]),
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
    """Run one simulation and write one ``sim_XXXXXX.npz`` output file."""
    state = run_terminal_state(
        params,
        M=cfg.solver.M,
        dt_seconds=cfg.solver.dt_seconds,
        time_days=cfg.solver.default_time_days,
        starttime_index=cfg.solver.starttime_index,
    )
    return _write_sim_record(
        sim_idx=sim_idx,
        params=params,
        state=state,
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
        "Generated %d/%d terminal sims | elapsed=%.1fs | ETA=%.1fs",
        completed,
        total,
        elapsed,
        remain,
    )


def generate_dataset(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Generate terminal-state dataset using MY_SWAMP truth only."""
    ensure_my_swamp_importable(config_path.parent)

    dataset_dir = resolve_path(config_path, cfg.paths.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if cfg.paths.overwrite_dataset:
        _clear_dataset_dir(dataset_dir)
    else:
        existing = list(dataset_dir.glob("sim_*.npz"))
        if existing:
            raise FileExistsError(
                f"Dataset directory already contains {len(existing)} simulation files and overwrite_dataset=false: {dataset_dir}"
            )

    n_sims = int(cfg.sampling.n_sims)
    rng = np.random.default_rng(cfg.sampling.seed)
    param_names = list(param_names_extended9())
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
    compress_raw = os.environ.get("GCMULATOR_COMPRESS_RAW", "0") == "1"
    LOGGER.info(
        "Dataset generation backend=%s | generation_workers=%d (requested=%d) | sim_batch_size=%d | raw_compression=%s",
        jax_backend,
        generation_workers,
        int(cfg.sampling.generation_workers),
        sim_batch_size,
        "on" if compress_raw else "off",
    )
    if generation_workers == 1 and sim_batch_size > 1:
        LOGGER.info(
            "Batched terminal generation is active: vmap batch size=%d.",
            sim_batch_size,
        )
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
        if sim_batch_size == 1:
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
            full_batches = n_sims // sim_batch_size
            for batch_idx in range(full_batches):
                i0 = batch_idx * sim_batch_size
                i1 = i0 + sim_batch_size
                params_batch = sampled_params[i0:i1]
                states = run_terminal_state_batch(
                    params_batch,
                    M=cfg.solver.M,
                    dt_seconds=cfg.solver.dt_seconds,
                    time_days=cfg.solver.default_time_days,
                    starttime_index=cfg.solver.starttime_index,
                )
                if len(states) != len(params_batch):
                    raise RuntimeError(
                        f"Batched run returned {len(states)} states for {len(params_batch)} params."
                    )
                for offset, (params, state) in enumerate(zip(params_batch, states)):
                    sim_idx = i0 + offset
                    written_by_idx[sim_idx] = _write_sim_record(
                        sim_idx=sim_idx,
                        params=params,
                        state=state,
                        cfg=cfg,
                        dataset_dir=dataset_dir,
                        param_names=param_names,
                    )
                    completed += 1
                    if (completed % log_every) == 0 or completed == n_sims:
                        _log_progress(completed=completed, total=n_sims, start_t=start_t)

            remainder_start = full_batches * sim_batch_size
            for sim_idx in range(remainder_start, n_sims):
                params = sampled_params[sim_idx]
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
        },
        "sampling": {
            "seed": int(cfg.sampling.seed),
            "generation_workers_requested": int(cfg.sampling.generation_workers),
            "generation_workers_used": int(generation_workers),
            "jax_backend": str(jax_backend),
            "parameter_names_config": [x.name for x in cfg.sampling.parameters],
            "extended9_param_names": list(EXTENDED9_PARAM_NAMES),
            "jax_sim_batch_size": int(sim_batch_size),
        },
        "items": written,
    }

    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Dataset generation complete -> %s", dataset_dir)
    return manifest
