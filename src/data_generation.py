"""Raw dataset generation for trajectory-transition emulator training.

The generator samples physical parameters, runs MY_SWAMP to obtain one
contiguous transition window per simulation, and writes those windows in the
raw format consumed by preprocessing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import (
    GCMulatorConfig,
    PHYSICAL_STATE_FIELDS,
    PROGNOSTIC_TARGET_FIELDS,
    SECONDS_PER_DAY,
    resolve_path,
)
from geometry import apply_geometry_state
from my_swamp_backend import (
    conditioning_param_names,
    detect_jax_backend,
    ensure_my_swamp_importable,
    params_to_conditioning_vector,
    params_to_public_json_dict,
    run_trajectory_window,
    run_trajectory_windows_batched,
)
from sampling import sample_parameter_dict, to_extended9


LOGGER = logging.getLogger("generate")


def _list_existing_raw_files(dataset_dir: Path) -> List[Path]:
    """Return all raw simulation files, including unsupported legacy leftovers."""
    return sorted(list(dataset_dir.glob("sim_*.npy")) + list(dataset_dir.glob("sim_*.npz")))


def _clear_dataset_dir(dataset_dir: Path) -> None:
    """Remove existing generated simulation files and manifest."""
    for path in _list_existing_raw_files(dataset_dir):
        path.unlink()
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def _sample_window_start_step(*, cfg: GCMulatorConfig, rng: np.random.Generator) -> int:
    """Sample one contiguous post-burn-in transition window start index."""
    total_steps = int(
        round(float(cfg.solver.default_time_days) * SECONDS_PER_DAY / float(cfg.solver.dt_seconds))
    )
    total_steps = max(1, total_steps)
    burn_in_steps = int(
        round(float(cfg.sampling.burn_in_days) * SECONDS_PER_DAY / float(cfg.solver.dt_seconds))
    )
    max_window_start = (
        total_steps
        - int(cfg.sampling.transition_jump_steps)
        - int(cfg.sampling.transitions_per_simulation)
        + 1
    )
    if max_window_start < burn_in_steps:
        raise ValueError(
            "No valid post-burn-in transition window exists for the configured horizon: "
            f"burn_in_steps={burn_in_steps}, max_window_start={max_window_start}"
        )
    if max_window_start == burn_in_steps:
        return int(burn_in_steps)
    return int(rng.integers(burn_in_steps, max_window_start + 1))


def _write_sim_record(
    *,
    sim_idx: int,
    state_inputs: np.ndarray,
    state_targets: np.ndarray,
    transition_days: np.ndarray,
    anchor_steps: np.ndarray,
    params_vector: np.ndarray,
    params_json: Dict[str, float],
    cfg: GCMulatorConfig,
    dataset_dir: Path,
) -> Dict[str, Any]:
    """Write one trajectory-window simulation into a raw ``sim_XXXXXX.npy`` record."""
    if state_inputs.ndim != 4 or state_targets.ndim != 4:
        raise ValueError("state_inputs/state_targets must be [T,C,H,W]")
    if state_inputs.shape[0] != state_targets.shape[0]:
        raise ValueError("state_inputs/state_targets must have the same transition count")
    if transition_days.ndim != 1 or int(transition_days.shape[0]) != int(state_inputs.shape[0]):
        raise ValueError("transition_days must align with transition count")
    if anchor_steps.shape != transition_days.shape:
        raise ValueError("anchor_steps must align with transition count")

    t_len = int(state_inputs.shape[0])
    inputs_geom = np.empty_like(state_inputs, dtype=np.float64)
    targets_geom = np.empty_like(state_targets, dtype=np.float64)
    geometry_info: Dict[str, Any] | None = None

    # Store geometry-canonicalized inputs and targets so downstream stages never
    # need to infer whether a file was already flipped or rolled.
    for time_index in range(t_len):
        input_state, input_geom = apply_geometry_state(
            state_inputs[time_index].astype(np.float64, copy=False),
            flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
            roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
        )
        target_state, target_geom = apply_geometry_state(
            state_targets[time_index].astype(np.float64, copy=False),
            flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
            roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
        )
        if input_geom != target_geom:
            raise RuntimeError("Geometry conversion mismatch between input and target states")
        if geometry_info is None:
            geometry_info = dict(input_geom)
        elif geometry_info != dict(input_geom):
            raise RuntimeError("Inconsistent geometry metadata across trajectory transitions")
        inputs_geom[time_index] = input_state
        targets_geom[time_index] = target_state

    if geometry_info is None:
        raise RuntimeError("Failed to infer geometry metadata for trajectory record")

    raw_path = dataset_dir / f"sim_{sim_idx:06d}.npy"
    payload = {
        "state_inputs": inputs_geom,
        "state_targets": targets_geom,
        "transition_days": transition_days.astype(np.float64, copy=False),
        "anchor_steps": anchor_steps.astype(np.int64, copy=False),
        "input_fields": np.asarray(list(PHYSICAL_STATE_FIELDS), dtype=object),
        "target_fields": np.asarray(list(PROGNOSTIC_TARGET_FIELDS), dtype=object),
        "params": params_vector.astype(np.float64, copy=False),
        "param_names": np.asarray(list(conditioning_param_names()), dtype=object),
        "default_time_days": np.asarray(
            float(cfg.solver.default_time_days),
            dtype=np.float64,
        ),
        "burn_in_days": np.asarray(float(cfg.sampling.burn_in_days), dtype=np.float64),
        "dt_seconds": np.asarray(float(cfg.solver.dt_seconds), dtype=np.float64),
        "starttime_index": np.asarray(int(cfg.solver.starttime_index), dtype=np.int64),
        "transition_jump_steps": np.asarray(
            int(cfg.sampling.transition_jump_steps),
            dtype=np.int64,
        ),
        "n_transitions": np.asarray(int(t_len), dtype=np.int64),
        "M": np.asarray(int(cfg.solver.M), dtype=np.int64),
        "nlat": np.asarray(int(inputs_geom.shape[-2]), dtype=np.int64),
        "nlon": np.asarray(int(inputs_geom.shape[-1]), dtype=np.int64),
        "lat_order": np.asarray(str(geometry_info["lat_order"]), dtype=object),
        "lon_origin": np.asarray(str(geometry_info["lon_origin"]), dtype=object),
        "lon_shift": np.asarray(int(geometry_info["lon_shift"]), dtype=np.int64),
    }
    np.save(raw_path, payload, allow_pickle=True)

    return {
        "sim_idx": int(sim_idx),
        "file": raw_path.name,
        "input_fields": list(PHYSICAL_STATE_FIELDS),
        "target_fields": list(PROGNOSTIC_TARGET_FIELDS),
        "param_names": list(conditioning_param_names()),
        "params": params_json,
        "default_time_days": float(cfg.solver.default_time_days),
        "burn_in_days": float(cfg.sampling.burn_in_days),
        "dt_seconds": float(cfg.solver.dt_seconds),
        "starttime_index": int(cfg.solver.starttime_index),
        "transition_jump_steps": int(cfg.sampling.transition_jump_steps),
        "n_transitions": int(t_len),
        "anchor_step_start": int(anchor_steps[0]),
        "anchor_step_end": int(anchor_steps[-1]),
        "transition_days_min": float(np.min(transition_days)),
        "transition_days_max": float(np.max(transition_days)),
        "M": int(cfg.solver.M),
        "nlat": int(inputs_geom.shape[-2]),
        "nlon": int(inputs_geom.shape[-1]),
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


def generate_dataset(cfg: GCMulatorConfig, *, config_path: Path) -> Dict[str, Any]:
    """Generate the raw transition dataset using MY_SWAMP as the source of truth."""
    ensure_my_swamp_importable(config_path.parent)

    dataset_dir = resolve_path(config_path, cfg.paths.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if cfg.paths.overwrite_dataset:
        _clear_dataset_dir(dataset_dir)
    else:
        existing = _list_existing_raw_files(dataset_dir)
        if existing:
            raise FileExistsError(
                "Dataset directory already contains raw simulation files "
                f"({len(existing)} .npy/.npz files) and overwrite_dataset=false: {dataset_dir}"
            )

    n_sims = int(cfg.sampling.n_sims)
    rng = np.random.default_rng(int(cfg.sampling.seed))
    sampled_params = [
        to_extended9(sample_parameter_dict(rng, cfg.sampling.parameters))
        for _ in range(n_sims)
    ]
    window_start_steps = np.asarray(
        [_sample_window_start_step(cfg=cfg, rng=rng) for _ in range(n_sims)],
        dtype=np.int64,
    )
    generation_batch_size = max(1, int(cfg.sampling.generation_workers))

    jax_backend = detect_jax_backend()
    LOGGER.info(
        "Dataset generation | backend=%-6s | batch_size=%d | input_fields=%s | target_fields=%s",
        jax_backend,
        generation_batch_size,
        list(PHYSICAL_STATE_FIELDS),
        list(PROGNOSTIC_TARGET_FIELDS),
    )

    items: List[Dict[str, Any]] = []
    start_time = time.time()
    log_every = max(1, n_sims // 20)

    if generation_batch_size == 1:
        for sim_idx, params in enumerate(sampled_params):
            state_inputs, state_targets, transition_days, anchor_steps = run_trajectory_window(
                params,
                M=int(cfg.solver.M),
                dt_seconds=float(cfg.solver.dt_seconds),
                time_days=float(cfg.solver.default_time_days),
                starttime_index=int(cfg.solver.starttime_index),
                window_start_step=int(window_start_steps[sim_idx]),
                n_transitions=int(cfg.sampling.transitions_per_simulation),
                transition_jump_steps=int(cfg.sampling.transition_jump_steps),
            )
            item = _write_sim_record(
                sim_idx=sim_idx,
                state_inputs=state_inputs,
                state_targets=state_targets,
                transition_days=transition_days,
                anchor_steps=anchor_steps,
                params_vector=params_to_conditioning_vector(params),
                params_json=params_to_public_json_dict(params),
                cfg=cfg,
                dataset_dir=dataset_dir,
            )
            items.append(item)
            completed = sim_idx + 1
            if (completed % log_every) == 0 or completed == n_sims:
                _log_progress(completed=completed, total=n_sims, start_time=start_time)
    else:
        for batch_start in range(0, n_sims, generation_batch_size):
            batch_end = min(batch_start + generation_batch_size, n_sims)
            params_batch = sampled_params[batch_start:batch_end]
            params_matrix = np.asarray(
                [params_to_conditioning_vector(params) for params in params_batch],
                dtype=np.float64,
            )
            batch_window_starts = window_start_steps[batch_start:batch_end]
            state_inputs_batch, state_targets_batch, transition_days_batch, anchor_steps_batch = (
                run_trajectory_windows_batched(
                    params_matrix,
                    M=int(cfg.solver.M),
                    dt_seconds=float(cfg.solver.dt_seconds),
                    time_days=float(cfg.solver.default_time_days),
                    starttime_index=int(cfg.solver.starttime_index),
                    window_start_steps=batch_window_starts,
                    n_transitions=int(cfg.sampling.transitions_per_simulation),
                    transition_jump_steps=int(cfg.sampling.transition_jump_steps),
                    k6=float(params_batch[0].K6),
                    k6phi=params_batch[0].K6Phi,
                )
            )
            for local_index, params in enumerate(params_batch):
                sim_idx = batch_start + local_index
                item = _write_sim_record(
                    sim_idx=sim_idx,
                    state_inputs=state_inputs_batch[local_index],
                    state_targets=state_targets_batch[local_index],
                    transition_days=transition_days_batch[local_index],
                    anchor_steps=anchor_steps_batch[local_index],
                    params_vector=params_to_conditioning_vector(params),
                    params_json=params_to_public_json_dict(params),
                    cfg=cfg,
                    dataset_dir=dataset_dir,
                )
                items.append(item)

            completed = batch_end
            if (completed % log_every) == 0 or completed == n_sims:
                _log_progress(completed=completed, total=n_sims, start_time=start_time)

    # The manifest is purely descriptive; training reads the raw files directly.
    manifest = {
        "created_unix": time.time(),
        "n_sims_requested": n_sims,
        "n_sims_written": len(items),
        "input_fields": list(PHYSICAL_STATE_FIELDS),
        "target_fields": list(PROGNOSTIC_TARGET_FIELDS),
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
            "burn_in_days": float(cfg.sampling.burn_in_days),
            "transitions_per_simulation": int(cfg.sampling.transitions_per_simulation),
            "transition_jump_steps": int(cfg.sampling.transition_jump_steps),
        },
        "geometry": {
            "flip_latitude_to_north_south": bool(cfg.geometry.flip_latitude_to_north_south),
            "roll_longitude_to_0_2pi": bool(cfg.geometry.roll_longitude_to_0_2pi),
        },
        "items": items,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
