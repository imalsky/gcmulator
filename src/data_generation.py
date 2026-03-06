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
from geometry import geometry_shift_for_nlon
from my_swamp_backend import (
    conditioning_param_names,
    detect_jax_backend,
    ensure_my_swamp_importable,
    params_to_conditioning_vector,
    params_to_public_json_dict,
    run_trajectory_window,
    run_trajectory_windows_batched,
)
from sampling import sample_parameter_dict, sample_transition_jump_steps, to_extended9


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


def _sample_window_start_step(
    *,
    cfg: GCMulatorConfig,
    transition_jump_steps: int,
    rng: np.random.Generator,
) -> int:
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
        - int(transition_jump_steps) * int(cfg.sampling.transitions_per_simulation)
    )
    if max_window_start < burn_in_steps:
        raise ValueError(
            "No valid post-burn-in transition window exists for the configured horizon: "
            f"transition_jump_steps={transition_jump_steps}, "
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
    transition_jump_steps: int,
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

    if int(state_inputs.shape[0]) == 0:
        raise RuntimeError("Failed to infer geometry metadata for trajectory record")

    # Vectorized geometry canonicalization: apply the same flip/roll to the
    # entire [T, C, H, W] batch instead of looping over individual timesteps.
    nlon = int(state_inputs.shape[3])
    lon_shift = geometry_shift_for_nlon(nlon, cfg.geometry.roll_longitude_to_0_2pi)

    inputs_geom = state_inputs.astype(np.float64, copy=True)
    targets_geom = state_targets.astype(np.float64, copy=True)
    if cfg.geometry.flip_latitude_to_north_south:
        inputs_geom = inputs_geom[:, :, ::-1, :]
        targets_geom = targets_geom[:, :, ::-1, :]
    if lon_shift:
        inputs_geom = np.roll(inputs_geom, shift=int(lon_shift), axis=-1)
        targets_geom = np.roll(targets_geom, shift=int(lon_shift), axis=-1)
    inputs_geom = np.ascontiguousarray(inputs_geom)
    targets_geom = np.ascontiguousarray(targets_geom)

    nlat = int(state_inputs.shape[2])
    geometry_info = {
        "lat_order": "north_to_south" if cfg.geometry.flip_latitude_to_north_south else "south_to_north",
        "lon_origin": "0_to_2pi" if cfg.geometry.roll_longitude_to_0_2pi else "minus_pi_to_pi",
        "lon_shift": int(lon_shift),
        "nlat": int(nlat),
        "nlon": int(nlon),
    }

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
        "transition_jump_steps": np.asarray(int(transition_jump_steps), dtype=np.int64),
        "anchor_stride_steps": np.asarray(int(transition_jump_steps), dtype=np.int64),
        "n_transitions": np.asarray(int(state_inputs.shape[0]), dtype=np.int64),
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
        "transition_jump_steps": int(transition_jump_steps),
        "anchor_stride_steps": int(transition_jump_steps),
        "n_transitions": int(state_inputs.shape[0]),
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
    transition_jump_steps_per_sim = sample_transition_jump_steps(
        rng,
        cfg.sampling,
        n_samples=n_sims,
    )
    window_start_steps = np.asarray(
        [
            _sample_window_start_step(
                cfg=cfg,
                transition_jump_steps=int(transition_jump_steps_per_sim[index]),
                rng=rng,
            )
            for index in range(n_sims)
        ],
        dtype=np.int64,
    )
    generation_batch_size = max(1, int(cfg.sampling.generation_workers))

    jax_backend = detect_jax_backend()
    LOGGER.info(
        "Dataset generation | backend=%-6s | batch_size=%d | jump_steps=[%d,%d] | "
        "input_fields=%s | target_fields=%s",
        jax_backend,
        generation_batch_size,
        int(np.min(transition_jump_steps_per_sim)),
        int(np.max(transition_jump_steps_per_sim)),
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
                transition_jump_steps=int(transition_jump_steps_per_sim[sim_idx]),
            )
            item = _write_sim_record(
                sim_idx=sim_idx,
                state_inputs=state_inputs,
                state_targets=state_targets,
                transition_days=transition_days,
                anchor_steps=anchor_steps,
                transition_jump_steps=int(transition_jump_steps_per_sim[sim_idx]),
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
        completed = 0
        for transition_jump_steps in sorted(
            {int(value) for value in transition_jump_steps_per_sim.tolist()}
        ):
            group_indices = np.flatnonzero(
                transition_jump_steps_per_sim == int(transition_jump_steps)
            ).astype(np.int64)
            for batch_start in range(0, int(group_indices.shape[0]), generation_batch_size):
                batch_indices = group_indices[
                    batch_start : batch_start + generation_batch_size
                ]
                params_batch = [sampled_params[int(index)] for index in batch_indices.tolist()]
                params_matrix = np.asarray(
                    [params_to_conditioning_vector(params) for params in params_batch],
                    dtype=np.float64,
                )
                batch_window_starts = window_start_steps[batch_indices]
                state_inputs_batch, state_targets_batch, transition_days_batch, anchor_steps_batch = (
                    run_trajectory_windows_batched(
                        params_matrix,
                        M=int(cfg.solver.M),
                        dt_seconds=float(cfg.solver.dt_seconds),
                        time_days=float(cfg.solver.default_time_days),
                        starttime_index=int(cfg.solver.starttime_index),
                        window_start_steps=batch_window_starts,
                        n_transitions=int(cfg.sampling.transitions_per_simulation),
                        transition_jump_steps=int(transition_jump_steps),
                        k6=float(params_batch[0].K6),
                        k6phi=params_batch[0].K6Phi,
                    )
                )
                for local_index, params in enumerate(params_batch):
                    sim_idx = int(batch_indices[local_index])
                    item = _write_sim_record(
                        sim_idx=sim_idx,
                        state_inputs=state_inputs_batch[local_index],
                        state_targets=state_targets_batch[local_index],
                        transition_days=transition_days_batch[local_index],
                        anchor_steps=anchor_steps_batch[local_index],
                        transition_jump_steps=int(transition_jump_steps),
                        params_vector=params_to_conditioning_vector(params),
                        params_json=params_to_public_json_dict(params),
                        cfg=cfg,
                        dataset_dir=dataset_dir,
                    )
                    items.append(item)

                completed += int(batch_indices.shape[0])
                if (completed % log_every) == 0 or completed == n_sims:
                    _log_progress(completed=completed, total=n_sims, start_time=start_time)

    # The manifest is purely descriptive; training reads the raw files directly.
    items.sort(key=lambda item: int(item["sim_idx"]))
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
            "transition_jump_steps": int(cfg.sampling.min_transition_jump_steps()),
            "transition_jump_steps_max": int(cfg.sampling.max_transition_jump_steps()),
            "uses_variable_transition_jump": bool(cfg.sampling.uses_variable_transition_jump()),
        },
        "geometry": {
            "flip_latitude_to_north_south": bool(cfg.geometry.flip_latitude_to_north_south),
            "roll_longitude_to_0_2pi": bool(cfg.geometry.roll_longitude_to_0_2pi),
        },
        "items": items,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
