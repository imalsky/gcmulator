"""Raw dataset generation for variable-dt trajectory-transition emulator training.

The generator samples physical parameters, runs MY_SWAMP to obtain one
trajectory per simulation, then samples random (anchor, target) pairs with
log-uniform time jumps.  Each transition within a simulation may use a
different time gap, enabling the model to learn dynamics across all
timescales — from fast transients to slow relaxation toward equilibrium.

Setting ``burn_in_days`` to 0 allows anchor times at the very start of the
simulation so the model can learn to evolve an arbitrary initial state
(e.g. a rest state) toward the forced equilibrium.
"""

from __future__ import annotations

import json
import logging
import os
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
from sampling import sample_parameter_dict, sample_transition_pairs, to_extended9


LOGGER = logging.getLogger("generate")
GPU_BACKENDS = {"gpu", "cuda", "rocm", "metal"}


def _list_existing_raw_files(dataset_dir: Path) -> List[Path]:
    """Return all raw simulation files, including unsupported legacy leftovers."""
    return sorted(
        list(dataset_dir.glob("sim_*.npy")) + list(dataset_dir.glob("sim_*.npz"))
    )


def _clear_dataset_dir(dataset_dir: Path) -> None:
    """Remove existing generated simulation files and manifest."""
    for path in _list_existing_raw_files(dataset_dir):
        path.unlink()
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


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
    if int(state_inputs.shape[0]) == 0:
        raise RuntimeError("Failed to infer geometry metadata for trajectory record")

    inputs_geom, geometry_info = apply_geometry_state(
        state_inputs.astype(np.float64, copy=True),
        flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
        roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
    )
    targets_geom, _ = apply_geometry_state(
        state_targets.astype(np.float64, copy=True),
        flip_latitude_to_north_south=cfg.geometry.flip_latitude_to_north_south,
        roll_longitude_to_0_2pi=cfg.geometry.roll_longitude_to_0_2pi,
    )

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
        "transition_jump_days_min": np.asarray(
            float(cfg.sampling.transition_jump_days_min), dtype=np.float64
        ),
        "transition_jump_days_max": np.asarray(
            float(cfg.sampling.transition_jump_days_max), dtype=np.float64
        ),
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
        "transition_jump_days_min": float(cfg.sampling.transition_jump_days_min),
        "transition_jump_days_max": float(cfg.sampling.transition_jump_days_max),
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
    """Generate the raw transition dataset using MY_SWAMP as the source of truth.

    For each simulation, physical parameters are sampled from the configured
    distributions and MY_SWAMP is run for ``default_time_days``.  Random
    (anchor, target) pairs are then drawn with log-uniform time jumps so that
    the model sees a wide range of timescales — from sub-day transients to
    near-equilibrium relaxation over tens of days.

    When ``burn_in_days`` is 0, anchor times can start at the very beginning
    of the trajectory (right after the ``starttime_index`` numerical warm-up).
    This is critical for training the model to evolve an initial state toward
    the forced equilibrium.
    """
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
                f"({len(existing)} .npy/.npz files) and overwrite_dataset=false: "
                f"{dataset_dir}"
            )

    n_sims = int(cfg.sampling.n_sims)
    rng = np.random.default_rng(int(cfg.sampling.seed))

    n_steps_total = max(
        1,
        int(round(
            float(cfg.solver.default_time_days) * SECONDS_PER_DAY
            / float(cfg.solver.dt_seconds)
        )),
    )
    burn_in_steps = int(round(
        float(cfg.sampling.burn_in_days) * SECONDS_PER_DAY
        / float(cfg.solver.dt_seconds)
    ))

    jax_backend = detect_jax_backend()
    generation_batch_size = _resolve_generation_batch_size(
        requested_workers=int(cfg.sampling.generation_workers),
        n_sims=n_sims,
        jax_backend=jax_backend,
    )
    LOGGER.info(
        "Dataset generation | backend=%-6s | n_sims=%d | generation_batch_size=%d "
        "| jump_days=[%.3f,%.3f] | input_fields=%s | target_fields=%s",
        jax_backend,
        n_sims,
        generation_batch_size,
        float(cfg.sampling.transition_jump_days_min),
        float(cfg.sampling.transition_jump_days_max),
        list(PHYSICAL_STATE_FIELDS),
        list(PROGNOSTIC_TARGET_FIELDS),
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
        anchor_steps, target_steps, transition_days = sample_transition_pairs(
            rng,
            n_transitions=int(cfg.sampling.transitions_per_simulation),
            burn_in_steps=burn_in_steps,
            n_steps_total=n_steps_total,
            dt_seconds=float(cfg.solver.dt_seconds),
            transition_jump_days_min=float(cfg.sampling.transition_jump_days_min),
            transition_jump_days_max=float(cfg.sampling.transition_jump_days_max),
        )
        sampled_runs.append(
            {
                "sim_idx": int(sim_idx),
                "params": params,
                "anchor_steps": anchor_steps,
                "target_steps": target_steps,
                "transition_days": transition_days,
            }
        )

    for batch_start in range(0, n_sims, generation_batch_size):
        batch = sampled_runs[batch_start:batch_start + generation_batch_size]
        if len(batch) == 1:
            entry = batch[0]
            state_inputs, state_targets = run_trajectory_window(
                entry["params"],
                M=int(cfg.solver.M),
                dt_seconds=float(cfg.solver.dt_seconds),
                time_days=float(cfg.solver.default_time_days),
                starttime_index=int(cfg.solver.starttime_index),
                anchor_steps=entry["anchor_steps"],
                target_steps=entry["target_steps"],
            )
            batch_state_inputs = np.asarray(state_inputs, dtype=np.float64)[None, ...]
            batch_state_targets = np.asarray(state_targets, dtype=np.float64)[None, ...]
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
            anchor_steps_batch = np.stack(
                [entry["anchor_steps"] for entry in batch],
                axis=0,
            )
            target_steps_batch = np.stack(
                [entry["target_steps"] for entry in batch],
                axis=0,
            )
            batch_state_inputs, batch_state_targets = run_trajectory_windows_batched(
                params_batch,
                M=int(cfg.solver.M),
                dt_seconds=float(cfg.solver.dt_seconds),
                time_days=float(cfg.solver.default_time_days),
                starttime_index=int(cfg.solver.starttime_index),
                anchor_steps_batch=anchor_steps_batch,
                target_steps_batch=target_steps_batch,
                k6=float(next(iter(k6_values))),
                k6phi=next(iter(k6phi_values)),
            )

        for batch_index, entry in enumerate(batch):
            item = _write_sim_record(
                sim_idx=int(entry["sim_idx"]),
                state_inputs=batch_state_inputs[batch_index],
                state_targets=batch_state_targets[batch_index],
                transition_days=entry["transition_days"],
                anchor_steps=entry["anchor_steps"],
                params_vector=params_to_conditioning_vector(entry["params"]),
                params_json=params_to_public_json_dict(entry["params"]),
                cfg=cfg,
                dataset_dir=dataset_dir,
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
            "generation_workers": int(cfg.sampling.generation_workers),
            "resolved_generation_batch_size": int(generation_batch_size),
            "burn_in_days": float(cfg.sampling.burn_in_days),
            "transitions_per_simulation": int(cfg.sampling.transitions_per_simulation),
            "transition_jump_days_min": float(cfg.sampling.transition_jump_days_min),
            "transition_jump_days_max": float(cfg.sampling.transition_jump_days_max),
            "uses_variable_transition_jump": bool(cfg.sampling.uses_variable_transition_jump()),
        },
        "geometry": {
            "flip_latitude_to_north_south": bool(cfg.geometry.flip_latitude_to_north_south),
            "roll_longitude_to_0_2pi": bool(cfg.geometry.roll_longitude_to_0_2pi),
        },
        "items": items,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
