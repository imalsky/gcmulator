"""Benchmark held-out multi-jump rollout latency on the test split."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_MPL_CACHE_DIR = PROJECT_CACHE_DIR / "mplconfig"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR.resolve()))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", str(DEFAULT_MPL_CACHE_DIR))).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch


SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling import build_state_conditioned_transition_model, ensure_torch_harmonics_importable
from config import resolve_path
from my_swamp_backend import (
    ensure_my_swamp_importable,
    reconstruct_full_state_from_prognostics,
    run_trajectory_windows_batched,
)
from normalization import (
    denormalize_state_tensor,
    normalize_conditioning,
    normalize_state_tensor,
    stats_from_json,
)
from sampling import INTERNAL_FIXED_K6, INTERNAL_FIXED_K6PHI, to_extended9


BATCH_SIZES = (1, 2, 4)
TEST_SHARD_START = 0
INPUT_TIME_DAYS = 0.0
TARGET_TIME_DAYS = 10.0
NUM_JUMPS = 1
WARMUP_RUNS = 1
TIMING_REPEATS = 5
FIGURE_DPI = 180
FIGURE_NAME = "benchmark_test_rollout_latency_vs_batch_size.png"
CSV_NAME = "benchmark_test_rollout_latency_vs_batch_size.csv"
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")
DEVICE_COLORS = {
    "cpu": "#1d4ed8",
    "cuda": "#b45309",
}
DEVICE_MARKERS = {
    "cpu": "o",
    "cuda": "s",
}

# User-editable run settings
RUN_NAME = "v1"
RUN_DIR: Path | None = (PROJECT_ROOT / "models" / RUN_NAME).resolve()
CHECKPOINT_PATH: Path | None = None
PROCESSED_DIR: Path | None = None
DEVICE_MODES: Sequence[str] = ("cpu", "gpu")
FIGURE_PATH: Path | None = None
CSV_PATH: Path | None = None


def _dict_to_namespace(obj: Any) -> Any:
    """Convert nested dictionaries into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _dict_to_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(value) for value in obj]
    return obj


def _resolve_checkpoint_path(*, run_dir: Path | None, checkpoint: Path | None) -> Path:
    """Resolve checkpoint path from the top-level run settings."""
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    if run_dir is None:
        raise ValueError("Set RUN_DIR or CHECKPOINT_PATH at the top of this file")
    ckpt_path = (run_dir.resolve() / "best.pt").resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _resolve_device(mode: str) -> torch.device:
    """Resolve one benchmark device."""
    normalized = str(mode).lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU benchmarking but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device mode: {mode}")


def _resolve_devices(modes: Sequence[str]) -> list[torch.device]:
    """Resolve requested devices, skipping unavailable GPU backends."""
    devices: list[torch.device] = []
    seen: set[str] = set()
    for mode in modes:
        try:
            device = _resolve_device(str(mode))
        except RuntimeError as exc:
            warnings.warn(str(exc))
            continue
        if device.type not in seen:
            devices.append(device)
            seen.add(device.type)
    if not devices:
        raise RuntimeError("No benchmark devices are available")
    return devices


def _resolve_processed_dir(ckpt: Dict[str, Any]) -> Path:
    """Resolve the processed directory, defaulting to the checkpoint's dataset."""
    if PROCESSED_DIR is not None:
        resolved = PROCESSED_DIR.resolve()
    else:
        source_config_path = ckpt.get("source_config_path")
        resolved_config = ckpt.get("resolved_config")
        if not isinstance(source_config_path, str) or not isinstance(resolved_config, dict):
            raise ValueError(
                "Checkpoint is missing source config metadata; set PROCESSED_DIR explicitly"
            )
        paths_cfg = resolved_config.get("paths")
        if not isinstance(paths_cfg, dict):
            raise ValueError(
                "Checkpoint is missing resolved paths metadata; set PROCESSED_DIR explicitly"
            )
        processed_dir_value = paths_cfg.get("processed_dir")
        if not isinstance(processed_dir_value, str):
            raise ValueError(
                "Checkpoint is missing paths.processed_dir; set PROCESSED_DIR explicitly"
            )
        resolved = resolve_path(Path(source_config_path), processed_dir_value)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {resolved}")
    return resolved


def _denormalize_params(params_norm: np.ndarray, stats: Any) -> Dict[str, float]:
    """Invert parameter normalization back to physical values."""
    params_norm = np.asarray(params_norm, dtype=np.float64)
    params_phys = params_norm * (stats.std + stats.zscore_eps) + stats.mean
    const_mask = np.asarray(stats.is_constant, dtype=bool)
    if np.any(const_mask):
        params_phys[const_mask] = stats.mean[const_mask]
    return {
        str(name): float(params_phys[index])
        for index, name in enumerate(stats.param_names)
    }


def _load_test_split_entries(processed_dir: Path) -> list[dict[str, Any]]:
    """Return held-out processed shard entries from the test split."""
    meta_path = (processed_dir / "processed_meta.json").resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"Processed metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split_entries = list(meta["splits"]["test"])
    if not split_entries:
        raise RuntimeError("Test split is empty")
    return split_entries


def _resolve_time_steps(
    *,
    input_time_days: float,
    target_time_days: float,
    dt_seconds: float,
    max_time_days: float,
) -> tuple[int, int, float, float]:
    """Convert requested day-valued times to validated solver steps."""
    if input_time_days < 0.0:
        raise ValueError("INPUT_TIME_DAYS must be >= 0")
    if target_time_days <= input_time_days:
        raise ValueError("TARGET_TIME_DAYS must be > INPUT_TIME_DAYS")
    if target_time_days > max_time_days:
        raise ValueError(
            "TARGET_TIME_DAYS exceeds the simulated horizon from the checkpoint: "
            f"{target_time_days} > {max_time_days}"
        )

    input_step = int(round(float(input_time_days) * 86400.0 / float(dt_seconds)))
    target_step = int(round(float(target_time_days) * 86400.0 / float(dt_seconds)))
    if target_step <= input_step:
        raise ValueError("Resolved target step must be greater than the input step")

    actual_input_day = float(input_step) * float(dt_seconds) / 86400.0
    actual_target_day = float(target_step) * float(dt_seconds) / 86400.0
    return input_step, target_step, actual_input_day, actual_target_day


def _build_jump_schedule(*, input_step: int, target_step: int, num_jumps: int) -> np.ndarray:
    """Split one solver-step interval into ``num_jumps`` positive integer jumps."""
    if num_jumps < 1:
        raise ValueError("NUM_JUMPS must be >= 1")
    total_steps = int(target_step) - int(input_step)
    if total_steps < 1:
        raise ValueError("target_step must be greater than input_step")
    if int(num_jumps) > total_steps:
        raise ValueError(
            "NUM_JUMPS cannot exceed the number of solver steps between input and target"
        )

    base = total_steps // int(num_jumps)
    remainder = total_steps % int(num_jumps)
    jump_steps = np.full((int(num_jumps),), base, dtype=np.int64)
    jump_steps[:remainder] += 1
    if np.any(jump_steps < 1):
        raise RuntimeError("Jump schedule construction produced a non-positive jump")
    return jump_steps


def _load_test_rollout_batch(
    *,
    processed_dir: Path,
    ckpt: Dict[str, Any],
    stats: Any,
    test_shard_start: int,
    max_batch_size: int,
    input_time_days: float,
    target_time_days: float,
    num_jumps: int,
) -> tuple[list[Any], np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Load one held-out test batch and construct one rollout input per shard."""
    test_entries = _load_test_split_entries(processed_dir)
    start = int(test_shard_start)
    stop = start + int(max_batch_size)
    if start < 0 or stop > len(test_entries):
        raise IndexError(
            f"Need test shards [{start}, {stop}), but only {len(test_entries)} test shards exist"
        )

    params_batch: list[Any] = []
    conditioning_rows: list[np.ndarray] = []
    for entry in test_entries[start:stop]:
        shard_path = (processed_dir / str(entry["file"])).resolve()
        if not shard_path.is_file():
            raise FileNotFoundError(f"Processed shard not found: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as npz:
            params_norm = np.asarray(npz["params_norm"], dtype=np.float32)
        params = to_extended9(_denormalize_params(params_norm, stats.params))
        params_batch.append(params)
        conditioning_rows.append(np.asarray(params.to_vector(), dtype=np.float64))

    solver_cfg = dict(ckpt["solver"])
    input_step, target_step, actual_input_day, actual_target_day = _resolve_time_steps(
        input_time_days=float(input_time_days),
        target_time_days=float(target_time_days),
        dt_seconds=float(solver_cfg["dt_seconds"]),
        max_time_days=float(solver_cfg["default_time_days"]),
    )
    jump_schedule = _build_jump_schedule(
        input_step=input_step,
        target_step=target_step,
        num_jumps=int(num_jumps),
    )
    params_array = np.stack(conditioning_rows, axis=0).astype(np.float64)
    anchor_steps = np.full((int(max_batch_size), 1), input_step, dtype=np.int64)
    target_steps = np.full((int(max_batch_size), 1), target_step, dtype=np.int64)
    state_inputs_phys, _ = run_trajectory_windows_batched(
        params_array,
        M=int(solver_cfg["M"]),
        dt_seconds=float(solver_cfg["dt_seconds"]),
        time_days=float(solver_cfg["default_time_days"]),
        starttime_index=int(solver_cfg["starttime_index"]),
        anchor_steps_batch=anchor_steps,
        target_steps_batch=target_steps,
        k6=float(INTERNAL_FIXED_K6),
        k6phi=INTERNAL_FIXED_K6PHI,
    )
    return (
        params_batch,
        params_array,
        np.asarray(state_inputs_phys[:, 0], dtype=np.float32),
        actual_input_day,
        actual_target_day,
        jump_schedule,
    )


def _sync_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator backends before or after timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _configure_runtime(device: torch.device) -> None:
    """Apply lightweight backend settings for repeatable inference timing."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _build_model(ckpt: Dict[str, Any], *, device: torch.device) -> torch.nn.Module:
    """Instantiate the trained transition model on one device."""
    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    input_fields = list(ckpt["input_fields"])
    target_fields = list(ckpt["target_fields"])
    residual_input_indices = [input_fields.index(field_name) for field_name in target_fields]
    model = build_state_conditioned_transition_model(
        img_size=(int(shape["H"]), int(shape["W"])),
        input_state_chans=int(shape["input_C"]),
        target_state_chans=int(shape["target_C"]),
        param_dim=int(len(ckpt["conditioning_names"])),
        residual_input_indices=residual_input_indices,
        cfg_model=model_cfg,
        lat_order="north_to_south",
        lon_origin="0_to_2pi",
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device=device).eval()
    return model


def _run_multi_jump_rollout_batch(
    *,
    model: torch.nn.Module,
    stats: Any,
    params_batch: Sequence[Any],
    params_array: np.ndarray,
    initial_states_phys: np.ndarray,
    jump_schedule: Sequence[int],
    solver_cfg: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Run one recursive multi-jump rollout batch and return the final prognostics."""
    current_states_phys = np.asarray(initial_states_phys, dtype=np.float32)
    n_samples = int(current_states_phys.shape[0])

    for jump_index, jump_steps in enumerate(jump_schedule):
        jump_days = float(jump_steps) * float(solver_cfg["dt_seconds"]) / 86400.0
        state_inputs_norm = normalize_state_tensor(
            current_states_phys,
            stats.input_state,
        ).astype(np.float32)
        conditioning_norm = normalize_conditioning(
            params_array,
            np.full((n_samples,), jump_days, dtype=np.float64),
            param_stats=stats.params,
            transition_time_stats=stats.transition_time,
        ).astype(np.float32)

        with torch.inference_mode():
            pred_norm = model(
                torch.from_numpy(state_inputs_norm).to(device=device),
                torch.from_numpy(conditioning_norm).to(device=device),
            )
        pred_phys = denormalize_state_tensor(
            pred_norm.detach().cpu().numpy(),
            stats.target_state,
        ).astype(np.float32)
        if jump_index == (len(jump_schedule) - 1):
            return pred_phys

        current_states_phys = np.stack(
            [
                reconstruct_full_state_from_prognostics(
                    np.asarray(pred_phys[sample_index], dtype=np.float64),
                    params=params_batch[sample_index],
                    M=int(solver_cfg["M"]),
                    dt_seconds=float(solver_cfg["dt_seconds"]),
                ).astype(np.float32)
                for sample_index in range(n_samples)
            ],
            axis=0,
        )

    raise RuntimeError("Multi-jump rollout did not produce a final prediction")


def _benchmark_batch(
    *,
    model: torch.nn.Module,
    stats: Any,
    params_batch: Sequence[Any],
    params_array: np.ndarray,
    initial_states_phys: np.ndarray,
    jump_schedule: Sequence[int],
    solver_cfg: Dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean and std held-out rollout latency in seconds."""
    sample_count = int(batch_size)
    times: List[float] = []

    with torch.inference_mode():
        for _ in range(int(WARMUP_RUNS)):
            _ = _run_multi_jump_rollout_batch(
                model=model,
                stats=stats,
                params_batch=params_batch[:sample_count],
                params_array=np.asarray(params_array[:sample_count], dtype=np.float64),
                initial_states_phys=np.asarray(initial_states_phys[:sample_count], dtype=np.float32),
                jump_schedule=jump_schedule,
                solver_cfg=solver_cfg,
                device=device,
            )
            _sync_device(device)
        for _ in range(int(TIMING_REPEATS)):
            _sync_device(device)
            start = time.perf_counter()
            _ = _run_multi_jump_rollout_batch(
                model=model,
                stats=stats,
                params_batch=params_batch[:sample_count],
                params_array=np.asarray(params_array[:sample_count], dtype=np.float64),
                initial_states_phys=np.asarray(initial_states_phys[:sample_count], dtype=np.float32),
                jump_schedule=jump_schedule,
                solver_cfg=solver_cfg,
                device=device,
            )
            _sync_device(device)
            times.append(time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times, ddof=0))


def _save_outputs(
    *,
    rows: List[Dict[str, float | int | str]],
    figure_path: Path,
    csv_path: Path,
    actual_input_day: float,
    actual_target_day: float,
    num_jumps: int,
) -> None:
    """Write benchmark CSV and plot."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device",
                "batch_size",
                "input_time_days",
                "target_time_days",
                "rollout_span_days",
                "num_jumps",
                "batch_latency_seconds",
                "batch_latency_std_seconds",
                "seconds_per_sample",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"Plot style not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=int(FIGURE_DPI), constrained_layout=True)

    device_names = list(dict.fromkeys(str(row["device"]) for row in rows))
    for device_name in device_names:
        device_rows = [row for row in rows if str(row["device"]) == device_name]
        device_rows.sort(key=lambda row: int(row["batch_size"]))
        ax.plot(
            [int(row["batch_size"]) for row in device_rows],
            [float(row["seconds_per_sample"]) for row in device_rows],
            color=DEVICE_COLORS.get(device_name, "#111827"),
            marker=DEVICE_MARKERS.get(device_name, "o"),
            linewidth=2.0,
            label=device_name,
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Rollout Time Per Sample (s)")
    ax.set_title(
        "Held-Out Test Rollout Latency\n"
        f"input_day={actual_input_day:.6f}, target_day={actual_target_day:.6f}, jumps={int(num_jumps)}"
    )
    ax.set_xticks(list(BATCH_SIZES))
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def main() -> None:
    """Run the held-out multi-jump rollout latency benchmark."""
    ckpt_path = _resolve_checkpoint_path(run_dir=RUN_DIR, checkpoint=CHECKPOINT_PATH)
    run_dir = ckpt_path.parent
    figure_path = (
        FIGURE_PATH.resolve()
        if FIGURE_PATH is not None
        else (run_dir / "plots" / FIGURE_NAME).resolve()
    )
    csv_path = (
        CSV_PATH.resolve()
        if CSV_PATH is not None
        else (run_dir / "plots" / CSV_NAME).resolve()
    )

    ensure_torch_harmonics_importable()
    ensure_my_swamp_importable()
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
    stats = stats_from_json(ckpt["normalization"])
    processed_dir = _resolve_processed_dir(ckpt)
    (
        params_batch,
        params_array,
        initial_states_phys,
        actual_input_day,
        actual_target_day,
        jump_schedule,
    ) = _load_test_rollout_batch(
        processed_dir=processed_dir,
        ckpt=ckpt,
        stats=stats,
        test_shard_start=int(TEST_SHARD_START),
        max_batch_size=int(max(BATCH_SIZES)),
        input_time_days=float(INPUT_TIME_DAYS),
        target_time_days=float(TARGET_TIME_DAYS),
        num_jumps=int(NUM_JUMPS),
    )

    rows: List[Dict[str, float | int | str]] = []
    for device in _resolve_devices(DEVICE_MODES):
        try:
            _configure_runtime(device)
            model = _build_model(ckpt, device=device)
            solver_cfg = dict(ckpt["solver"])

            for batch_size in BATCH_SIZES:
                mean_seconds, std_seconds = _benchmark_batch(
                    model=model,
                    stats=stats,
                    params_batch=params_batch,
                    params_array=params_array,
                    initial_states_phys=initial_states_phys,
                    jump_schedule=jump_schedule,
                    solver_cfg=solver_cfg,
                    batch_size=int(batch_size),
                    device=device,
                )
                rows.append(
                    {
                        "device": str(device.type),
                        "batch_size": int(batch_size),
                        "input_time_days": float(actual_input_day),
                        "target_time_days": float(actual_target_day),
                        "rollout_span_days": float(actual_target_day - actual_input_day),
                        "num_jumps": int(len(jump_schedule)),
                        "batch_latency_seconds": float(mean_seconds),
                        "batch_latency_std_seconds": float(std_seconds),
                        "seconds_per_sample": float(mean_seconds / float(batch_size)),
                    }
                )
        except Exception as exc:
            warnings.warn(f"Skipping {device.type} benchmark: {exc}")

    if not rows:
        raise RuntimeError("Benchmark did not produce any timing rows")

    _save_outputs(
        rows=rows,
        figure_path=figure_path,
        csv_path=csv_path,
        actual_input_day=actual_input_day,
        actual_target_day=actual_target_day,
        num_jumps=int(len(jump_schedule)),
    )
    print(f"Saved benchmark plot: {figure_path}")
    print(f"Saved benchmark CSV: {csv_path}")


if __name__ == "__main__":
    main()
