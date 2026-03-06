"""Benchmark projected direct-jump rollout latency as a function of batch size and horizon."""

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
MPL_CACHE_DIR = Path(
    os.environ.get("GCMULATOR_MPLCONFIGDIR", str(DEFAULT_MPL_CACHE_DIR))
).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling import build_state_conditioned_transition_model, ensure_torch_harmonics_importable


BATCH_SIZES = (1, 4, 8)
ROLLOUT_DAYS = (1.0, 50.0, 100.0)
N_SAMPLES = max(BATCH_SIZES)
WARMUP_RUNS = 1
TIMING_REPEATS = 5
FIGURE_DPI = 180
FIGURE_NAME = "benchmark_rollout_days_vs_batch_size.png"
CSV_NAME = "benchmark_rollout_days_vs_batch_size.csv"
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")
DAY_COLORS = ("#1d4ed8", "#0f766e", "#b45309")
DEVICE_LINESTYLES = {
    "cpu": "-",
    "cuda": "--",
}
DEVICE_MARKERS = {
    "cpu": "o",
    "cuda": "s",
}

# User-editable run settings
RUN_NAME = "v1"
RUN_DIR: Path | None = (PROJECT_ROOT / "models" / RUN_NAME).resolve()
CHECKPOINT_PATH: Path | None = None
PROCESSED_DIR: Path | None = (PROJECT_ROOT / "data" / "processed").resolve()
SPLIT = "test"
DEVICE_MODES: Sequence[str] = ("cpu", "gpu")
FIGURE_PATH: Path | None = None
CSV_PATH: Path | None = None


def _dict_to_namespace(obj: Any) -> Any:
    """Convert nested dictionaries into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: _dict_to_namespace(value) for key, value in obj.items()}
        )
    if isinstance(obj, list):
        return [_dict_to_namespace(value) for value in obj]
    return obj


def _resolve_checkpoint_path(*, run_dir: Path | None, checkpoint: Path | None) -> Path:
    """Resolve checkpoint path from top-level run settings."""
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
    """Resolve benchmark device."""
    normalized = str(mode).lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested GPU benchmarking but CUDA is unavailable")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device mode: {mode}")


def _resolve_devices(modes: Sequence[str]) -> list[torch.device]:
    """Resolve requested benchmark devices, skipping unavailable GPU backends."""
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


def _resolve_processed_dir() -> Path:
    """Resolve processed directory from top-level settings."""
    if PROCESSED_DIR is None:
        raise ValueError("Set PROCESSED_DIR at the top of this file")
    resolved = PROCESSED_DIR.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {resolved}")
    return resolved


def _load_split_samples(
    processed_dir: Path,
    *,
    split: str,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load up to ``max_samples`` normalized examples from processed NPZ shards."""
    meta_path = (processed_dir / "processed_meta.json").resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"Processed metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split_entries = list(meta["splits"][split])
    if not split_entries:
        raise RuntimeError(f"Split '{split}' is empty")

    conditioning_rows: List[np.ndarray] = []
    state_input_rows: List[np.ndarray] = []
    transition_days_rows: List[np.ndarray] = []
    remaining = int(max_samples)

    for entry in split_entries:
        if remaining <= 0:
            break
        shard_path = (processed_dir / str(entry["file"])).resolve()
        with np.load(shard_path, allow_pickle=False) as npz:
            take = min(int(remaining), int(npz["state_inputs_norm"].shape[0]))
            conditioning = np.asarray(npz["conditioning_norm"][:take], dtype=np.float32)
            state_inputs = np.asarray(npz["state_inputs_norm"][:take], dtype=np.float32)
            transition_days = np.asarray(npz["transition_days"][:take], dtype=np.float64)
            conditioning_rows.append(conditioning)
            state_input_rows.append(state_inputs)
            transition_days_rows.append(transition_days)
        remaining -= take

    conditioning = torch.from_numpy(np.concatenate(conditioning_rows, axis=0))
    state_inputs = torch.from_numpy(np.concatenate(state_input_rows, axis=0))
    return conditioning, state_inputs, np.concatenate(transition_days_rows, axis=0)


def _sync_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator backends before/after timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _configure_runtime(device: torch.device) -> None:
    """Apply lightweight backend settings for repeatable inference benchmarking."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _step_count_for_days(days: float, *, step_days: float) -> int:
    """Convert rollout horizon in days to model-step count."""
    if days <= 0.0:
        raise ValueError(f"ROLLOUT_DAYS entries must be > 0, got {days}")
    if step_days <= 0.0:
        raise ValueError(f"Model step duration must be > 0, got {step_days}")
    return max(1, int(np.ceil(float(days) / float(step_days))))


def _device_label(device: torch.device) -> str:
    """Return a compact device label for plots and CSV."""
    return str(device.type)


def _benchmark_batch(
    *,
    model: torch.nn.Module,
    state_inputs: torch.Tensor,
    conditioning: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean and std direct-jump batch latency in seconds."""
    sample_states = state_inputs[: int(batch_size)]
    sample_conditioning = conditioning[: int(batch_size)]
    times: List[float] = []

    with torch.inference_mode():
        for _ in range(int(WARMUP_RUNS)):
            _ = model(sample_states, sample_conditioning)
            _sync_device(device)
        for _ in range(int(TIMING_REPEATS)):
            _sync_device(device)
            start = time.perf_counter()
            _ = model(sample_states, sample_conditioning)
            _sync_device(device)
            times.append(time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times, ddof=0))


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


def _save_outputs(
    *,
    rows: List[Dict[str, float | int | str]],
    figure_path: Path,
    csv_path: Path,
) -> None:
    """Write benchmark CSV and plot."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device",
                "rollout_days",
                "rollout_steps",
                "batch_size",
                "step_batch_latency_seconds",
                "step_batch_latency_std_seconds",
                "amortized_step_seconds_per_sample",
                "projected_rollout_seconds_per_sample",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=int(FIGURE_DPI), constrained_layout=True)

    color_by_days = {
        float(days): DAY_COLORS[index % len(DAY_COLORS)]
        for index, days in enumerate(ROLLOUT_DAYS)
    }
    device_names = list(dict.fromkeys(str(row["device"]) for row in rows))

    for device_name in device_names:
        for horizon_days in ROLLOUT_DAYS:
            curve_rows = [
                row
                for row in rows
                if str(row["device"]) == device_name
                and float(row["rollout_days"]) == float(horizon_days)
            ]
            if not curve_rows:
                continue
            curve_rows.sort(key=lambda row: int(row["batch_size"]))
            ax.plot(
                [int(row["batch_size"]) for row in curve_rows],
                [float(row["projected_rollout_seconds_per_sample"]) for row in curve_rows],
                color=color_by_days[float(horizon_days)],
                marker=DEVICE_MARKERS.get(device_name, "o"),
                linestyle=DEVICE_LINESTYLES.get(device_name, "-"),
                linewidth=2.0,
            )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Projected Rollout Time Per Sample (s)")
    ax.set_title("Projected Rollout Time vs Batch Size")
    ax.set_xticks(list(BATCH_SIZES))
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    horizon_handles = [
        Line2D(
            [0],
            [0],
            color=color_by_days[float(horizon_days)],
            linewidth=2.0,
            label=f"{float(horizon_days):g} day",
        )
        for horizon_days in ROLLOUT_DAYS
    ]
    device_handles = [
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=2.0,
            linestyle=DEVICE_LINESTYLES.get(device_name, "-"),
            marker=DEVICE_MARKERS.get(device_name, "o"),
            label=device_name,
        )
        for device_name in device_names
    ]
    legend_days = ax.legend(handles=horizon_handles, title="Horizon", loc="upper right")
    ax.add_artist(legend_days)
    ax.legend(handles=device_handles, title="Device", loc="lower left")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def main() -> None:
    """Run the projected rollout latency benchmark."""
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
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
    processed_dir = _resolve_processed_dir()
    conditioning_cpu, state_inputs_cpu, transition_days_cpu = _load_split_samples(
        processed_dir,
        split=SPLIT,
        max_samples=int(N_SAMPLES),
    )
    rollout_step_days = float(np.mean(transition_days_cpu))

    if int(conditioning_cpu.shape[0]) < max(BATCH_SIZES):
        raise ValueError(
            f"Need at least {max(BATCH_SIZES)} samples, "
            f"found {int(conditioning_cpu.shape[0])}"
        )

    rows: List[Dict[str, float | int | str]] = []
    for device in _resolve_devices(DEVICE_MODES):
        try:
            _configure_runtime(device)
            model = _build_model(ckpt, device=device)
            state_inputs = state_inputs_cpu.to(device=device)
            conditioning = conditioning_cpu.to(device=device)

            for batch_size in BATCH_SIZES:
                mean_seconds, std_seconds = _benchmark_batch(
                    model=model,
                    state_inputs=state_inputs,
                    conditioning=conditioning,
                    batch_size=int(batch_size),
                    device=device,
                )
                amortized_step_seconds = float(mean_seconds) / float(batch_size)
                for horizon_days in ROLLOUT_DAYS:
                    rollout_steps = _step_count_for_days(
                        float(horizon_days),
                        step_days=float(rollout_step_days),
                    )
                    rows.append(
                        {
                            "device": _device_label(device),
                            "rollout_days": float(horizon_days),
                            "rollout_steps": int(rollout_steps),
                            "batch_size": int(batch_size),
                            "step_batch_latency_seconds": float(mean_seconds),
                            "step_batch_latency_std_seconds": float(std_seconds),
                            "amortized_step_seconds_per_sample": float(
                                amortized_step_seconds
                            ),
                            "projected_rollout_seconds_per_sample": float(
                                amortized_step_seconds * float(rollout_steps)
                            ),
                        }
                    )
        except Exception as exc:
            warnings.warn(f"Skipping {device.type} benchmark: {exc}")

    if not rows:
        raise RuntimeError("Benchmark did not produce any timing rows")

    _save_outputs(rows=rows, figure_path=figure_path, csv_path=csv_path)
    print(f"Saved benchmark plot: {figure_path}")
    print(f"Saved benchmark CSV: {csv_path}")


if __name__ == "__main__":
    main()
