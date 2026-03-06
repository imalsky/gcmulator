"""Benchmark one-step inference latency as a function of batch size."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List

MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", "/tmp/gcmulator_mplcache")).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import resolve_path
from modeling import build_state_conditioned_transition_model, ensure_torch_harmonics_importable


BATCH_SIZES = (1, 4, 16, 64)
N_SAMPLES = 64
WARMUP_RUNS = 0
TIMING_REPEATS = 1
FIGURE_DPI = 180
FIGURE_NAME = "benchmark_batch_size_vs_latency.png"
CSV_NAME = "benchmark_batch_size_vs_latency.csv"
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark GCMulator one-step inference"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory containing best.pt")
    source.add_argument("--checkpoint", type=Path, help="Checkpoint path to benchmark")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Processed split to sample",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Benchmark device",
    )
    parser.add_argument("--figure", type=Path, default=None, help="Explicit output plot path")
    parser.add_argument("--csv", type=Path, default=None, help="Explicit CSV output path")
    return parser.parse_args()


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
    """Resolve checkpoint path from CLI inputs."""
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    if run_dir is None:
        raise ValueError("Either --run-dir or --checkpoint must be provided")
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
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=gpu but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device mode: {mode}")


def _processed_dir_from_checkpoint(ckpt: Dict[str, Any]) -> Path:
    """Resolve processed directory using the checkpoint's saved config."""
    resolved_config = dict(ckpt["resolved_config"])
    source_config_path = Path(str(ckpt["source_config_path"])).resolve()
    return resolve_path(source_config_path, str(resolved_config["paths"]["processed_dir"]))


def _load_split_samples(
    processed_dir: Path,
    *,
    split: str,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    state_target_rows: List[np.ndarray] = []
    remaining = int(max_samples)

    for entry in split_entries:
        if remaining <= 0:
            break
        shard_path = (processed_dir / str(entry["file"])).resolve()
        with np.load(shard_path, allow_pickle=False) as npz:
            take = min(int(remaining), int(npz["state_inputs_norm"].shape[0]))
            params_norm = np.asarray(npz["params_norm"], dtype=np.float32)
            state_inputs = np.asarray(npz["state_inputs_norm"][:take], dtype=np.float32)
            conditioning_rows.append(
                np.repeat(
                    params_norm[None, :],
                    int(state_inputs.shape[0]),
                    axis=0,
                )
            )
            state_input_rows.append(state_inputs)
            state_target_rows.append(
                np.asarray(npz["state_targets_norm"][:take], dtype=np.float32)
            )
        remaining -= take

    conditioning = torch.from_numpy(np.concatenate(conditioning_rows, axis=0))
    state_inputs = torch.from_numpy(np.concatenate(state_input_rows, axis=0))
    targets = torch.from_numpy(np.concatenate(state_target_rows, axis=0))
    return conditioning, state_inputs, targets


def _sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA device before/after timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _benchmark_batch(
    *,
    model: torch.nn.Module,
    state_inputs: torch.Tensor,
    conditioning: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean and std batch latency in seconds."""
    sample_states = state_inputs[: int(batch_size)]
    sample_conditioning = conditioning[: int(batch_size)]
    times: List[float] = []

    with torch.inference_mode():
        for _ in range(int(WARMUP_RUNS)):
            _ = model(sample_states, sample_conditioning)
            _sync_if_cuda(device)
        for _ in range(int(TIMING_REPEATS)):
            _sync_if_cuda(device)
            start = time.perf_counter()
            _ = model(sample_states, sample_conditioning)
            _sync_if_cuda(device)
            times.append(time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times, ddof=0))


def _quick_mse(
    *,
    model: torch.nn.Module,
    state_inputs: torch.Tensor,
    conditioning: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute a small normalized-space MSE estimate."""
    sample_count = min(4, int(state_inputs.shape[0]))
    with torch.inference_mode():
        pred = model(state_inputs[:sample_count], conditioning[:sample_count])
    return float(torch.mean((pred - targets[:sample_count]) ** 2).item())


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
                "batch_size",
                "latency_seconds",
                "latency_std_seconds",
                "quick_mse",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=int(FIGURE_DPI), constrained_layout=True)
    device_names = sorted({str(row["device"]) for row in rows})
    for device_name in device_names:
        device_rows = [row for row in rows if str(row["device"]) == device_name]
        device_rows.sort(key=lambda row: int(row["batch_size"]))
        ax.plot(
            [int(row["batch_size"]) for row in device_rows],
            [float(row["latency_seconds"]) for row in device_rows],
            marker="o",
            linewidth=2.0,
            label=device_name,
        )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (s)")
    ax.set_title("One-Step Inference Latency vs Batch Size")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def main() -> None:
    """Run the latency benchmark."""
    args = _parse_args()
    ckpt_path = _resolve_checkpoint_path(run_dir=args.run_dir, checkpoint=args.checkpoint)
    run_dir = ckpt_path.parent
    figure_path = (
        args.figure.resolve()
        if args.figure is not None
        else (run_dir / "plots" / FIGURE_NAME).resolve()
    )
    csv_path = (
        args.csv.resolve()
        if args.csv is not None
        else (run_dir / "plots" / CSV_NAME).resolve()
    )

    device = _resolve_device(args.device)
    ensure_torch_harmonics_importable()
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
    processed_dir = _processed_dir_from_checkpoint(ckpt)
    conditioning_cpu, state_inputs_cpu, targets_cpu = _load_split_samples(
        processed_dir,
        split=args.split,
        max_samples=int(N_SAMPLES),
    )

    if int(conditioning_cpu.shape[0]) < max(BATCH_SIZES):
        raise ValueError(
            f"Need at least {max(BATCH_SIZES)} samples, "
            f"found {int(conditioning_cpu.shape[0])}"
        )

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    input_fields = list(ckpt["input_fields"])
    target_fields = list(ckpt["target_fields"])
    residual_input_indices = [input_fields.index(field_name) for field_name in target_fields]
    model = build_state_conditioned_transition_model(
        img_size=(int(shape["H"]), int(shape["W"])),
        input_state_chans=int(shape["input_C"]),
        target_state_chans=int(shape["target_C"]),
        param_dim=int(len(ckpt["param_names"])),
        residual_input_indices=residual_input_indices,
        cfg_model=model_cfg,
        lat_order="north_to_south",
        lon_origin="0_to_2pi",
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device=device).eval()

    state_inputs = state_inputs_cpu.to(device=device)
    conditioning = conditioning_cpu.to(device=device)
    targets = targets_cpu.to(device=device)
    quick_mse = _quick_mse(
        model=model,
        state_inputs=state_inputs,
        conditioning=conditioning,
        targets=targets,
    )

    rows: List[Dict[str, float | int | str]] = []
    for batch_size in BATCH_SIZES:
        mean_seconds, std_seconds = _benchmark_batch(
            model=model,
            state_inputs=state_inputs,
            conditioning=conditioning,
            batch_size=int(batch_size),
            device=device,
        )
        rows.append(
            {
                "device": str(device),
                "batch_size": int(batch_size),
                "latency_seconds": float(mean_seconds),
                "latency_std_seconds": float(std_seconds),
                "quick_mse": float(quick_mse),
            }
        )

    _save_outputs(rows=rows, figure_path=figure_path, csv_path=csv_path)
    print(f"Saved benchmark plot: {figure_path}")
    print(f"Saved benchmark CSV: {csv_path}")


if __name__ == "__main__":
    main()
