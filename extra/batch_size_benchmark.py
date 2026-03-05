"""Simple local benchmark: amortized model time vs batch size."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

MPL_CACHE_DIR = Path(
    os.environ.get("GCMULATOR_MPLCONFIGDIR", "/tmp/gcmulator_mplcache")
).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import build_state_conditioned_rollout_model, ensure_torch_harmonics_importable

# Fixed local benchmark config.
MODEL_DIR = PROJECT_ROOT / "models/model"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
CHECKPOINT_PATH = MODEL_DIR / "best.pt"
PROCESSED_META_PATH = PROCESSED_DIR / "processed_meta.json"
SPLIT_NAME = "test"

BATCH_SIZES = (1, 4, 16, 64)
N_SAMPLES = 64
ROLLOUT_STEPS = (1,)
WARMUP_RUNS = 0
TIMING_REPEATS = 1
QUICK_MSE_SAMPLES = 1

PLOTS_DIR = MODEL_DIR / "plots"
FIGURE_PATH = PLOTS_DIR / "benchmark_batch_size_vs_latency.png"
CSV_PATH = PLOTS_DIR / "benchmark_batch_size_vs_latency.csv"
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")
FIGURE_DPI = 180


def to_namespace(obj: Any) -> Any:
    """Convert nested dict/list values into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def load_payload(path: Path) -> Dict[str, Any]:
    """Load one processed payload from .npy dict format."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        return obj.item()
    return obj


def sync_if_cuda(device: torch.device) -> None:
    """Sync CUDA for accurate elapsed time measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def quick_mse(
    model: torch.nn.Module,
    state_inputs: torch.Tensor,
    conditioning: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
) -> float:
    """Small MSE estimate for legend labels."""
    bsz = int(QUICK_MSE_SAMPLES)
    with torch.inference_mode():
        pred = model(state_inputs[:bsz], conditioning[:bsz], steps=int(steps))
    return float(torch.mean((pred - targets[:bsz]) ** 2).item())


def benchmark_batch(
    model: torch.nn.Module,
    state_inputs: torch.Tensor,
    conditioning: torch.Tensor,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean_batch_time, std_batch_time) in seconds."""
    s = state_inputs[: int(batch_size)]
    x = conditioning[: int(batch_size)]
    times: List[float] = []

    with torch.inference_mode():
        for _ in range(int(WARMUP_RUNS)):
            _ = model(s, x, steps=int(steps))
            sync_if_cuda(device)

        for _ in range(int(TIMING_REPEATS)):
            sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = model(s, x, steps=int(steps))
            sync_if_cuda(device)
            times.append(time.perf_counter() - t0)

    return float(np.mean(times)), float(np.std(times, ddof=0))


def main() -> None:
    """Run CPU benchmark and optional CUDA benchmark."""
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    ckpt: Dict[str, Any] = torch.load(CHECKPOINT_PATH, map_location="cpu")
    source_cfg_dir = Path(str(ckpt["source_config_path"])).expanduser().resolve().parent
    ensure_torch_harmonics_importable(source_cfg_dir)

    meta = json.loads(PROCESSED_META_PATH.read_text(encoding="utf-8"))
    split_files = list(meta["splits"][SPLIT_NAME])[: int(N_SAMPLES)]

    conditioning_rows: List[np.ndarray] = []
    state_input_rows: List[np.ndarray] = []
    state_target_rows: List[np.ndarray] = []
    for name in tqdm(split_files, desc="loading samples"):
        payload = load_payload(PROCESSED_DIR / str(name))
        if "conditioning_norm" in payload:
            conditioning_rows.append(np.asarray(payload["conditioning_norm"], dtype=np.float32))
        else:
            conditioning_rows.append(np.asarray(payload["params_norm"], dtype=np.float32))
        state_input_rows.append(np.asarray(payload["state_input_norm"], dtype=np.float32))
        state_target_rows.append(np.asarray(payload["state_target_norm"], dtype=np.float32))

    conditioning_cpu = torch.from_numpy(np.stack(conditioning_rows, axis=0))
    state_inputs_cpu = torch.from_numpy(np.stack(state_input_rows, axis=0))
    targets_cpu = torch.from_numpy(np.stack(state_target_rows, axis=0))

    # Script is intentionally simple: assumes enough rows for requested batches.
    assert int(conditioning_cpu.shape[0]) >= max(BATCH_SIZES)

    rollout_steps = [int(s) for s in ROLLOUT_STEPS]

    model_cfg = to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    geom = dict(ckpt.get("geometry", {}))
    conditioning_names = list(ckpt.get("conditioning_names", ckpt["param_names"]))
    model = build_state_conditioned_rollout_model(
        img_size=(int(shape["H"]), int(shape["W"])),
        state_chans=int(shape["C"]),
        param_dim=int(len(conditioning_names)),
        cfg_model=model_cfg,
        lat_order=str(geom.get("lat_order", "north_to_south")),
        lon_origin=str(geom.get("lon_origin", "0_to_2pi")),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devices: List[torch.device] = [run_device]

    rows: List[Dict[str, float | int | str]] = []
    curves: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for device in tqdm(devices, desc="devices"):
        model_device = model.to(device=device).eval()
        state_inputs_device = state_inputs_cpu.to(device=device)
        conditioning_device = conditioning_cpu.to(device=device)
        targets_device = targets_cpu.to(device=device)

        step_curves: Dict[int, Dict[str, Any]] = {}
        for steps in tqdm(rollout_steps, desc=f"{device} rollout", leave=False):
            mse = quick_mse(model_device, state_inputs_device, conditioning_device, targets_device, int(steps))

            amortized: List[float] = []
            for bsz in tqdm(BATCH_SIZES, desc=f"{device} steps={steps}", leave=False):
                mean_batch, std_batch = benchmark_batch(
                    model_device,
                    state_inputs_device,
                    conditioning_device,
                    int(steps),
                    int(bsz),
                    device,
                )
                amort = mean_batch / float(bsz)
                amortized.append(amort)
                rows.append(
                    {
                        "device": str(device),
                        "rollout_steps": int(steps),
                        "quick_mse": float(mse),
                        "batch_size": int(bsz),
                        "mean_batch_latency_seconds": float(mean_batch),
                        "std_batch_latency_seconds": float(std_batch),
                        "amortized_time_seconds": float(amort),
                    }
                )

            step_curves[int(steps)] = {
                "batch_sizes": list(BATCH_SIZES),
                "amortized_time_seconds": amortized,
                "quick_mse": float(mse),
            }
        curves[str(device)] = step_curves

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "rollout_steps",
                "quick_mse",
                "batch_size",
                "mean_batch_latency_seconds",
                "std_batch_latency_seconds",
                "amortized_time_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    fig, axes = plt.subplots(
        1,
        len(curves),
        figsize=(6.2 * len(curves), 5.0),
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    for i, (device_name, step_map) in enumerate(curves.items()):
        ax = axes[0, i]
        for j, (steps, line) in enumerate(step_map.items()):
            label = f"steps={steps} | mse={line['quick_mse']:.3e}"
            ax.plot(
                line["batch_sizes"],
                line["amortized_time_seconds"],
                marker="o",
                color=colors[j % len(colors)],
                label=label,
            )
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Amortized time (s) = batch_time / batch_size")
        ax.set_title(f"{device_name.upper()} inference")
        ax.margins(x=0.15)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Amortized model time vs batch size")
    fig.savefig(FIGURE_PATH)
    plt.close(fig)

    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Devices: {[str(d) for d in devices]} (MPS excluded)")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Batch sizes: {list(BATCH_SIZES)}")


if __name__ == "__main__":
    main()
