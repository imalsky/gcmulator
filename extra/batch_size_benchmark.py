"""Benchmark held-out direct-jump inference latency on the test split."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Sequence
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

from gcmulator.config import resolve_path
from gcmulator.modeling import (
    build_state_conditioned_transition_model,
    ensure_torch_harmonics_importable,
)
from gcmulator.normalization import stats_from_json
from gcmulator.training import (
    _catalog_to_device,
    _iter_live_pair_batches,
    _live_transition_catalog_from_json,
    _load_sequence_split_to_device,
    _release_preloaded_sequence_split,
    _sample_live_pair_table,
)


BATCH_SIZES = (1, 2, 4)
TEST_SHARD_INDEX = 0
WARMUP_RUNS = 1
TIMING_REPEATS = 5
FIGURE_DPI = 180
FIGURE_NAME = "benchmark_test_direct_jump_latency_vs_batch_size.png"
CSV_NAME = "benchmark_test_direct_jump_latency_vs_batch_size.csv"
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


def _load_direct_jump_batch(
    *,
    processed_dir: Path,
    test_shard_index: int,
    max_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one normalized held-out batch sampled live from a test sequence.

    Returns:
        conditioning_norm:
            Conditioning matrix with shape ``[N, P]`` where ``P`` includes the
            physical parameters plus ``log10_transition_days``.
        state_inputs_norm:
            Normalized prognostic states with shape ``[N, 3, H, W]``.
        shard_name:
            Processed shard filename used for the benchmark.
    """
    test_entries = _load_test_split_entries(processed_dir)
    if test_shard_index < 0 or test_shard_index >= len(test_entries):
        raise IndexError(
            f"TEST_SHARD_INDEX={test_shard_index} is out of range for {len(test_entries)} test shards"
        )

    shard_entry = dict(test_entries[test_shard_index])
    shard_name = str(shard_entry["file"])
    shard_path = (processed_dir / shard_name).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(f"Processed shard not found: {shard_path}")

    meta_path = (processed_dir / "processed_meta.json").resolve()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = stats_from_json(meta["normalization"])
    split = _load_sequence_split_to_device(
        processed_dir=processed_dir,
        shard_entries=[shard_entry],
        device=torch.device("cpu"),
    )
    catalog = _catalog_to_device(
        catalog=_live_transition_catalog_from_json(meta["live_transition_catalog"]),
        transition_time_stats=stats.transition_time,
        device=torch.device("cpu"),
    )
    pair_table = _sample_live_pair_table(
        split=split,
        catalog=catalog,
        live_pairs_per_sequence=int(meta["sampling"]["live_pairs_per_sequence"]),
        seed=300_000,
        shuffle_pairs=False,
    )
    available = int(pair_table.n_pairs)
    if available < int(max_batch_size):
        split = _release_preloaded_sequence_split(split)
        raise ValueError(
            f"Shard {shard_name} has only {available} live-sampled pairs, but "
            f"batch size {max_batch_size} was requested"
        )
    conditioning_batch, state_input_batch, _ = next(
        _iter_live_pair_batches(
            split=split,
            pair_table=pair_table,
            batch_size=int(max_batch_size),
        )
    )
    conditioning_norm = conditioning_batch.detach().cpu().numpy()
    state_inputs_norm = state_input_batch.detach().cpu().numpy()
    split = _release_preloaded_sequence_split(split)
    return (
        conditioning_norm[: int(max_batch_size)],
        state_inputs_norm[: int(max_batch_size)],
        shard_name,
    )


def _sync_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator work before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _benchmark_batch_size(
    *,
    model: torch.nn.Module,
    conditioning_norm: np.ndarray,
    state_inputs_norm: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean and std direct-jump latency for one batch size."""
    conditioning_batch = torch.from_numpy(
        np.ascontiguousarray(conditioning_norm[:batch_size])
    ).to(device=device)
    state_batch = torch.from_numpy(
        np.ascontiguousarray(state_inputs_norm[:batch_size])
    ).to(device=device)

    with torch.inference_mode():
        for _ in range(int(WARMUP_RUNS)):
            _ = model(state_batch, conditioning_batch)
            _sync_device(device)

        durations: list[float] = []
        for _ in range(int(TIMING_REPEATS)):
            start = time.perf_counter()
            _ = model(state_batch, conditioning_batch)
            _sync_device(device)
            durations.append(time.perf_counter() - start)

    mean_seconds = float(np.mean(durations))
    std_seconds = float(np.std(durations, ddof=0))
    return mean_seconds, std_seconds


def _apply_plot_style() -> None:
    """Load the shared plotting style."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"Plot style not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)


def _write_results_csv(*, rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    """Write benchmark results to CSV."""
    field_names = [
        "device",
        "batch_size",
        "mean_seconds",
        "std_seconds",
        "samples_per_second",
        "shard_name",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_figure(*, rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    """Save latency-vs-batch-size figure for each benchmark device."""
    fig, ax = plt.subplots(
        figsize=(6.8, 5.2),
        dpi=int(FIGURE_DPI),
        constrained_layout=True,
    )

    for device_name in sorted({str(row["device"]) for row in rows}):
        device_rows = [row for row in rows if str(row["device"]) == device_name]
        batch_sizes = [int(row["batch_size"]) for row in device_rows]
        mean_seconds = [float(row["mean_seconds"]) for row in device_rows]
        std_seconds = [float(row["std_seconds"]) for row in device_rows]
        ax.errorbar(
            batch_sizes,
            mean_seconds,
            yerr=std_seconds,
            color=DEVICE_COLORS.get(device_name, "#374151"),
            marker=DEVICE_MARKERS.get(device_name, "o"),
            linewidth=2.0,
            markersize=5.0,
            capsize=4.0,
            label=device_name.upper(),
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Held-Out Direct-Jump Inference Latency")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Benchmark held-out direct-jump inference latency across batch sizes."""
    _apply_plot_style()
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
    processed_dir = _resolve_processed_dir(ckpt)
    max_batch_size = max(int(batch_size) for batch_size in BATCH_SIZES)
    conditioning_norm, state_inputs_norm, shard_name = _load_direct_jump_batch(
        processed_dir=processed_dir,
        test_shard_index=int(TEST_SHARD_INDEX),
        max_batch_size=max_batch_size,
    )

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    geometry = dict(ckpt.get("geometry", {}))
    state_fields = list(ckpt["state_fields"])
    residual_input_indices = list(range(len(state_fields)))
    devices = _resolve_devices(DEVICE_MODES)

    rows: list[Dict[str, Any]] = []
    for device in devices:
        model = build_state_conditioned_transition_model(
            img_size=(int(shape["H"]), int(shape["W"])),
            input_state_chans=int(shape["C"]),
            target_state_chans=int(shape["C"]),
            param_dim=int(len(ckpt["conditioning_names"])),
            residual_input_indices=residual_input_indices,
            cfg_model=model_cfg,
            lat_order=str(geometry.get("lat_order", "north_to_south")),
                lon_origin=str(geometry.get("lon_origin", "0_to_2pi")),
        )
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device=device).eval()

        for batch_size in BATCH_SIZES:
            mean_seconds, std_seconds = _benchmark_batch_size(
                model=model,
                conditioning_norm=conditioning_norm,
                state_inputs_norm=state_inputs_norm,
                batch_size=int(batch_size),
                device=device,
            )
            rows.append(
                {
                    "device": device.type,
                    "batch_size": int(batch_size),
                    "mean_seconds": mean_seconds,
                    "std_seconds": std_seconds,
                    "samples_per_second": float(batch_size) / max(mean_seconds, 1.0e-12),
                    "shard_name": shard_name,
                }
            )

    _write_results_csv(rows=rows, csv_path=csv_path)
    _save_figure(rows=rows, out_path=figure_path)
    print(f"Saved direct-jump latency CSV: {csv_path}")
    print(f"Saved direct-jump latency figure: {figure_path}")


if __name__ == "__main__":
    main()
