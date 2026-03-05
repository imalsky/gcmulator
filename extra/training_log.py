"""Utility script to plot validation vs test loss over training epochs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
from typing import List, Tuple

# Keep matplotlib cache in a writable temp path to avoid permission/cache issues.
MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", "/tmp/gcmulator_mplcache")).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("training_log.py requires matplotlib. Install it in your environment first.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Global configuration.
MODEL_DIR = Path("models/model")
HISTORY_CSV_NAME = "training_history.csv"
PLOTS_DIR_NAME = "plots"
FIGURE_NAME = "val_vs_test_loss_by_epoch.png"
FIGURE_DPI = 180
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")


def _resolve_model_dir(path_value: Path) -> Path:
    """Resolve model directory from absolute, repo-relative, or cwd-relative path."""
    if path_value.is_absolute():
        return path_value.resolve()

    # Default behavior: treat relative paths as repo-root relative.
    project_candidate = (PROJECT_ROOT / path_value).resolve()
    if project_candidate.exists():
        return project_candidate

    # Fallback: allow cwd-relative custom paths if the user set one.
    cwd_candidate = path_value.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return project_candidate


def _apply_plot_style() -> None:
    """Load shared matplotlib style and configure figure DPI."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"science.mplstyle not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)


def _read_history_csv(csv_path: Path) -> Tuple[List[int], List[float], List[float], str]:
    """Read epoch and loss columns from training history CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "epoch" not in fieldnames or "val_loss" not in fieldnames:
            raise KeyError(f"{csv_path} must include epoch and val_loss columns, found: {fieldnames}")

        compare_key = "train_loss" if "train_loss" in fieldnames else "test_loss"
        if compare_key not in fieldnames:
            raise KeyError(
                f"{csv_path} must include test_loss (preferred) or train_loss columns, found: {fieldnames}"
            )

        epochs: List[int] = []
        val_losses: List[float] = []
        compare_losses: List[float] = []
        for row in reader:
            if row.get("epoch") is None:
                continue
            epochs.append(int(round(float(row["epoch"]))))
            val_losses.append(float(row["val_loss"]))
            compare_losses.append(float(row[compare_key]))

    if not epochs:
        raise ValueError(f"No rows found in {csv_path}")
    return epochs, val_losses, compare_losses, compare_key


def _save_loss_figure(
    *,
    epochs: List[int],
    val_losses: List[float],
    compare_losses: List[float],
    compare_label: str,
    out_path: Path,
) -> None:
    """Render and save loss-vs-epoch line plot."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=int(FIGURE_DPI), constrained_layout=True)
    ax.plot(epochs, val_losses, color="#1f77b4", linewidth=2.0, marker="o", markersize=3.0, label="Val")
    ax.plot(
        epochs,
        compare_losses,
        color="#d62728",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        label=compare_label,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_box_aspect(1.0)
    ax.set_title("Validation vs Test Loss by Epoch")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Load training history CSV and save loss-vs-epoch plot."""
    _apply_plot_style()
    model_dir = _resolve_model_dir(MODEL_DIR)
    history_csv_path = (model_dir / HISTORY_CSV_NAME).resolve()
    if not history_csv_path.is_file():
        raise FileNotFoundError(f"Training history CSV not found: {history_csv_path}")

    epochs, val_losses, compare_losses, compare_key = _read_history_csv(history_csv_path)
    if compare_key == "train_loss":
        compare_label = "Train"
    else:
        compare_label = "Test"

    out_path = (model_dir / PLOTS_DIR_NAME / FIGURE_NAME).resolve()
    _save_loss_figure(
        epochs=epochs,
        val_losses=val_losses,
        compare_losses=compare_losses,
        compare_label=compare_label,
        out_path=out_path,
    )

    print(f"Saved figure: {out_path}")
    print(f"Read history: {history_csv_path}")
    print(f"Comparison column: {compare_key}")


if __name__ == "__main__":
    main()
