"""Plot train/validation loss curves from one GCMulator run directory."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", "/tmp/gcmulator_mplcache")).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt


HISTORY_CSV_NAME = "training_history.csv"
PLOTS_DIR_NAME = "plots"
FIGURE_NAME = "val_vs_train_loss_by_epoch.png"
FIGURE_DPI = 180
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot train-vs-val loss from a GCMulator run")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing training_history.csv",
    )
    parser.add_argument("--figure", type=Path, default=None, help="Explicit output figure path")
    return parser.parse_args()


def _apply_plot_style() -> None:
    """Load shared matplotlib styling."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"Plot style not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)


def _read_history_csv(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Read epoch, train loss, and validation loss columns."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"epoch", "train_loss", "val_loss"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise KeyError(f"{csv_path} is missing required columns: {missing}")

        epochs: list[int] = []
        train_losses: list[float] = []
        val_losses: list[float] = []
        for row in reader:
            epochs.append(int(round(float(row["epoch"]))))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not epochs:
        raise ValueError(f"No rows found in {csv_path}")
    return epochs, train_losses, val_losses


def _save_loss_figure(
    *,
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    out_path: Path,
) -> None:
    """Render and save loss curves."""
    fig, ax = plt.subplots(
        figsize=(6.5, 6.5),
        dpi=int(FIGURE_DPI),
        constrained_layout=True,
    )
    ax.plot(
        epochs,
        train_losses,
        color="#d62728",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        label="Train",
    )
    ax.plot(
        epochs,
        val_losses,
        color="#1f77b4",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        label="Val",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    # A square plotting area makes train/validation divergence visually easier
    # to compare across runs with different epoch counts.
    ax.set_box_aspect(1.0)
    ax.set_title("Training vs Validation Loss by Epoch")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Load a run's history and save the loss plot."""
    _apply_plot_style()
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    history_csv_path = (run_dir / HISTORY_CSV_NAME).resolve()
    if not history_csv_path.is_file():
        raise FileNotFoundError(f"Training history CSV not found: {history_csv_path}")

    figure_path = (
        args.figure.resolve()
        if args.figure is not None
        else (run_dir / PLOTS_DIR_NAME / FIGURE_NAME).resolve()
    )
    epochs, train_losses, val_losses = _read_history_csv(history_csv_path)
    _save_loss_figure(
        epochs=epochs,
        train_losses=train_losses,
        val_losses=val_losses,
        out_path=figure_path,
    )
    print(f"Saved training history figure: {figure_path}")


if __name__ == "__main__":
    main()
