"""Training log-format regression tests."""

from __future__ import annotations

import csv
from pathlib import Path

from gcmulator.training import _format_scientific, _write_training_history_csv


def test_format_scientific_uses_fixed_scientific_notation() -> None:
    """Human-facing training diagnostics should use fixed scientific notation."""
    assert _format_scientific(0.0) == "0.000e+00"
    assert _format_scientific(1.23456e-3) == "1.235e-03"
    assert _format_scientific(12.3456) == "1.235e+01"


def test_write_training_history_csv_formats_metric_columns_scientifically(
    tmp_path: Path,
) -> None:
    """CSV history should preserve raw columns while formatting diagnostics consistently."""
    csv_path = tmp_path / "training_history.csv"
    _write_training_history_csv(
        history=[
            {
                "epoch": 3.0,
                "train_loss": 1.23456e-3,
                "val_loss": 9.87654e-4,
                "lr": 3.0e-4,
                "train_seconds": 12.3456,
                "val_seconds": 2.5,
                "epoch_seconds": 14.8456,
                "train_samples": 128.0,
                "val_samples": 32.0,
                "train_samples_per_second": 10.368,
                "val_samples_per_second": 12.8,
            }
        ],
        csv_path=csv_path,
    )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == "3"
    assert row["train_loss"] == "1.235e-03"
    assert row["val_loss"] == "9.877e-04"
    assert row["lr"] == "3.000e-04"
    assert row["train_seconds"] == "1.235e+01"
    assert row["val_seconds"] == "2.500e+00"
    assert row["epoch_seconds"] == "1.485e+01"
    assert row["train_samples"] == "1.280e+02"
    assert row["val_samples"] == "3.200e+01"
    assert row["train_samples_per_second"] == "1.037e+01"
    assert row["val_samples_per_second"] == "1.280e+01"
