"""Command-line entry point for dataset generation and training."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import warnings


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    # Allow running ``python src/main.py`` without requiring editable install.
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config
from data_generation import generate_dataset
from my_swamp_backend import enforce_no_tpu_backend
from training import train_emulator


def _setup_logging() -> None:
    """Configure process-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _setup_warning_filters() -> None:
    """Suppress known benign warnings when configured."""
    if os.environ.get("GCMULATOR_SUPPRESS_KNOWN_WARNINGS", "1") != "1":
        return
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\],.*",
        category=UserWarning,
    )


def main() -> None:
    """Run the configured generation or training stage."""
    enforce_no_tpu_backend()
    _setup_warning_filters()
    _setup_logging()

    parser = argparse.ArgumentParser(description="Run GCMulator generation or training")
    stage_group = parser.add_mutually_exclusive_group(required=True)
    stage_group.add_argument(
        "--gen",
        action="store_true",
        help="Generate raw MY_SWAMP trajectory transitions",
    )
    stage_group.add_argument(
        "--train",
        action="store_true",
        help="Train the one-step transition emulator",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON/YAML")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    if args.gen:
        manifest = generate_dataset(cfg, config_path=config_path)
        print(json.dumps({"status": "ok", "n_sims_written": manifest["n_sims_written"]}, indent=2))
        return

    # Training reuses preprocessing internally, so there is only one explicit
    # train stage at the CLI level.
    summary = train_emulator(cfg, config_path=config_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
