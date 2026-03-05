"""CLI entrypoint for dataset generation and emulator training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_generation import generate_dataset
from src.my_swamp_backend import enforce_no_tpu_backend
from src.training import train_emulator


def _setup_logging() -> None:
    """Configure process-wide logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _setup_warning_filters() -> None:
    """Silence known benign warnings when suppression is enabled."""
    if os.environ.get("GCMULATOR_SUPPRESS_KNOWN_WARNINGS", "1") != "1":
        return

    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\],.*",
        category=UserWarning,
    )


def _resolve_config_path(config_arg: str | None) -> Path:
    """Resolve CLI config path or fall back to repository ``config.json``."""
    if config_arg is None:
        return (PROJECT_ROOT / "config.json").resolve()
    return Path(config_arg).resolve()


def main() -> None:
    """Execute one CLI mode: generate dataset or train emulator."""
    enforce_no_tpu_backend()
    _setup_warning_filters()

    parser = argparse.ArgumentParser(description="Run GCMulator data generation or training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gen", action="store_true", help="Generate MY_SWAMP trajectory-transition dataset")
    group.add_argument("--train", action="store_true", help="Train emulator from generated dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML (default: ./config.json)")
    args = parser.parse_args()

    _setup_logging()

    config_path = _resolve_config_path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)

    if args.gen:
        manifest = generate_dataset(cfg, config_path=config_path)
        print(json.dumps({"status": "ok", "n_sims_written": manifest["n_sims_written"]}, indent=2))
        return

    summary = train_emulator(cfg, config_path=config_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
