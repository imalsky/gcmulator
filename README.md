# GCMulator

Core emulator package for MY_SWAMP terminal-state data generation and SFNO training.

## Structure

```
GCMulator/
  src/
    __init__.py
    config.py
    constants.py
    data_generation.py
    geometry.py
    modeling.py
    my_swamp_backend.py
    normalization.py
    sampling.py
    training.py
    main.py
  tests/
  config.json
  pyproject.toml
```

All runtime code lives directly under `src/`.

## Scope

This package intentionally contains only:

1. Config loading/validation
2. Dataset generation from `MY_SWAMP`
3. Training/preprocessing for SFNO terminal-state emulation

Inference/export/benchmark scripts are intentionally excluded from this package and live in `export_benchmark/`.
JAX TPU probing is disabled by policy in runtime setup (`my_swamp_backend.enforce_no_tpu_backend()`).
By default, JAX backend selection is left to JAX auto-detection (while excluding TPU), so GPU is used automatically when available.

## GPU throughput note

For dataset generation, `sampling.generation_workers` controls threaded parallel simulation:

1. `0` (default): auto (`4` workers on GPU backend, otherwise `1`)
2. `>=1`: use exactly that many workers (capped by `n_sims`)

## Install

```bash
cd GCMulator
pip install -e .
```

## Simple commands

Generate data:

```bash
python src/main.py --gen
```

Train:

```bash
python src/main.py --train
```

Training writes config provenance into `model/`:

1. `config_used.resolved.json`
2. `config_used.original.<ext>`
3. checkpoint keys `resolved_config` and `source_config_path`

Optional custom config path:

```bash
python src/main.py --gen --config /absolute/path/to/config.json
```

## Use as a library

```python
from pathlib import Path

from src.config import load_config
from src.data_generation import generate_dataset
from src.training import train_emulator

config_path = Path("config.json").resolve()
cfg = load_config(config_path)

manifest = generate_dataset(cfg, config_path=config_path)
summary = train_emulator(cfg, config_path=config_path)
```

## Tests

```bash
cd GCMulator
pytest -q
```
