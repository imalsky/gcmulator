# GCMulator Specification

## 1. Purpose
GCMulator is a personal scientific software project for learning an emulator of terminal shallow-water states produced by MY_SWAMP.

The current scope is intentionally narrow:
1. Validate and load experiment configuration.
2. Generate terminal-state datasets from MY_SWAMP truth runs.
3. Preprocess and train a conditional SFNO rollout emulator.
4. Run post-training utilities for prediction visualization and TorchScript export.

This repository is not treated as a Python package. The main runtime entrypoint is `src/main.py`.

## 2. Repository Structure
- `src/`: runtime code for config, generation, preprocessing, model definition, and training.
- `extra/`: utility scripts for prediction figure generation, TorchScript export, and SWAMPE parity checks.
- `config.json`: default experiment configuration.
- `run.sh`: Slurm-oriented convenience script for generation/training.
- `run.pbs`: PBS-oriented convenience script for generation/training on GH200 systems.

## 3. Core Domain Objects
Defined in `src/config.py`.

- `Extended9Params`: canonical ordered retrieval parameter vector.
- `TerminalState`: 5-channel physical terminal state.
- `GCMulatorConfig`: top-level config object with sections:
  - `paths`
  - `solver`
  - `geometry`
  - `sampling`
  - `normalization`
  - `model`
  - `training`

Fixed terminal state channel ordering is:
1. `Phi`
2. `U`
3. `V`
4. `eta`
5. `delta`

`Phi` in generated files is the SWAMPE/MY_SWAMP perturbation geopotential field (not `Phi + Phibar`), so signed values are expected and physically valid.

Fixed canonical parameter ordering is:
1. `a_m`
2. `omega_rad_s`
3. `Phibar`
4. `DPhieq`
5. `taurad_s`
6. `taudrag_s`
7. `g_m_s2`
8. `K6`
9. `K6Phi`

## 4. End-to-End Flow
### 4.1 CLI Layer (`src/main.py`)
- Provides mutually exclusive modes:
  - `--gen`
  - `--train`
- Loads config path (`--config` or default `config.json`).
- Enforces no-TPU JAX policy before any runtime work.
- Applies generation-safe JAX defaults unless user-overridden:
  - `SWAMPE_JAX_ENABLE_X64=0`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`

### 4.2 Dataset Generation (`src/data_generation.py`)
1. Ensure MY_SWAMP can be imported.
2. Resolve output dataset directory.
3. Sample `n_sims` parameter sets from config distributions.
4. Run MY_SWAMP terminal solves either:
   - scalar (`GCMULATOR_JAX_SIM_BATCH=1`), or
   - batched via JAX `vmap` (`GCMULATOR_JAX_SIM_BATCH>1`, only when `generation_workers==1`), or
   - auto mode (`GCMULATOR_JAX_SIM_BATCH=auto`, default), which currently selects batch `4` on single-worker GPU generation and `1` otherwise.
5. Each solve uses terminal-only MY_SWAMP integration with:
   - `jit_scan=True`
   - `donate_state=True`
6. Stack 5 fields, apply geometry conventions, and write `sim_XXXXXX.npy`.
   - Raw saves use uncompressed `.npy` dictionary payloads (no zip/deflate compression).
7. Write `manifest.json` with run metadata.

### 4.3 Preprocessing (`src/training.py`, `preprocess_dataset`)
1. Read raw `sim_*.npy` (legacy `sim_*.npz` is auto-migrated to `.npy` on first preprocess).
2. Validate raw geometry metadata consistency and ensure it matches `config.geometry`.
3. Train/val split by `training.split_seed` and `training.val_fraction`.
4. Fit normalization stats on train split only (streaming moments).
5. Write normalized per-simulation processed files (`*_train.npy`, `*_val.npy`).
   - Processed saves are written as uncompressed `.npy` dictionary payloads.
6. Write `processed_meta.json` with shape, splits, normalization, geometry, and time-mapping metadata.
7. Reuse cached processed files when raw-file signatures and preprocessing config fingerprint match.

### 4.4 Training (`src/training.py`, `train_emulator`)
1. Build rollout model from config and data shape.
2. Use sphere quadrature-weighted loss (`SphereLoss`).
3. Train for configured epochs.
4. Save `last.pt` every epoch and `best.pt` by validation loss.
5. Evaluate best checkpoint on validation split and save metrics/history JSON.

### 4.5 Extra Utilities
- `extra/predictions.py`:
  - selects the split sample with the maximum `time_days`,
  - allows explicit sample override via `PICKED_PROCESSED_NAME`,
  - predicts terminal state and denormalizes,
  - plots only the signed `Phi` field (`true` vs `predicted`) using a signed symlog color normalization (no positive-only masking),
  - uses `extra/science.mplstyle`,
  - saves the figure under `<model_dir>/plots/`.
- `extra/pytorch_export.py`:
  - exports TorchScript model,
  - bakes in all parameter/state normalization transforms,
  - export consumes physical params and returns physical state.
- `extra/swampe_parity_compare.py`:
  - runs matched forced SWAMPE and MY_SWAMP simulations on CPU for a configurable horizon (default `5` days in-file, editable),
  - compares terminal `Phi/U/V/eta/delta` arrays with strict absolute tolerances,
  - saves a prediction-style side-by-side `Phi` figure plus a JSON metrics report under `extra/parity_outputs/`.

## 5. Configuration Design
Config may be JSON (or YAML if PyYAML is installed). Unknown keys are rejected.

### 5.1 Sampling Rules
- Exactly one of each alias pair must appear:
  - `taurad_s` xor `taurad_hours`
  - `taudrag_s` xor `taudrag_hours`
- Allowed distributions:
  - `uniform`
  - `loguniform`
  - `const`/`fixed`
  - `mixture_off_loguniform`
- Hours aliases are converted to seconds inside sampling.
- `K6Phi` semantics:
  - `K6Phi: 0.0` means explicit zero geopotential diffusion.
  - `K6Phi: null` means inherit `K6` (same diffusion as vorticity/divergence).

### 5.2 Time Mapping
Rollout steps for a sample are computed by:
`round(time_days / default_time_days * rollout_steps_at_default_time)`

Validation enforces finite positive values and minimum step count `>= 1`.

### 5.3 Model Controls
- SFNO base model is wrapped as a conditional rollout model.
- Grid choices are locked by config validation for consistency with MY_SWAMP data and torch-harmonics:
  - `model.grid == "legendre-gauss"`
  - `model.grid_internal == "legendre-gauss"`
  - `model.grid_internal == model.grid`
- Optional input augmentations:
  - parameter maps
  - coordinate channels
- Initial state is generated by a parameter-conditioned IC MLP over basis maps.

### 5.4 Training Controls
- `scheduler.type`: `cosine_warmup`, `plateau`, `none`.
- AMP modes: `none`, `bf16`, `fp16` (fp16 scaler only on CUDA).
- Device: `auto`, `cpu`, `cuda`.
- `training.preload_to_gpu`: when `true`, preloads train/val tensors onto CUDA memory before epoch loops (requires `training.num_workers=0`).

## 6. Geometry Convention
`src/geometry.py` applies two optional conversions:
1. Latitude flip to north-to-south ordering.
2. Longitude roll from `[-pi, pi)` origin to `[0, 2pi)` origin (requires even `nlon`).

Metadata about these choices is stored in each raw simulation file.
Current training pipeline enforces the torch-harmonics-compatible convention via config validation:
- `geometry.flip_latitude_to_north_south == true`
- `geometry.roll_longitude_to_0_2pi == true`
Preprocessing also validates every raw file to ensure geometry metadata is consistent across files and matches config.

## 7. Normalization Design
`src/normalization.py` supports per-channel state transforms before z-score:
- `none`
- `log10`
- `signed_log1p`

Then z-score normalization is applied channelwise for state and parameter vectors.

Important guardrails:
- Standard deviation floors prevent divide-by-zero.
- `log10` transform is treated as strictly positive-only; non-positive values raise an error instead of being silently clipped.
- Constant parameter channels are detected during preprocessing, zeroed after normalization, and recorded in metadata.
- Parameter normalization is clipped to a large finite bound.
- Inverse log10 transform clips exponent range to prevent overflow.

Best-practice for SWAMPE/MY_SWAMP perturbation `Phi` is `signed_log1p` (or `none`) rather than `log10`.
Best-practice for z-score epsilon on these fields is small (`~1e-8`) to avoid compressing low-variance channels (notably `delta`).

Normalization stats are serialized into processed metadata and checkpoints.

## 8. Model Architecture
`src/modeling.py` composes the trainable model as:
1. `ParamICBasisMLP`: maps parameters to initial state on spherical grid.
2. `SphericalFourierNeuralOperator` stepper (from `torch_harmonics.examples`).
3. `ConditionalResidualWrapper`: optional residual prediction with learnable scale.
4. `ParamRolloutModel`: iterative rollout for a fixed number of steps.

Each rollout step receives current state plus optional static channels (params/maps, coords).
Coordinate channels and IC basis maps are built with the same `lat_order` and `lon_origin` conventions persisted in preprocessing metadata/checkpoints.

Current default SFNO profile is intentionally moderate-large and aligned with torch-harmonics patterns:
- `scale_factor=2`
- `embed_dim=128`
- `num_layers=6`
- `encoder_layers=2`
- `activation_function=gelu`
- `use_mlp=true`, `mlp_ratio=2.0`
- `normalization_layer=instance_norm`
- `drop_rate=0.0`, `drop_path_rate=0.05`
- `pos_embed=spectral`, `hard_thresholding_fraction=1.0`

Compatibility note: the installed `torch_harmonics` SFNO currently accepts `encoder_layers` but internally hard-codes a one-layer encoder. `gcmulator` therefore reapplies encoder construction after SFNO init so configured `encoder_layers` is actually enforced.

## 9. Data and Artifact Contracts
### 9.1 Raw Simulation File (`sim_XXXXXX.npy`)
Contains:
- `state_final` `[C,H,W]`
- `fields`
- `params`
- `param_names`
- `time_days`
- `dt_seconds`
- `M`
- `nlat`, `nlon`
- geometry metadata (`lat_order`, `lon_origin`, `lon_shift`)
- stored as a pickled `.npy` dictionary payload for zero-compression writes

### 9.2 Raw Manifest (`manifest.json`)
Contains run metadata, solver settings, sampling settings, and item list.
Sampling metadata includes:
- `generation_workers_requested`
- `generation_workers_used`
- `jax_backend`
- `jax_sim_batch_size`

### 9.3 Processed File (`*_train.npy` / `*_val.npy`)
Contains:
- `state_final_norm`
- `params_norm`
- `time_days`
- written as uncompressed `.npy` dictionary payload (no zip/deflate compression)

### 9.4 Processed Metadata (`processed_meta.json`)
Contains:
- `fields`
- `param_names`
- `constant_param_names`
- `shape`
- split file lists
- normalization stats
- solver/time-mapping info

### 9.5 Training Checkpoints (`best.pt`, `last.pt`)
Contain model weights plus enough metadata to reconstruct inference:
- model config
- solver config
- normalization stats
- field/parameter names and shape
- full resolved config snapshot and source config path

### 9.6 Additional Outputs
- `training_history.json`
- `training_history.csv`
- `val_metrics.json`
- `config_used.resolved.json`
- `config_used.original.<ext>`
- `model_export.torchscript.pt`
- `model_export.meta.json`
- `plots/phi_true_vs_pred_max_days.png`
- `extra/parity_outputs/phi_swampe_vs_my_swamp_<time>d.png`
- `extra/parity_outputs/swampe_vs_my_swamp_<time>d_metrics.json`

## 10. Runtime Dependencies and Import Resolution
Required core dependencies:
- `numpy`
- `torch`
- `torch_harmonics` (for training/model)
  - launcher defaults install from NVIDIA GitHub: `git+https://github.com/NVIDIA/torch-harmonics.git`

Optional/conditional dependencies:
- `my_swamp` (for data generation)
- `matplotlib` (prediction figure script)
- `PyYAML` (only for YAML config loading)

Import strategy:
- `src/main.py` and post-training scripts ensure project root is on `sys.path`.
- `ensure_my_swamp_importable` and `ensure_torch_harmonics_importable` try installed imports first, then nearby sibling checkouts.

## 11. Execution Interfaces
Local environment requirement:
- All local `gcmulator` generation/training/export commands should be run from conda environment `swamp_compare`.
- The same `swamp_compare` environment must provide `torch_harmonics` importability (for example from editable install rooted at `/Users/imalsky/Desktop/SWAMPE_Project/torch-harmonics-main`).

### 11.1 Direct CLI
- `python src/main.py --gen --config config.json`
- `python src/main.py --train --config config.json`

### 11.2 Convenience Script (`run.sh`)
- Activates conda env (`CONDA_ENV`, default `swamp_compare`).
- Uses `MAIN_PY` (default `src/main.py`).
- Optionally runs generation only when raw sims are missing (`RUN_GEN_IF_MISSING=1`).
- Always refreshes `my_swamp` from package source each run (`MY_SWAMP_PACKAGE_SPEC`, default `my-swamp` from TestPyPI via `MY_SWAMP_PIP_ARGS`).
- Reinstalls only `my_swamp` with `--no-deps` so existing environment package versions remain unchanged.
- Always refreshes `torch_harmonics` before training (`TORCH_HARMONICS_PACKAGE_SPEC`, default `git+https://github.com/NVIDIA/torch-harmonics.git`, configurable pip args via `TORCH_HARMONICS_PIP_ARGS`, default `--no-deps --no-build-isolation`).
- Defaults to CPU-extension build mode during install (`TORCH_HARMONICS_FORCE_CPU_BUILD=1`) to avoid host CUDA toolkit mismatches with PyTorch CUDA runtime; set `TORCH_HARMONICS_FORCE_CPU_BUILD=0` to allow CUDA-extension auto-detection.
- Verifies `torch_harmonics` import immediately after install and fails early if unavailable.
- Exposes generation/runtime defaults via environment:
  - `SWAMPE_JAX_ENABLE_X64` (default `0`)
  - `XLA_PYTHON_CLIENT_PREALLOCATE` (default `false`)
  - `GCMULATOR_JAX_SIM_BATCH` (default `auto`)
- When generation already produced `sim_*.npy` (or legacy `sim_*.npz`), reruns skip regeneration and reuse existing raw data.
- Runs training.

### 11.3 PBS Convenience Script (`run.pbs`)
- Uses PBS job directives for GH200 targets (`gpu_long`, `model=gh200`, configured walltime/resources).
- Loads module stack (`miniconda3/gh2`) and activates conda env (`CONDA_ENV`, default `pyt2_8_gh`).
- Always refreshes `my_swamp` from package source each run (`MY_SWAMP_PACKAGE_SPEC`, default `my-swamp` from TestPyPI via `MY_SWAMP_PIP_ARGS`).
- Reinstalls only `my_swamp` with `--no-deps` so existing environment package versions remain unchanged.
- Always refreshes `torch_harmonics` before training (`TORCH_HARMONICS_PACKAGE_SPEC`, default `git+https://github.com/NVIDIA/torch-harmonics.git`, configurable pip args via `TORCH_HARMONICS_PIP_ARGS`, default `--no-deps --no-build-isolation`).
- Defaults to CPU-extension build mode during install (`TORCH_HARMONICS_FORCE_CPU_BUILD=1`) to avoid host CUDA toolkit mismatches with PyTorch CUDA runtime; set `TORCH_HARMONICS_FORCE_CPU_BUILD=0` to allow CUDA-extension auto-detection.
- Verifies `torch_harmonics` import immediately after install and includes it in runtime preflight checks.
- Enforces runtime GPU preflight by default (`REQUIRE_TORCH_CUDA=1`, `REQUIRE_JAX_GPU=1`) and records periodic `nvidia-smi` samples (`ENABLE_GPU_MONITOR=1`).
- Exposes generation/runtime defaults via environment:
  - `SWAMPE_JAX_ENABLE_X64` (default `0`)
  - `XLA_PYTHON_CLIENT_PREALLOCATE` (default `false`)
  - `GCMULATOR_JAX_SIM_BATCH` (default `auto`)
- Uses generation/training flow control `RUN_GEN_IF_MISSING` consistent with `run.sh`.
- When generation already produced `sim_*.npy` (or legacy `sim_*.npz`), reruns skip regeneration and reuse existing raw data.
- Mirrors training stdout/stderr into a dedicated file via `TRAIN_LOG` (default `${PROJECT_ROOT}/training_${PBS_JOBID}.log`) while still streaming to terminal/PBS output.

### 11.4 SWAMPE Parity Check
- `python extra/swampe_parity_compare.py`
- Designed for the `swamp_compare` environment with both `SWAMPE` and `my_swamp` available.
- Enforces CPU execution via JAX platform settings inside the script.

## 12. Verification Status
- There is currently no committed `tests/` directory in this repository snapshot.
- Runtime correctness is currently guarded by:
  - strict config schema/value validation (`src/config.py`),
  - explicit shape/finite checks in preprocessing and training code paths,
  - launcher-level environment and prerequisite checks in `run.sh` and `run.pbs`.
- Reintroducing unit/smoke tests is recommended before major architecture changes.

## 13. Current Design Notes and Gaps
1. Processed data is rebuilt only when preprocessing fingerprint or raw-file signatures change.
2. All training samples in a batch must share the same `time_days`; mixed values in one batch are rejected.
3. Validation split is file-level random split, not stratified.
4. Batched generation (`GCMULATOR_JAX_SIM_BATCH>1`) is currently only applied in single-worker generation mode (`generation_workers==1`).
5. Post-training scripts are globals-only by design.
6. Post-training prediction figure is intentionally `Phi`-only and uses the max-`time_days` sample in the chosen split (unless a specific sample is forced).
7. Any normalization change (for example switching `Phi` from `log10` to `signed_log1p` or changing z-score epsilons) requires retraining artifacts from preprocessing onward; old checkpoints/processed files are not cross-compatible for scientific comparison.
