# GCMulator Specification

## 1. Purpose
`gcmulator` trains a one-step spherical surrogate for the visible flow produced by `MY_SWAMP`.

The canonical task is:
1. Sample physical parameters.
2. Run `MY_SWAMP` on a Legendre-Gauss grid.
3. Extract many short one-step transitions from each simulation after burn-in.
4. Train an SFNO-based surrogate on `(state_t, params) -> prognostics_{t+Δ}`.
5. Reconstruct diagnostic winds outside the network when a full physical state is needed.

This repository does not attempt to emulate the exact internal two-level solver carry. The target is a clean surrogate of the visible flow map.

## 2. Scientific Contract
### 2.1 Input State
The model input is the full visible physical state in this fixed order:
1. `Phi`
2. `U`
3. `V`
4. `eta`
5. `delta`

### 2.2 Learned Target
The model predicts only the prognostic variables in this fixed order:
1. `Phi`
2. `eta`
3. `delta`

`U` and `V` are not independent network targets. They are reconstructed deterministically from `eta` and `delta` using `MY_SWAMP` spectral inversion utilities when downstream tools need a full five-field state.

## 3. Conditioning
The conditioning vector is fixed and ordered as:
1. `a_m`
2. `omega_rad_s`
3. `Phibar`
4. `DPhieq`
5. `taurad_s`
6. `taudrag_s`
7. `g_m_s2`

`transition_days` remains stored as raw metadata, but it is not part of the active model input for the fixed-jump training pipeline.

Conditioning is injected with a small MLP that produces FiLM parameters for the SFNO latent states. The active path does not broadcast parameter maps over every grid cell.

## 4. Geometry
The geometry is locked to Legendre-Gauss.

Stored grid conventions are:
1. Latitude ordered north-to-south.
2. Longitude rolled to `[0, 2π)`.

These conventions are enforced in generation and validated again in preprocessing.

## 5. Data Generation
Each simulation produces one contiguous post-burn-in window of short transitions.

Active sampling controls:
1. `sampling.burn_in_days`
2. `sampling.transitions_per_simulation`
3. `sampling.transition_jump_steps`

The window start is sampled uniformly after burn-in. This keeps local dynamics dense enough for one-step learning while preserving coverage across the trajectory ensemble.

## 6. Learning Objective
The model learns a one-step spherical flow map in physical state space after normalization:
- input: full visible state at time `t`
- conditioning: static physical parameter vector
- target: prognostic state at time `t + Δ`

Loss is sphere-weighted MSE using quadrature weights on the Legendre-Gauss grid.

## 7. Architecture
The architecture family remains SFNO via `torch_harmonics`.

Active stack:
1. SFNO encoder/backbone/decoder
2. FiLM-style conditioning MLP over latent states
3. Residual prognostic head

Coordinate channels remain optional.

## 8. Evaluation
Acceptance is not based on one-step loss alone.

Required evaluation outputs:
1. One-step normalized and physical RMSE/MAE on prognostic targets.
2. True spherical harmonic power-spectrum mismatch using `torch_harmonics.RealSHT`, not planar FFTs on the latitude-longitude array.
3. Free-run rollout metrics over stored contiguous windows.

Rollout metrics use recursive model predictions with deterministic `U,V` reconstruction at each step.

## 9. Normalization
State transforms support:
1. `none`
2. `log10`
3. `signed_log1p`

Normalization statistics are stored separately for:
1. input visible state
2. target prognostic state
3. conditioning parameters

Constant conditioning channels are zeroed after normalization.

## 10. Raw Data Contract
Each raw `sim_XXXXXX.npy` file stores:
- `state_inputs` with shape `[T, 5, H, W]`
- `state_targets` with shape `[T, 3, H, W]`
- `transition_days` with shape `[T]`
- `anchor_steps` with shape `[T]`
- `input_fields`
- `target_fields`
- `params`
- `param_names`
- `default_time_days`
- `burn_in_days`
- `dt_seconds`
- `starttime_index`
- `transition_jump_steps`
- `n_transitions`
- `M`
- `nlat`
- `nlon`
- `lat_order`
- `lon_origin`
- `lon_shift`

Raw files must match the active config exactly for solver, sampling, parameter ordering, field ordering, and geometry metadata.

## 11. Processed Data Contract
Processed data is written as split-level shard files, one shard per raw simulation file.

Each shard `.npz` stores:
- `state_inputs_norm`
- `state_targets_norm`
- `params_norm`
- `anchor_steps`

The processed format does not duplicate a full `conditioning_norm` row for every transition. Parameter normalization is stored once per shard and expanded by the loader.

Splitting is done by simulation file, not by transition, to prevent leakage.

## 12. Metadata and Checkpoints
`processed_meta.json` stores:
- `input_fields`
- `target_fields`
- `param_names`
- `conditioning_names`
- `input_shape`
- `target_shape`
- split shard entries
- normalization statistics
- solver metadata
- sampling metadata
- geometry metadata
- preprocessing fingerprint

`best.pt` and `last.pt` store:
- model weights
- input field ordering
- target field ordering
- parameter ordering
- normalization statistics
- solver metadata
- sampling metadata
- resolved config snapshot
- runtime AMP mode

## 13. Dependency Contract
Required core dependencies:
- `numpy`
- `torch`
- `torch-harmonics==0.8.1`

Required generation and diagnostic dependency:
- `my_swamp`

Optional plotting dependency:
- `matplotlib`

The active training and generation code assumes these dependencies are importable in the environment. Runtime fallback search paths are not part of the contract.

## 14. Repository Layout
The repository keeps the current flat `src/` layout.

Primary files:
- `src/main.py`
- `src/config.py`
- `src/data_generation.py`
- `src/my_swamp_backend.py`
- `src/modeling.py`
- `src/normalization.py`
- `src/training.py`
- `src/geometry.py`
- `src/sampling.py`
- `extra/predictions.py`
- `extra/pytorch_export.py`
- `extra/training_log.py`
- `extra/batch_size_benchmark.py`
- `unit_tests/`

## 15. Local Integrity Checks
The minimum local integrity pass for this repository is:
1. `python -m pytest unit_tests`
2. `python -m compileall src extra unit_tests`

For code cleanliness, run these developer checks when the tools are available:
1. `ruff check src extra unit_tests`
2. `vulture src extra unit_tests --min-confidence 80`

In the standard local `swamp_compare` conda environment, both `ruff` and `vulture`
are installed. `ruff` must be invoked with a subcommand such as `check`, and
`vulture` must be invoked with one or more paths.

`ruff` and `vulture` are development-time quality gates, not runtime dependencies.

## 16. Exactness Statement
This version intentionally models the visible flow, not the exact discrete MY_SWAMP solver carry.

That means:
- no learned previous-step carry in the network state
- prognostic-only targets
- deterministic diagnostic reconstruction outside the network when full-state outputs are required

This is an approximate surrogate of the flow map seen through the visible physical state.
