# GCMulator Specification

## 1. Purpose
GCMulator trains a state-conditioned spherical surrogate for MY_SWAMP dynamics.

Current objective:
1. Validate and load experiment configuration.
2. Generate trajectory-transition truth data from MY_SWAMP.
3. Preprocess and train a transition model on `(state_t, conditioning) -> state_{t+Δt}`.
4. Run post-training utilities for rollout visualization and TorchScript export.

Main runtime entrypoint: `src/main.py`.

### 1.1 Change Status (2026-03-05)
Changed:
- Training target moved from terminal-state supervision to trajectory-transition supervision.
- Raw and processed datasets now store transition pairs, not single terminal states.
- Model interface in training is state-conditioned (`state_t` + conditioning input).
- Model conditioning now appends normalized transition duration (`transition_days`).
- User-facing conditioning excludes internal diffusion controls.
- Runtime AMP safety gate remains active and records `runtime_amp_mode`.
- Trajectory segmentation now preserves full MY_SWAMP internal state and absolute time index continuity.
- Transition sampling now supports configurable jump size (`model.transition_jump_steps`).

Unchanged:
- MY_SWAMP solver physics and integration internals.
- Geometry conventions and field ordering.
- Sphere-weighted loss (`SphereLoss`) and core SFNO family.

## 2. Repository Structure
- `src/`: config, generation, preprocessing, model, training.
- `extra/`: prediction/export/parity utilities.
- `config.json`: default experiment configuration.
- `run.sh`, `run.pbs`: convenience launch scripts.

## 3. Core Domain Objects
Defined in `src/config.py`.

- `Extended9Params`: internal solver parameter object passed to MY_SWAMP.
- `TerminalState`: 5-channel physical state container used in backend conversion.
- `GCMulatorConfig`: top-level config with sections: `paths`, `solver`, `geometry`, `sampling`, `normalization`, `model`, `training`.

Fixed state channel ordering:
1. `Phi`
2. `U`
3. `V`
4. `eta`
5. `delta`

Fixed user-facing conditioning-vector ordering:
1. `a_m`
2. `omega_rad_s`
3. `Phibar`
4. `DPhieq`
5. `taurad_s`
6. `taudrag_s`
7. `g_m_s2`

Conditioning vector definition:
- User-facing parameter vector: the 7 physical parameters listed above.
- Model conditioning vector: user-facing parameter vector plus trailing normalized `transition_days`.
- Current model-conditioning order is:
  1. `a_m`
  2. `omega_rad_s`
  3. `Phibar`
  4. `DPhieq`
  5. `taurad_s`
  6. `taudrag_s`
  7. `g_m_s2`
  8. `transition_days`

## 4. End-to-End Flow
### 4.1 CLI (`src/main.py`)
- Modes: `--gen`, `--train`.
- Loads config (`--config` or default `config.json`).
- Enforces no-TPU JAX policy and generation-safe defaults.

### 4.2 Dataset Generation (`src/data_generation.py`)
1. Ensure MY_SWAMP importability.
2. Sample configured parameter sets.
3. For each simulation, run chunked MY_SWAMP trajectory segments to build transitions:
   - Use physical horizon from `solver.default_time_days` and `solver.dt_seconds`.
   - Use `model.rollout_steps_at_default_time` as number of sampled transitions per simulation.
   - Use `model.transition_jump_steps` as direct transition jump size in solver steps.
   - Sample anchor times across one continuous integration, then pair `state_t -> state_{t+jump}`.
   - Integration preserves full solver state and absolute time index continuity.
4. Save transition tensors and metadata into `sim_XXXXXX.npy`.
5. Write `manifest.json`.

Notes:
- Generation currently uses scalar simulation execution for trajectory transitions.
- Raw saves are uncompressed `.npy` dictionary payloads.

### 4.3 Preprocessing (`src/training.py`)
1. Read raw `sim_*.npy` (legacy `sim_*.npz` auto-migrates).
2. Validate geometry consistency and config match.
3. Split by simulation file into train/val/test.
4. Fit normalization stats on train split only.
5. Fit transition-duration normalization stats on train split.
6. Expand each raw trajectory file into per-transition processed files.
7. Write `processed_meta.json` and cache fingerprint.

Important:
- Processed samples are transition-level, but splitting is simulation-level to avoid leakage across splits.

### 4.4 Training (`src/training.py`, `train_emulator`)
1. Build state-conditioned rollout model.
2. Train one-step transition prediction with sphere quadrature-weighted loss.
3. Save `last.pt` per epoch and `best.pt` by validation loss.
4. Evaluate best checkpoint on val/test transition sets.

Model call during training:
- `y_hat = model(state_t_norm, conditioning_norm, steps=1)`
- `conditioning_norm` includes normalized parameters and normalized `transition_days`.
- Supervision is currently single-transition (`steps=1`) by design.
- The model interface is rollout-capable (`steps>=1`) for future multi-step objectives.

### 4.5 Extra Utilities
- `extra/predictions.py`:
  - picks one transition sample from a split,
  - builds transition-aware conditioning (`params + transition_days`),
  - predicts one direct transition (`state_t -> state_{t+jump}`),
  - compares predicted vs truth `Phi` map for that sample.
- `extra/pytorch_export.py`:
  - exports TorchScript wrapper with baked normalization,
  - exported module consumes `(state0_physical, params_physical, transition_days_physical)` and returns one direct transition.
- `extra/swampe_parity_compare.py`:
  - SWAMPE vs MY_SWAMP terminal parity utility.

## 5. Configuration Design
Unknown config keys are rejected.

### 5.1 Sampling Rules
- Exactly one alias from each pair:
  - `taurad_s` xor `taurad_hours`
  - `taudrag_s` xor `taudrag_hours`
- Allowed dists: `uniform`, `loguniform`, `const`/`fixed`, `mixture_off_loguniform`.
- Internal diffusion controls are fixed in code and not user-configurable in `sampling.parameters`.

### 5.2 Time/Transition Mapping
- Truth integration horizon is `solver.default_time_days`.
- Number of sampled transitions per simulation is `model.rollout_steps_at_default_time`.
- Direct transition jump size is `model.transition_jump_steps` (in solver steps).
- Per-sample physical duration is `transition_days = transition_jump_steps * dt_seconds / 86400`.
- Current default config uses `dt_seconds=240` and `transition_jump_steps=1`.

### 5.3 Model Controls
- Grid is locked to Legendre-Gauss conventions for torch-harmonics compatibility.
- Inputs per step: current state + optional parameter maps + optional coordinate channels.
- Transition duration enters conditioning as a normalized scalar (`transition_days`) appended to normalized physical parameters.
- `model.transition_jump_steps=1` is the default consecutive-transition training mode.
- Legacy IC-network controls remain in config but are not used by transition training path.

### 5.4 Training Controls
- Scheduler: `cosine_warmup`, `plateau`, `none`.
- AMP: `none`, `bf16`, `fp16`.
- Runtime AMP safety gate disables AMP when transformed grid dimensions are not powers of two.
- `training.preload_to_gpu=true` requires CUDA and `num_workers=0`.

## 6. Geometry Convention
`src/geometry.py` applies:
1. Latitude flip to north-to-south.
2. Longitude roll to `[0, 2pi)` origin.

Training enforces:
- `geometry.flip_latitude_to_north_south == true`
- `geometry.roll_longitude_to_0_2pi == true`

## 7. Normalization
`src/normalization.py` supports per-channel transforms before z-score:
- `none`
- `log10`
- `signed_log1p`

Then z-score normalization is applied channelwise for states and parameters.

Guardrails:
- Std floors, constant-parameter handling, finite clipping, inverse-transform safety.

## 8. Model Architecture
Transition model stack:
1. SFNO stepper (`torch_harmonics.examples`).
2. `ConditionalResidualWrapper` for residual update.
3. `StateConditionedRolloutModel` for iterative state-conditioned rollout.

Each step consumes current state and static conditioning channels.

## 9. Data and Artifact Contracts
### 9.1 Raw Simulation File (`sim_XXXXXX.npy`)
Contains:
- `state_inputs` `[T,C,H,W]`
- `state_targets` `[T,C,H,W]`
- `transition_days` `[T]`
- `fields`
- `params`
- `param_names`
- `time_days`
- `dt_seconds`
- `transition_jump_steps`
- `n_transitions`
- `M`, `nlat`, `nlon`
- geometry metadata: `lat_order`, `lon_origin`, `lon_shift`

### 9.2 Raw Manifest (`manifest.json`)
Includes:
- solver settings,
- sampling metadata,
- `conditioning_param_names`,
- generated item summaries.

### 9.3 Processed Transition File (`train/*.npy`, `val/*.npy`, `test/*.npy`)
Contains:
- `state_input_norm` `[C,H,W]`
- `state_target_norm` `[C,H,W]`
- `params_norm` `[P]`
- `conditioning_norm` `[P+1]` when transition-duration conditioning is present
- `transition_days` scalar
- `transition_days_norm` scalar

### 9.4 Processed Metadata (`processed_meta.json`)
Contains:
- `task="trajectory_transition"`
- `fields`, `param_names`, `conditioning_names`, `constant_param_names`, `constant_conditioning_names`
- shape (`C,H,W`)
- split file lists
- normalization stats
- transition-duration normalization stats (`transition_days_norm`)
- geometry and solver metadata
- preprocess fingerprint

### 9.5 Training Checkpoints (`best.pt`, `last.pt`)
Contain:
- `mode="state_conditioned_trajectory_transition"`
- model weights
- field/parameter names, conditioning names, and shape
- normalization stats
- transition-duration normalization stats (`transition_days_norm`)
- config snapshots
- `runtime_amp_mode`

### 9.6 Additional Outputs
- `training_history.json`
- `training_history.csv`
- `val_metrics.json`
- `test_metrics.json`
- `config_used.resolved.json`
- `config_used.original.<ext>`
- `model_export.torchscript.pt`
- `model_export.meta.json`
- `plots/phi_true_vs_pred_max_days.png`

## 10. Runtime Dependencies and Import Resolution
Core:
- `numpy`
- `torch`
- `torch_harmonics`

Optional:
- `my_swamp` (generation)
- `matplotlib` (prediction utility)
- `PyYAML` (YAML config)

Import strategy:
- Prefer installed packages, fallback to sibling checkouts.

## 11. Execution Interfaces
### 11.1 Direct CLI
- `python src/main.py --gen --config config.json`
- `python src/main.py --train --config config.json`

### 11.2 Convenience Scripts
- `run.sh` and `run.pbs` provide environment setup, optional generation, and training launch.

## 12. Verification Status
Current correctness gates:
- strict config schema/value validation,
- preprocessing/data-contract checks,
- finite checks and training/runtime checks,
- denormalized-space RMSE metrics,
- spectral relative-power RMSE metrics,
- lightweight physical-summary diagnostics.

## 13. Current Design Notes and Gaps
1. Current training is intentionally one-step transition supervision (`steps=1`) and no rollout curriculum is planned in current scope (user decision on 2026-03-05).
2. Prognostic-only target (`eta, delta, Phi` with diagnostic `U,V` reconstruction) is not implemented yet.
3. Prediction utility expansion beyond the current one-sample `Phi` panel is deferred for now (user decision on 2026-03-05).
4. IC strategy changes (analytic IC or analytic+residual IC) are deferred for now (user decision on 2026-03-05).
5. Mixed jump sizes within a single run are deferred for now; jump size remains configured per run (`model.transition_jump_steps`) (user decision on 2026-03-05).
6. Rebuilding raw+processed data is required after schema changes.
