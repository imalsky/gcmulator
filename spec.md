# GCMulator Specification

## 1. Purpose And Scope
`gcmulator` trains a direct-jump spherical surrogate for the visible flow
produced by `MY_SWAMP`.

The canonical task is:
1. Sample physical parameters.
2. Run `MY_SWAMP` on a Legendre-Gauss spherical grid.
3. Extract a post-burn-in window of direct-jump transitions.
4. Train a state-conditioned SFNO surrogate on
   `(state_t, params, transition_days) -> target_{t+Δ}`.
5. Evaluate single-call direct-jump accuracy, spherical spectra, and free-run
   rollout behavior.
6. Export and inspect trained checkpoints with the utilities in `extra/`.

This repository does not emulate the exact internal two-level solver carry. It
learns a surrogate of the visible flow map seen through the physical state.

### 1.1 Working Environment And First Checks
Use the local `swamp_compare` conda environment for generation, training,
export, and repository checks.

Default operator routine:
1. Activate the environment:
   `conda activate swamp_compare`
2. Work from the repository root:
   `cd /Users/imalsky/Desktop/SWAMPE_Project/gcmulator`
3. Confirm the required imports before debugging repository code:
   `python -c "import torch, torch_harmonics, my_swamp; print(torch.__version__, getattr(torch_harmonics, '__version__', 'unknown'))"`
4. Run the minimum integrity checks:
   `python -m pytest unit_tests`
   `python -m compileall src extra unit_tests`
5. Run developer cleanliness checks when available:
   `ruff check src extra unit_tests`
   `vulture src extra unit_tests --min-confidence 80`
6. Use the main entrypoints from the same environment:
   `python src/main.py --gen --config config.json`
   `python src/main.py --train --config config.json`

Important dependency naming:
1. the install package is `torch-harmonics==0.8.1`
2. the Python import name is `torch_harmonics`

If `python -c "import torch_harmonics"` fails inside `swamp_compare`, fix the
environment before changing repository code.

## 2. Engineering Principles
### 2.1 Fail Fast
The repository should prefer a small number of high-value contract checks over a
large amount of defensive fallback logic.

Required engineering stance:
1. Validate external boundaries aggressively.
2. Raise immediately on contract violations.
3. Do not silently coerce invalid inputs into "best effort" behavior.
4. Do not add fallback search paths, fallback schemas, silent retries, or
   compatibility branches unless there is a strong documented reason.
5. Once a config, artifact, or tensor contract is validated, downstream code
   should stay simple rather than repeating the same checks everywhere.

Boundary checks are appropriate at:
1. config parsing
2. raw-file loading
3. processed-metadata loading
4. model input shape validation
5. export/checkpoint loading

Current code still contains a few transitional compatibility shims, such as
`python src/main.py` bootstrap logic and the `torch_harmonics` import shim in
`src/modeling.py`. These should be treated as technical debt, not as a pattern
to extend.

### 2.2 No Fallback Contract
The active contract is:
1. no silent fallback from missing dependencies to alternate backends
2. no silent fallback from wrong field ordering to inferred ordering
3. no silent fallback from wrong geometry metadata to guessed geometry
4. no silent fallback from variable-jump training to fixed-jump training
5. no silent fallback from missing files to synthetic defaults

If a dependency or artifact is wrong, fail immediately with a concrete error.

### 2.3 Performance Style
Numerical code should prefer:
1. batch tensor operations over Python loops
2. `numpy` vectorization for CPU-side preprocessing
3. `jax.vmap` + `jax.lax.scan` for batched MY_SWAMP rollouts
4. batched `torch` operations for model inference and loss evaluation

`sampling.generation_workers` is not an OS worker pool. In the active
implementation it is a JAX trajectory batch size for vectorized generation.

## 3. Scientific And Physics Contract
### 3.1 MY_SWAMP Overview
According to the local `MY_SWAMP` code and README, `my_swamp` is a JAX rewrite
of the SWAMPE spectral shallow-water model on the sphere.

Key preserved numerical ideas:
1. single-layer global spectral shallow-water dynamics
2. triangular truncation with `M = N`
3. Gaussian quadrature in latitude
4. FFT in longitude
5. prognostic evolution in spectral space
6. diagnostic reconstruction of winds from vorticity/divergence

`gcmulator` uses `MY_SWAMP` in forced mode with:
1. `forcflag=True`
2. `diffflag=True`
3. `expflag=False`
4. `modalflag=True`
5. `diagnostics=False`
6. `alpha=0.01`

That means the checked-in surrogate is aligned to the modified-Euler,
diffusion-enabled, forced MY_SWAMP path, not to every possible solver mode.

### 3.2 Physical State Variables
The visible physical state has five channels in this exact order:
1. `Phi`
2. `U`
3. `V`
4. `eta`
5. `delta`

Definitions:
1. `Phi`
   Perturbation geopotential-like field. In MY_SWAMP, the total geopotential-like
   quantity is `Phi + Phibar`.
2. `U`
   Physical-space wind component carried by MY_SWAMP and exposed in the visible
   state. It is a real physical variable, not an artificial helper.
3. `V`
   The second physical-space wind component carried by MY_SWAMP and exposed in
   the visible state.
4. `eta`
   The vorticity-like prognostic channel named `eta`. The MY_SWAMP README
   describes this as absolute vorticity.
5. `delta`
   Horizontal divergence.

Important wording:
`U` and `V` are physical wind variables. In this repository they are called
"diagnosed" only in the narrower ML sense that the emulator reconstructs them
from predicted prognostics rather than learning them as direct targets.

### 3.3 Learned Target Variables
The active target is the prognostic subset:
1. `Phi`
2. `eta`
3. `delta`

Current target ordering is fixed as:
`("Phi", "eta", "delta")`

This is an intentional scientific simplification:
1. the emulator sees the full visible state
2. the emulator predicts only the prognostic subset
3. `U` and `V` are reconstructed deterministically from `eta` and `delta`
   during rollout evaluation or full-state reconstruction

### 3.4 Conditioning Parameters
The physical conditioning vector is fixed and ordered as:
1. `a_m`
2. `omega_rad_s`
3. `Phibar`
4. `DPhieq`
5. `taurad_s`
6. `taudrag_s`
7. `g_m_s2`

Definitions:
1. `a_m`
   Planetary radius in meters.
2. `omega_rad_s`
   Rotation rate in radians per second.
3. `Phibar`
   Mean background geopotential-like offset added back to `Phi` inside MY_SWAMP.
4. `DPhieq`
   Amplitude of the equilibrium geopotential contrast used in the dayside forcing.
5. `taurad_s`
   Newtonian relaxation timescale in seconds.
6. `taudrag_s`
   Rayleigh drag timescale in seconds.
7. `g_m_s2`
   Gravitational acceleration in meters per second squared.

Internally fixed but not part of the learned conditioning vector:
1. `K6`
   Sixth-order diffusion coefficient for vorticity/divergence.
2. `K6Phi`
   Geopotential diffusion control. `None` disables the dedicated Phi diffusion
   path in some MY_SWAMP builds.

The config may specify `taurad_hours` and `taudrag_hours`, but they are converted
to canonical second-valued parameters before entering the model pipeline.

Model-conditioning adds one explicit time feature:
1. `transition_days`
   The requested direct-jump horizon in physical days.

### 3.5 Forcing Physics
The local `MY_SWAMP/src/my_swamp/forcing.py` code defines:
1. `Phieq`
   Equilibrium geopotential field
2. `Q`
   Radiative/Newtonian relaxation forcing on `Phi`
3. `R = (F, G)`
   Velocity forcing terms including drag

Active forced-mode formulas:
1. `Phieq = Phibar + DPhieq * cos(lambda) * sqrt(1 - mu^2)` on the dayside,
   and `Phibar` on the nightside
2. `Q = (Phieq - (Phi + Phibar)) / taurad`
3. velocity forcing applies the SWAMPE drag logic and keeps the original
   `Q < 0` handling

This matters because the surrogate is learning trajectories generated under
forced, damped shallow-water dynamics, not free decay and not unforced test cases.

### 3.6 Spectral Diagnosis Of Winds
The repository reconstructs winds using MY_SWAMP spectral transforms:
1. transform `eta` and `delta` into spectral coefficients
2. apply inverse wind reconstruction with `invrsUV`
3. return real-valued physical-space `U,V`

That diagnosis is implemented in `src/my_swamp_backend.py::diagnose_winds`.

### 3.7 Time Variables
Definitions:
1. `dt_seconds`
   Solver time step for one MY_SWAMP advance.
2. `default_time_days`
   Physical integration horizon for one raw simulation.
3. `burn_in_days`
   Initial simulated duration discarded before transition sampling.
4. `transition_jump_steps`
   Integer solver-step gap between input and target states.
5. `transition_jump_steps_max`
   Optional maximum solver-step gap when variable-jump training is enabled.
6. `transition_days`
   Physical duration of one sampled jump:
   `transition_jump_steps * dt_seconds / 86400`.
7. `anchor_steps`
   Integer solver step index of each input state inside a sampled trajectory window.
   In the active contract, consecutive anchors are separated by `transition_jump_steps`
   so that `state_targets[t]` becomes the next visible input in free-run rollout.
8. `anchor_stride_steps`
   Stored integer spacing between successive anchors. Active value:
   `anchor_stride_steps = transition_jump_steps`.
9. `starttime_index`
   MY_SWAMP initial time index. The active config requires `>= 2`, matching the
   two-level solver initialization constraints enforced by the code.

Important distinction:
`gcmulator` does not learn a variable internal MY_SWAMP solver `dt_seconds`.
The solver step remains fixed inside data generation. The learned variable-time
behavior is the direct-jump horizon `transition_days`.

### 3.8 Flexible Jump Requirement
The active contract supports both:
1. fixed-jump training via `transition_jump_steps`
2. variable-jump training via the inclusive solver-step range
   `[transition_jump_steps, transition_jump_steps_max]`

Required semantics:
1. raw files store per-sample `transition_days`
2. preprocessing normalizes `transition_days` explicitly
3. model conditioning appends normalized `transition_days`
4. no stage silently collapses variable-jump data into a fixed-jump assumption
5. rollout windows must use `anchor_stride_steps = transition_jump_steps` so
   direct-jump rollout is physically consistent

## 4. End-To-End Pipeline
### 4.1 CLI
`src/main.py` exposes two stages:
1. `--gen`
2. `--train`

Generation writes raw files and a descriptive manifest.
Training performs preprocessing internally before optimization.

### 4.2 Generation
`src/data_generation.py`:
1. validates importability of `my_swamp`
2. samples parameter sets
3. samples one valid post-burn-in window start and one jump duration per simulation
4. runs either serial or batched trajectory extraction
5. writes `sim_XXXXXX.npy` raw files
6. writes `manifest.json`

### 4.3 Preprocessing
`src/training.py::preprocess_dataset`:
1. loads and validates every raw file against the active config
2. splits at the simulation-file level, never at the transition level
3. fits normalization on train files only
4. writes one processed shard per raw simulation file
5. writes `processed_meta.json`
6. reuses cached processed data only when the preprocessing fingerprint matches

### 4.4 Training
`src/training.py::train_emulator`:
1. loads processed metadata
2. builds the SFNO-based transition model
3. trains with sphere-weighted loss
4. saves `last.pt` and `best.pt`
5. computes single-call direct-jump metrics and rollout metrics
6. writes summary JSON/CSV artifacts

### 4.5 Export And Utilities
The `extra/` scripts provide:
1. direct-jump qualitative prediction plots
2. TorchScript export with baked normalization
3. training-curve plots
4. batch-size timing benchmarks for direct-jump inference
5. SWAMPE vs MY_SWAMP parity diagnostics

### 4.6 How To Run
All commands assume the working directory is the repository root (`gcmulator/`).

**Generate raw training data:**
```bash
python src/main.py --gen --config config.json
```

**Train the emulator** (includes preprocessing):
```bash
python src/main.py --train --config config.json
```

**Both stages in sequence** (equivalent to `run.sh`):
```bash
python src/main.py --gen --config config.json
python src/main.py --train --config config.json
```

**Utility scripts** (run from the repository root after training):
```bash
python extra/predictions.py        # qualitative prediction plots
python extra/pytorch_export.py     # TorchScript export
python extra/training_log.py       # training loss curves
python extra/batch_size_benchmark.py  # inference latency benchmark
```

Each utility script has user-editable constants at the top (e.g. `RUN_NAME`,
`CHECKPOINT_PATH`) that control which trained run is inspected. Edit those
before running.

## 5. Torch-Harmonics Interface
### 5.1 Required APIs
The active code depends on these `torch_harmonics` APIs:
1. `torch_harmonics.examples.models.sfno.SphericalFourierNeuralOperator`
2. `torch_harmonics.examples.losses.get_quadrature_weights`
3. `torch_harmonics.RealSHT`

Preferred baseline version:
`torch-harmonics==0.8.1` installed into `swamp_compare`, imported in Python as
`torch_harmonics`

### 5.2 How GCMulator Wraps SFNO
`src/modeling.py` does not use bare SFNO directly. It wraps it as:
1. visible state input
2. optional deterministic coordinate channels
3. SFNO encoder
4. FiLM modulation from parameter conditioning
5. SFNO positional embedding and block stack
6. SFNO decoder
7. optional residual prognostic head

The active wrapper class is:
`StateConditionedTransitionModel`

### 5.3 FiLM Conditioning
Conditioning is global, not per-grid-cell parameter maps.

The active structure is:
1. a small MLP maps `[B, P]` conditioning vectors to FiLM parameters
2. FiLM applies after the encoder and after every SFNO block
3. scale/shift tensors have shape `[B, S, C]`
4. the spatial latent maps are modulated via broadcasting

### 5.4 Encoder Depth Patch
`src/modeling.py` patches `torch_harmonics` encoder depth because some 0.8.1
builds ignore `encoder_layers`.

This is a repository-specific compatibility fix, not a scientific feature.

### 5.5 Losses And Spectral Metrics
`torch_harmonics` is used in two distinct ways:
1. `get_quadrature_weights` for sphere-aware training loss
2. `RealSHT` for true spherical power-spectrum evaluation

No planar FFT surrogate is part of the contract for spherical spectral metrics.

## 6. File Responsibilities
### 6.1 Repository Root
1. `config.json`
   Default experiment config.
2. `spec.md`
   This repository contract.
3. `run.sh`
   Convenience shell entrypoint for local/Slurm-style GPU runs.
4. `run.pbs`
   Convenience PBS entrypoint for HPC runs.

### 6.2 Source Files
1. `src/main.py`
   CLI entrypoint for generation and training.
2. `src/config.py`
   Config dataclasses, parsing, validation, constants, and path resolution.
3. `src/sampling.py`
   Parameter sampling and conversion into canonical conditioning vectors.
4. `src/geometry.py`
   Canonical latitude/longitude orientation helpers.
5. `src/data_generation.py`
   Raw simulation generation and raw-file writing.
6. `src/my_swamp_backend.py`
   MY_SWAMP interface layer: initialization, rollout extraction, batched JAX
   window extraction, wind diagnosis, and full-state reconstruction.
7. `src/normalization.py`
   State/parameter normalization, inverse normalization, and JSON metadata
   serialization helpers.
8. `src/modeling.py`
   Device helpers, AMP helpers, sphere loss, coordinate channels, SFNO import,
   FiLM conditioning, and model construction.
9. `src/training.py`
   Raw validation, preprocessing, shard datasets, training loop, metrics,
   checkpoint writing, and final run summary.

### 6.3 Extra Utilities
1. `extra/predictions.py`
   Plot one direct-jump prognostic prediction against truth from a checkpoint.
2. `extra/pytorch_export.py`
   Export a trained checkpoint as a physical-space TorchScript module and write
   export metadata.
3. `extra/training_log.py`
   Plot train-vs-val loss curves from `training_history.csv`.
4. `extra/batch_size_benchmark.py`
   Benchmark direct-jump inference latency across batch sizes.
5. `extra/swampe_parity_compare.py`
   Compare SWAMPE and MY_SWAMP outputs for parity/debugging.
6. `extra/science.mplstyle`
   Shared plotting style for utility scripts.

### 6.4 Unit Tests
1. `unit_tests/conftest.py`
   Local source-tree import bootstrap for tests.
2. `unit_tests/test_config.py`
   Config schema and burn-in regression checks.
3. `unit_tests/test_normalization.py`
   Normalization round-trip and constant-channel checks.
4. `unit_tests/test_geometry_and_sampling.py`
   Geometry transform and parameter-alias conversion checks.
5. `unit_tests/test_data_generation.py`
   High-value generation integrity checks, including serial-vs-batched parity.

## 7. Config Contract And Variable Definitions
### 7.1 `paths`
1. `dataset_dir`
   Raw simulation directory.
2. `processed_dir`
   Processed shard directory.
3. `model_dir`
   Run artifact directory for checkpoints and metrics.
4. `overwrite_dataset`
   Whether generation may delete existing raw files before writing new ones.

### 7.2 `solver`
1. `M`
   Spectral truncation. The active config validation accepts the supported
   MY_SWAMP resolutions only.
2. `dt_seconds`
   Solver time step.
3. `default_time_days`
   One raw simulation horizon.
4. `starttime_index`
   Initial MY_SWAMP time index.

Internal MY_SWAMP geometry terms derived from `M`:
1. `N`
   Triangular truncation degree. In the SWAMPE-JAX docs, `M = N`.
2. `I`
   Longitude count, equivalent to `nlon`.
3. `J`
   Latitude count, equivalent to `nlat`.
4. `lambdas`
   Longitude grid in radians.
5. `mus`
   `sin(latitude)` values on the Gaussian grid.
6. `w`
   Gaussian quadrature weights in latitude.

### 7.3 `geometry`
1. `flip_latitude_to_north_south`
   Whether stored arrays are flipped into north-to-south latitude order.
2. `roll_longitude_to_0_2pi`
   Whether longitudes are rolled from `[-π, π)` to `[0, 2π)`.

### 7.4 `sampling`
1. `seed`
   Generation seed.
2. `n_sims`
   Number of raw simulation files to generate.
3. `generation_workers`
   Vectorized JAX trajectory batch size during generation.
4. `burn_in_days`
   Discarded initial spin-up duration before the sampled window starts.
5. `transitions_per_simulation`
   Number of transition pairs extracted from one simulation window.
6. `transition_jump_steps`
   Minimum or fixed solver-step offset between input and target.
7. `transition_jump_steps_max`
   Optional maximum solver-step offset for variable-jump training.
8. `parameters`
   List of per-parameter sampling rules.

Config examples:

Fixed direct-jump training:
```json
"sampling": {
  "transition_jump_steps": 1,
  "transition_jump_steps_max": 1
}
```

Variable direct-jump training:
```json
"sampling": {
  "transition_jump_steps": 1,
  "transition_jump_steps_max": 8
}
```

Interpretation:
1. `dt_seconds` remains the fixed MY_SWAMP solver step
2. the sampled direct-jump horizon varies over the inclusive integer range
   `[transition_jump_steps, transition_jump_steps_max]`
3. the learned conditioning feature is the corresponding physical duration
   `transition_days`

Supported sampling distributions:
1. `uniform`
2. `loguniform`
3. `const`
4. `fixed`
5. `mixture_off_loguniform`

### 7.5 `normalization`
State fields support:
1. `none`
2. `log10`
3. `signed_log1p`

State normalization keys:
1. `field_transforms`
2. `zscore_eps`
3. `log10_eps`
4. `signed_log1p_scale`

Parameter normalization keys:
1. `mode`
2. `eps`

### 7.6 `model`
1. `grid`
   External spherical grid passed to SFNO. Must be `legendre-gauss`.
2. `grid_internal`
   Internal SFNO grid, also `legendre-gauss`.
3. `scale_factor`
   SFNO internal scaling factor.
4. `embed_dim`
   SFNO latent width.
5. `num_layers`
   Number of SFNO spectral blocks.
6. `encoder_layers`
   Desired pointwise encoder depth.
7. `activation_function`
   Pointwise activation choice.
8. `use_mlp`
   Whether SFNO block MLPs are enabled.
9. `mlp_ratio`
   Hidden width multiplier for MLPs.
10. `drop_rate`
    Dropout rate.
11. `drop_path_rate`
    Stochastic depth rate.
12. `normalization_layer`
    Normalization layer choice for SFNO internals.
13. `hard_thresholding_fraction`
    Spectral thresholding fraction.
14. `residual_prediction`
    Whether to predict residual updates on target channels.
15. `residual_init_scale`
    Initial learned residual scale.
16. `pos_embed`
    Positional embedding mode used by SFNO.
17. `bias`
    Whether pointwise layers use bias.
18. `include_coord_channels`
    Whether deterministic latitude/longitude sin/cos channels are concatenated.

### 7.7 `training`
1. `seed`
   Training seed.
2. `device`
   `auto`, `cpu`, or `cuda`.
3. `amp_mode`
   `none`, `bf16`, or `fp16`.
4. `optimizer`
   Active code uses `adamw`.
5. `epochs`
   Number of training epochs.
6. `batch_size`
   Batch size for processed samples.
7. `num_workers`
   Dataloader worker count.
8. `shuffle`
   Whether to shuffle the training dataset.
9. `pin_memory`
   Dataloader pinned-memory toggle.
10. `preload_to_gpu`
    Whether the processed splits are fully moved to GPU before training.
11. `learning_rate`
    Base optimizer learning rate.
12. `weight_decay`
    AdamW weight decay.
13. `val_fraction`
    File-level validation split fraction.
14. `test_fraction`
    File-level test split fraction.
15. `split_seed`
    Train/val/test split seed.
16. `scheduler`
    Scheduler configuration block.

Scheduler keys:
1. `type`
2. `warmup_epochs`
3. `factor`
4. `patience`
5. `min_lr`
6. `eps`

Active default scheduler behavior:
1. default `type` is `plateau`
2. warmup is linear for `warmup_epochs`
3. after warmup, validation loss is monitored with `ReduceLROnPlateau`
4. learning rate is multiplied by `factor` after `patience` epochs without an
   improvement larger than `eps`
5. if `min_lr` is omitted, it resolves to `learning_rate / 50` for active
   schedulers and to `0.0` for `type='none'`
6. cosine scheduling remains supported via `type='cosine_warmup'`

### 7.8 Common Artifact Variable Names
These names recur across raw files, processed metadata, checkpoints, and code:
1. `B`
   Batch size.
2. `T`
   Number of sampled transitions in one raw simulation window or processed shard.
3. `C`
   Channel count.
4. `H`
   Latitude dimension size in array layout.
5. `W`
   Longitude dimension size in array layout.
6. `nlat`
   Latitude grid size. Numerically equal to `H` for stored arrays.
7. `nlon`
   Longitude grid size. Numerically equal to `W` for stored arrays.
8. `input_fields`
   Ordered list of visible state channel names stored with a raw file or checkpoint.
9. `target_fields`
   Ordered list of target channel names stored with a raw file or checkpoint.
10. `param_names`
    Ordered list of canonical physical conditioning parameter names.
11. `conditioning_names`
    Ordered list of model-conditioning names. Active value:
    `param_names + ['transition_days']`.
12. `lat_order`
    Stored latitude ordering string. Active canonical value is `north_to_south`.
13. `lon_origin`
    Stored longitude-origin convention string. Active canonical value is `0_to_2pi`.
14. `lon_shift`
    Integer roll applied to longitude indices when converting from the native
    MY_SWAMP layout into the stored canonical layout.
15. `build_fingerprint`
    Preprocessing cache signature used to determine whether processed data is
    still valid for the current config and raw dataset.
16. `runtime_amp_mode`
    The actual AMP mode used at runtime after device/grid safety checks.

## 8. Data Contracts
### 8.1 Raw Simulation File Contract
Each raw file is stored as:
`sim_XXXXXX.npy`

The saved payload is a Python dict serialized by:
`np.save(..., allow_pickle=True)`

Exact required keys:
1. `state_inputs`
2. `state_targets`
3. `transition_days`
4. `anchor_steps`
5. `input_fields`
6. `target_fields`
7. `params`
8. `param_names`
9. `default_time_days`
10. `burn_in_days`
11. `dt_seconds`
12. `starttime_index`
13. `transition_jump_steps`
14. `anchor_stride_steps`
15. `n_transitions`
16. `M`
17. `nlat`
18. `nlon`
19. `lat_order`
20. `lon_origin`
21. `lon_shift`

Shapes and dtypes as written by generation:
1. `state_inputs`: `[T, 5, H, W]`, stored as `float64`
2. `state_targets`: `[T, 3, H, W]`, stored as `float64`
3. `transition_days`: `[T]`, stored as `float64`
4. `anchor_steps`: `[T]`, stored as `int64`
5. `input_fields`: `[5]`, object/string array
6. `target_fields`: `[3]`, object/string array
7. `params`: `[7]`, stored as `float64`
8. `param_names`: `[7]`, object/string array
9. scalar metadata values are stored as zero-dimensional NumPy scalars

Semantic contract:
1. `state_inputs[t]` is the visible physical state at the anchor step
2. `state_targets[t]` is the prognostic target at `anchor_steps[t] + transition_jump_steps`
3. `anchor_steps[t+1] - anchor_steps[t] = anchor_stride_steps = transition_jump_steps`
4. all channels are already canonicalized into the configured geometry
5. field ordering must exactly match the configured constants

### 8.2 Raw Manifest Contract
Generation also writes:
`manifest.json`

Top-level keys:
1. `created_unix`
2. `n_sims_requested`
3. `n_sims_written`
4. `input_fields`
5. `target_fields`
6. `param_names`
7. `dataset_dir`
8. `solver`
9. `sampling`
10. `geometry`
11. `items`

Each `items[]` entry includes:
1. `sim_idx`
2. `file`
3. `input_fields`
4. `target_fields`
5. `param_names`
6. `params`
7. `default_time_days`
8. `burn_in_days`
9. `dt_seconds`
10. `starttime_index`
11. `transition_jump_steps`
12. `anchor_stride_steps`
13. `n_transitions`
14. `anchor_step_start`
15. `anchor_step_end`
16. `transition_days_min`
17. `transition_days_max`
18. `M`
19. `nlat`
20. `nlon`
21. `lat_order`
22. `lon_origin`
23. `lon_shift`

The manifest is descriptive only. Training reads raw `.npy` files directly.

### 8.3 Processed Shard Contract
Processed data is written as one `.npz` shard per raw simulation file.

Each shard stores:
1. `state_inputs_norm`
2. `state_targets_norm`
3. `params_norm`
4. `conditioning_norm`
5. `transition_days`
6. `anchor_steps`

Shapes and dtypes:
1. `state_inputs_norm`: `[T, 5, H, W]`, `float32`
2. `state_targets_norm`: `[T, 3, H, W]`, `float32`
3. `params_norm`: `[7]`, `float32`
4. `conditioning_norm`: `[T, 8]`, `float32`
5. `transition_days`: `[T]`, `float64`
6. `anchor_steps`: `[T]`, `int64`

Semantic notes:
1. `conditioning_norm[t]` is the normalized concatenation of physical params
   and `transition_days[t]`
2. `params_norm` is retained separately for tooling that needs only the physical
   parameter vector
3. current training loaders do not consume `anchor_steps`; it is retained for
   traceability and future tooling
4. splitting is by raw simulation file, not by transition, to prevent leakage

### 8.4 Processed Metadata Contract
`processed_meta.json` stores:
1. `task`
2. `input_fields`
3. `target_fields`
4. `param_names`
5. `conditioning_names`
6. `input_shape`
7. `target_shape`
8. `splits`
9. `normalization`
10. `solver`
11. `sampling`
12. `geometry`
13. `build_fingerprint`

Important exact semantics:
1. `task` must be `trajectory_transition`
2. `conditioning_names` must equal `param_names + ['transition_days']`
3. `build_fingerprint` is the cache-reuse contract for processed data
4. `geometry` currently stores:
   `flip_latitude_to_north_south`, `roll_longitude_to_0_2pi`, `lat_order`,
   and `lon_origin`

### 8.5 Normalization Metadata Contract
Normalization JSON includes:
1. `input_state`
2. `target_state`
3. `params`
4. `transition_time`

Each state block contains:
1. `field_names`
2. `field_transforms`
3. `mean`
4. `std`
5. `zscore_eps`
6. `log10_eps`
7. `signed_log1p_scale`

The params block contains:
1. `param_names`
2. `mean`
3. `std`
4. `is_constant`
5. `zscore_eps`

The transition-time block contains the same fields, with
`param_names = ['transition_days']`.

### 8.6 Checkpoint Contract
`best.pt` and `last.pt` store:
1. `mode`
2. `model_state`
3. `input_fields`
4. `target_fields`
5. `param_names`
6. `conditioning_names`
7. `shape`
8. `geometry`
9. `normalization`
10. `solver`
11. `sampling`
12. `model_config`
13. `training_config`
14. `runtime_amp_mode`
15. `resolved_config`
16. `source_config_path`
17. `epoch`
18. `train_loss`
19. `val_loss`
20. `learning_rate`

These fields are part of the practical artifact contract because downstream
tools in `extra/` read them directly.

### 8.7 Training Output Contract
Training writes at least:
1. `best.pt`
2. `last.pt`
3. `training_history.json`
4. `training_history.csv`
5. `val_metrics.json`
6. `test_metrics.json`
7. `config_used.resolved.json`
8. `config_used.original.<suffix>`

### 8.8 Export Output Contract
TorchScript export writes:
1. `model_export.torchscript.pt`
2. `model_export.meta.json`

The export metadata contains:
1. `export_format`
2. `checkpoint_path`
3. `export_path`
4. `device`
5. `input`
6. `output`
7. `param_names`
8. `conditioning_names`
9. `normalization`
10. `verification`

## 9. Model And Evaluation Contract
### 9.1 Model Input And Output
The active model call is:
`model(state0, conditioning)`

Shapes:
1. `state0`: `[B, 5, H, W]`
2. `conditioning`: `[B, 8]`
3. output: `[B, 3, H, W]`

The physical-space export wrapper takes:
`export_model(state0, params, transition_days)`

Residual prediction uses the aligned input channels corresponding to the target
fields:
`Phi`, `eta`, `delta`

### 9.2 One-Step Metrics
Required single-call metrics, stored under the historical artifact key
`one_step`:
1. normalized-space global RMSE
2. normalized-space per-channel RMSE
3. physical-space global RMSE
4. physical-space per-channel RMSE
5. physical-space per-channel MAE
6. spherical power-spectrum mismatch using `RealSHT`

### 9.3 Rollout Metrics
Rollout evaluation:
1. feeds predicted full visible states back into the next step
2. reconstructs `U,V` from predicted prognostics at every step
3. compares both prognostic targets and full visible states
4. is valid because generation stores anchor chains with stride equal to the
   modeled jump duration

Reported rollout groups:
1. `prognostic_physical`
2. `full_state_physical`

## 10. Style, Vectorization, And Documentation Preferences
### 10.1 Numerical Style
Preferred style:
1. vectorized NumPy preprocessing
2. batched JAX generation
3. batched Torch training/inference
4. no Python loops over pixels or channels when tensor ops are available

### 10.2 Documentation Style
Required style:
1. docstrings on public modules, classes, and non-trivial functions
2. explicit type hints
3. concise comments only for non-obvious logic
4. shape-aware reasoning in code and tests

### 10.3 Code Cleanliness
Required hygiene:
1. no dead feature flags that are silently ignored
2. no stale artifact keys in the spec
3. no unused public plumbing carried across stages without a consumer
4. prefer explicit exact contracts over "best effort" convenience behavior

## 11. Local Integrity Checks
Run all commands in this section from the repository root inside
`swamp_compare`.

The minimum local integrity pass for this repository is:
1. `python -m pytest unit_tests`
2. `python -m compileall src extra unit_tests`

For code cleanliness, run these developer checks when the tools are available:
1. `ruff check src extra unit_tests`
2. `vulture src extra unit_tests --min-confidence 80`

In the standard local `swamp_compare` conda environment, both `ruff` and
`vulture` are installed. `ruff` must be invoked with a subcommand such as
`check`, and `vulture` must be invoked with one or more paths.

`ruff` and `vulture` are development-time quality gates, not runtime
dependencies.

## 12. Dependency Contract
Required core dependencies:
1. `numpy`
2. `torch`
3. `torch-harmonics==0.8.1` as the preferred baseline, imported as
   `torch_harmonics`

Required generation/physics dependency:
1. `my_swamp`

Optional plotting dependency:
1. `matplotlib`

The contract assumes these packages are importable in the active
`swamp_compare` environment.

## 13. Exactness Statement
This repository intentionally models the visible MY_SWAMP flow map, not the
exact discrete solver carry.

That means:
1. no learned previous-step carry in the network state
2. prognostic-only direct targets in the current baseline
3. deterministic reconstruction of wind channels when a full five-field state
   is needed
4. approximate surrogate behavior rather than solver identity

The emulator should be evaluated as a physically informed surrogate, not as an
exact symbolic replacement for MY_SWAMP internals.
