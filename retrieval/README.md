# Retrieval Contract

This folder now contains two pieces:

1. a strict artifact-contract check for the current direct-jump training design
2. a batched Torch runtime for many-sample surrogate inference when the export is valid

The important distinction is:

- A valid direct-jump Torch artifact is enough for fast batched surrogate calls on
  one GPU.
- It is still **not** enough for the originally requested fully GPU-native
  `jaxoplanet` retrieval unless the surrogate itself is exported in a JAX-native
  runtime.

## What changed

The retrieval code now enforces the current time-jump contract instead of
silently tolerating stale checkpoints:

1. The checkpoint must include `normalization.transition_time`.
2. That transition-time feature must be named `log10_transition_days`.
3. The conditioning vector must be
   `param_names + ["log10_transition_days"]`.
4. The export forward signature must accept
   `(state0, params, transition_days)`.
5. The checkpoint sampling metadata must record
   `transition_jump_days_min/max`.

That matches the current training/export methodology:

- raw data stores physical `transition_days`
- conditioning is learned in log space through `log10_transition_days`
- the export accepts physical `transition_days` and normalizes them internally

## Files

- [`surrogate_backend.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/surrogate_backend.py)
  inspects artifacts and provides `TorchSurrogateRuntime`, a batched inference
  wrapper for valid direct-jump Torch exports.
- [`run_surrogate_nss.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/run_surrogate_nss.py)
  has a small config block at the top, writes
  [`artifact_readiness.json`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/artifact_readiness.json),
  and optionally benchmarks the Torch runtime when the artifact is valid.

## GPU usage

For a valid direct-jump Torch artifact, the runtime is set up for throughput:

1. It batches likelihood-side surrogate calls with `max_batch_size`.
2. On CUDA it uses pinned host memory for CPU-to-GPU transfers.
3. State tensors can be moved in channels-last format for conv-heavy inference.
4. TF32 can be enabled for faster inference on recent NVIDIA GPUs.

Those controls live in the `SurrogateRuntimeConfig` nested inside the config
block at the top of
[`run_surrogate_nss.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/run_surrogate_nss.py).

## Current blocker

The checked-in artifact under
[`models/v1`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/models/v1)
is still stale relative to the new methodology:

1. Its checkpoint does not contain `transition_time`.
2. Its conditioning vector does not include `log10_transition_days`.
3. Its export signature is fixed-step instead of direct-jump.
4. Its sampling metadata still lacks the day-valued jump range.

So the repository now fails for the right reasons, with the right report, while
still providing the runtime you need once a fresh direct-jump export exists.

## Practical use

1. Leave `require_gpu_native_jax = true` if you want the runner to enforce the
   full intended `jaxoplanet` architecture.
2. Set `require_gpu_native_jax = false` if you want to smoke-test a valid
   direct-jump Torch artifact before the JAX-native surrogate export exists.
