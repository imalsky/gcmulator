# Retrieval Contract

This folder contains the active surrogate retrieval path for now:

1. strict contract inspection for the exported direct-jump artifact
2. a batched Torch runtime that runs on either `cpu` or `gpu`

The retrieval runtime is intentionally centered on the exported bundle, not the
training checkpoint. The checkpoint is optional and is only used when you want
an extra export-versus-checkpoint consistency check.

## Direct-Jump Contract

The current training and export methodology is:

1. raw data stores physical `transition_days`
2. conditioning is learned in log space through `log10_transition_days`
3. the export accepts physical `state0`, physical `params`, and physical
   `transition_days`
4. the export normalizes those inputs internally and returns physical `state1`

For a retrieval artifact to be valid, the export bundle must satisfy all of the
following:

1. `model_export.meta.json` must exist next to the TorchScript file
2. `artifact_kind` must be `direct_jump_physical_state_transition`
3. `supported_devices` must be exactly `["cpu", "gpu"]`
4. `physical_io` must mark `state0`, `params`, `transition_days`, and `state1`
   as physical-space tensors
5. `conditioning_names` must equal
   `param_names + ["log10_transition_days"]`
6. `normalization.transition_time.param_names` must be
   `["log10_transition_days"]`
7. `sampling.transition_jump_days_min/max` must be present
8. the exported forward signature must accept
   `(state0, params, transition_days)`

## Files

- [`surrogate_backend.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/surrogate_backend.py)
  inspects the export bundle and provides `TorchSurrogateRuntime`, a batched
  inference wrapper for valid direct-jump artifacts.
- [`run_surrogate_nss.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/run_surrogate_nss.py)
  exposes a small config block at the top, writes a JSON readiness report, and
  can benchmark the optimized runtime on representative inputs.

## Export Bundle

The export script writes:

1. `model_export.torchscript.pt`
2. `model_export.meta.json`

The TorchScript file already embeds the normalization tensors needed to accept
physical inputs directly. The JSON file repeats the same normalization contract
so retrieval code can validate the artifact and construct example physical-space
inputs without reaching back into training code.

## Runtime Behavior

`TorchSurrogateRuntime` uses the exported TorchScript model directly and applies
`torch.jit.optimize_for_inference()` on load when possible.

For throughput, the runtime:

1. batches calls with `max_batch_size`
2. uses pinned host memory for CPU-to-GPU transfers when enabled
3. moves state tensors in channels-last layout when enabled
4. enables TF32 on CUDA when requested

Those settings live in `SurrogateRuntimeConfig`, which sits inside the config
block near the top of
[`run_surrogate_nss.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/run_surrogate_nss.py).

## Practical Use

1. Export a fresh model with
   [`pytorch_export.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/extra/pytorch_export.py).
2. Edit the config block at the top of
   [`run_surrogate_nss.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/run_surrogate_nss.py).
3. Pick `runtime.device_mode = "cpu"` or `"gpu"`.
4. Run the script to write the readiness report and optional smoke benchmark.
