# Retrieval Status

This folder currently contains a contract check, not a runnable retrieval.

That is intentional. The checked-in surrogate artifact under
[`gcmulator/models/v1`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/models/v1)
does not satisfy the requirements for the retrieval you asked for:

1. It is a PyTorch/TorchScript export, while the requested light-curve stage is
   JAX / `jaxoplanet`.
2. Its forward signature is fixed-step, not direct-jump. The export takes
   `(state0, params)` instead of `(state0, params, transition_days)`.
3. The saved checkpoint conditions on the 7 physical parameters only. It does
   not include the current `transition_time` normalization block or the
   `log10_transition_days` conditioning feature.
4. The saved sampling metadata does not record a variable forward-only jump
   range, so the artifact is stale relative to the current flexible-jump
   training contract.
5. With `dt_seconds = 120`, a `100` day state would require about `72,000`
   recurrent surrogate steps per likelihood call.

Those four facts together mean the current artifact cannot support a fast,
GPU-native nested-sampling retrieval through `jaxoplanet`.

## What is in this folder

- [`surrogate_backend.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/surrogate_backend.py)
  inspects the saved export and checkpoint and reports whether they are ready
  for the requested retrieval contract.
- [`run_surrogate_nss.py`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/run_surrogate_nss.py)
  runs that inspection, writes
  [`artifact_readiness.json`](/Users/imalsky/Desktop/SWAMPE_Project/gcmulator/retrieval/artifact_readiness.json),
  and then fails fast with the blocker list.

## Why the current artifact is not enough

The earlier retrieval draft in this folder tried to bridge the Torch surrogate
into JAX with callbacks. That path was removed because it violates the current
requirements:

1. A callback bridge is not GPU-native.
2. A callback bridge still leaves the model fixed-step, so long-horizon
   retrieval remains too slow.
3. Keeping dead or misleading code here would make the repository contract
   worse, not better.

## What has to change outside this folder

The retrieval itself can only become fast once the saved model artifact changes.
The important changes are outside `retrieval/`:

1. Train a variable-jump surrogate that explicitly conditions on
   `transition_days`, ideally in log-space via `log10_transition_days`.
2. Export a direct-jump artifact whose runtime signature includes
   `transition_days`.
3. For the strict `jaxoplanet` requirement, export or port the surrogate to a
   JAX-native runtime so the entire likelihood stays inside one compiled graph.
4. Include long-horizon or equilibrium-target pairs in training so the model can
   learn trajectories starting from the initial state and relaxing toward
   equilibrium.
5. Store the variable-jump day range and transition-time normalization stats in
   the checkpoint/export metadata so retrieval can validate the artifact
   contract before running.

## Practical implication

If you want a real retrieval now, one of these must happen first:

1. Provide a JAX-native direct-jump surrogate artifact.
2. Retrain/export the current surrogate so it is direct-jump and time-aware,
   then decide whether the light-curve stage must remain in JAX.

Until then, the best thing the retrieval folder can do is fail fast and report
the exact blockers instead of advertising a retrieval path that will be slow or
architecturally wrong.
