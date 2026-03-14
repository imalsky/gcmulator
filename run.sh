#!/usr/bin/env bash
#SBATCH -J gcmulator
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -t 08:00:00

set -euo pipefail
#small change
# Resolve repository root from script location for reproducible relative paths.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-swamp_compare}"
CONFIG_PATH="${CONFIG_PATH:-config.json}"
MAIN_MODULE="${MAIN_MODULE:-gcmulator}"
RUN_GEN_IF_MISSING="${RUN_GEN_IF_MISSING:-1}"
MY_SWAMP_PACKAGE_SPEC="${MY_SWAMP_PACKAGE_SPEC:-my-swamp}"
MY_SWAMP_PIP_ARGS="${MY_SWAMP_PIP_ARGS:---index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/}"
TORCH_HARMONICS_PACKAGE_SPEC="${TORCH_HARMONICS_PACKAGE_SPEC:-torch-harmonics==0.8.1}"
TORCH_HARMONICS_PIP_ARGS="${TORCH_HARMONICS_PIP_ARGS:---no-deps --no-build-isolation}"
TORCH_HARMONICS_FORCE_CPU_BUILD="${TORCH_HARMONICS_FORCE_CPU_BUILD:-1}"
TORCH_HARMONICS_FORCE_REINSTALL="${TORCH_HARMONICS_FORCE_REINSTALL:-0}"
read -r -a MY_SWAMP_PIP_ARGS_ARR <<< "${MY_SWAMP_PIP_ARGS}"
read -r -a TORCH_HARMONICS_PIP_ARGS_ARR <<< "${TORCH_HARMONICS_PIP_ARGS}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH."
  exit 1
fi
# Activate user-selected environment before any imports/pip operations.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
export GCMULATOR_SUPPRESS_KNOWN_WARNINGS="${GCMULATOR_SUPPRESS_KNOWN_WARNINGS:-1}"
export GCMULATOR_JAX_SIM_BATCH="${GCMULATOR_JAX_SIM_BATCH:-1}"
export GCMULATOR_JAX_SIM_BATCH_AUTO_GPU="${GCMULATOR_JAX_SIM_BATCH_AUTO_GPU:-1}"
export SWAMPE_JAX_ENABLE_X64="${SWAMPE_JAX_ENABLE_X64:-1}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-${SWAMPE_JAX_ENABLE_X64}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot verify GPU allocation."
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: No visible NVIDIA GPU in this Slurm job."
    exit 1
  fi
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi
echo "Config path: $(python - <<'PY' "$CONFIG_PATH"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
echo "Launcher defaults: GCMULATOR_JAX_SIM_BATCH=${GCMULATOR_JAX_SIM_BATCH} GCMULATOR_JAX_SIM_BATCH_AUTO_GPU=${GCMULATOR_JAX_SIM_BATCH_AUTO_GPU} SWAMPE_JAX_ENABLE_X64=${SWAMPE_JAX_ENABLE_X64} JAX_ENABLE_X64=${JAX_ENABLE_X64}"
if [ ! -d "$PROJECT_ROOT/src/gcmulator" ]; then
  echo "ERROR: package directory not found: $PROJECT_ROOT/src/gcmulator"
  exit 1
fi

# Always refresh my_swamp from package source.
echo "Reinstalling my_swamp package: ${MY_SWAMP_PACKAGE_SPEC}"
python -m pip uninstall -y my_swamp my-swamp >/dev/null 2>&1 || true
python -m pip install --upgrade --no-deps "${MY_SWAMP_PIP_ARGS_ARR[@]}" "${MY_SWAMP_PACKAGE_SPEC}"
if ! python - <<'PY' >/dev/null 2>&1
import my_swamp  # noqa: F401
PY
then
  echo "ERROR: my_swamp import failed after package install (${MY_SWAMP_PACKAGE_SPEC})"
  exit 1
fi

if [ "${TORCH_HARMONICS_FORCE_REINSTALL}" != "1" ] && python - <<'PY' >/dev/null 2>&1
import torch_harmonics  # noqa: F401
PY
then
  echo "Using existing torch_harmonics install; skipping reinstall."
else
  if [[ "${TORCH_HARMONICS_PACKAGE_SPEC}" == git+* ]] && ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required to install TORCH_HARMONICS_PACKAGE_SPEC='${TORCH_HARMONICS_PACKAGE_SPEC}'."
    exit 1
  fi
  echo "Reinstalling torch_harmonics package: ${TORCH_HARMONICS_PACKAGE_SPEC}"
  python -m pip uninstall -y torch_harmonics torch-harmonics >/dev/null 2>&1 || true
  if [ "${TORCH_HARMONICS_FORCE_CPU_BUILD}" = "1" ]; then
    TH_INSTALL_PREFIX=(env CUDA_VISIBLE_DEVICES= FORCE_CUDA_EXTENSION=0)
  else
    TH_INSTALL_PREFIX=()
  fi
  "${TH_INSTALL_PREFIX[@]}" python -m pip install --upgrade "${TORCH_HARMONICS_PIP_ARGS_ARR[@]}" "${TORCH_HARMONICS_PACKAGE_SPEC}"
  if ! python - <<'PY' >/dev/null 2>&1
import torch_harmonics  # noqa: F401
PY
  then
    echo "ERROR: torch_harmonics import failed after package install (${TORCH_HARMONICS_PACKAGE_SPEC})"
    exit 1
  fi
fi

if [ "$RUN_GEN_IF_MISSING" = "1" ]; then
  # Resolve dataset path through config parser so relative path semantics match runtime code.
  mapfile -t DATA_CFG < <(python - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
from gcmulator.config import load_config, resolve_path

config_path = Path(sys.argv[1]).resolve()
cfg = load_config(config_path)
print(resolve_path(config_path, cfg.paths.dataset_dir))
print(int(cfg.sampling.n_sims))
PY
  )
  DATASET_DIR="${DATA_CFG[0]}"
  EXPECTED_SIMS="${DATA_CFG[1]}"
  SIM_COUNT=0
  if [ -d "$DATASET_DIR" ]; then
    SIM_COUNT="$(find "$DATASET_DIR" -maxdepth 1 -type f \( -name 'sim_*.npy' -o -name 'sim_*.npz' \) | wc -l | tr -d ' ')"
  fi
  if [ "$SIM_COUNT" -eq 0 ]; then
    # Generate only when raw trajectory transition files are not already present.
    python -m "$MAIN_MODULE" --gen --config "$CONFIG_PATH"
  elif [ "$SIM_COUNT" -ne "$EXPECTED_SIMS" ]; then
    echo "ERROR: dataset file count mismatch: found $SIM_COUNT files in $DATASET_DIR, expected $EXPECTED_SIMS from config.sampling.n_sims."
    echo "Set paths.overwrite_dataset=true and regenerate, or align config.sampling.n_sims with existing data."
    exit 1
  fi
fi

# Train using existing or newly generated dataset.
python -m "$MAIN_MODULE" --train --config "$CONFIG_PATH"
