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

# Resolve repository root from script location for reproducible relative paths.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-swamp_compare}"
CONFIG_PATH="${CONFIG_PATH:-config.json}"
MAIN_PY="${MAIN_PY:-src/main.py}"
RUN_GEN_IF_MISSING="${RUN_GEN_IF_MISSING:-1}"
MY_SWAMP_PACKAGE_SPEC="${MY_SWAMP_PACKAGE_SPEC:-my-swamp}"
MY_SWAMP_PIP_ARGS="${MY_SWAMP_PIP_ARGS:---index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/}"
MY_SWAMP_EXTRA_PACKAGES="${MY_SWAMP_EXTRA_PACKAGES:-jax[cuda13]}"
read -r -a MY_SWAMP_PIP_ARGS_ARR <<< "${MY_SWAMP_PIP_ARGS}"
read -r -a MY_SWAMP_EXTRA_PACKAGES_ARR <<< "${MY_SWAMP_EXTRA_PACKAGES}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH."
  exit 1
fi
# Activate user-selected environment before any imports/pip operations.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
export GCMULATOR_SUPPRESS_KNOWN_WARNINGS="${GCMULATOR_SUPPRESS_KNOWN_WARNINGS:-1}"

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
if [ ! -f "$MAIN_PY" ]; then
  echo "ERROR: main script not found: $MAIN_PY"
  exit 1
fi

# Always refresh my_swamp from package source.
echo "Reinstalling my_swamp package: ${MY_SWAMP_PACKAGE_SPEC}"
python -m pip uninstall -y my_swamp my-swamp >/dev/null 2>&1 || true
python -m pip install --no-cache-dir --upgrade --no-deps "${MY_SWAMP_PIP_ARGS_ARR[@]}" "${MY_SWAMP_PACKAGE_SPEC}"
if [ "${#MY_SWAMP_EXTRA_PACKAGES_ARR[@]}" -gt 0 ] && [ -n "${MY_SWAMP_EXTRA_PACKAGES_ARR[0]}" ]; then
  echo "Installing explicit runtime dependencies: ${MY_SWAMP_EXTRA_PACKAGES_ARR[*]}"
  python -m pip install --no-cache-dir --upgrade "${MY_SWAMP_PIP_ARGS_ARR[@]}" "${MY_SWAMP_EXTRA_PACKAGES_ARR[@]}"
fi
if ! python - <<'PY' >/dev/null 2>&1
import my_swamp  # noqa: F401
PY
then
  echo "ERROR: my_swamp import failed after package install (${MY_SWAMP_PACKAGE_SPEC})"
  exit 1
fi

if [ "$RUN_GEN_IF_MISSING" = "1" ]; then
  # Resolve dataset path through config parser so relative path semantics match runtime code.
  DATASET_DIR="$({ python - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
from src.config import load_config, resolve_path

config_path = Path(sys.argv[1]).resolve()
cfg = load_config(config_path)
print(resolve_path(config_path, cfg.paths.dataset_dir))
PY
  })"
  SIM_COUNT=0
  if [ -d "$DATASET_DIR" ]; then
    SIM_COUNT="$(find "$DATASET_DIR" -maxdepth 1 -type f -name 'sim_*.npz' | wc -l | tr -d ' ')"
  fi
  if [ "$SIM_COUNT" -eq 0 ]; then
    # Generate only when raw terminal states are not already present.
    python "$MAIN_PY" --gen --config "$CONFIG_PATH"
  fi
fi

# Train using existing or newly generated dataset.
python "$MAIN_PY" --train --config "$CONFIG_PATH"
