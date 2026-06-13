#!/usr/bin/env bash
# =============================================================================
# Hydra Submitit launcher for the Hooke pair validation scan (study v1)
# =============================================================================
#
# This script is intentionally a thin environment/setup wrapper. The study
# manifest owns the train config, study name, grid axes, run root, and default
# Slurm resources; launch_submitit.py turns that manifest into a Hydra Submitit
# multirun.
#
# Submit from the repo root or from this file's directory:
#
#   bash experiments/hooke/studies/pair_validation/launch_array.sh
#
# CPU example:
#
#   DEVICE=cpu bash experiments/hooke/studies/pair_validation/launch_array.sh
#
# Extra Hydra overrides can be appended after `--`, for example:
#
#   bash experiments/hooke/studies/pair_validation/launch_array.sh -- dry_run=true
#
# Keep slurm_logs/ around for reproducibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:-experiments/hooke/studies/pair_validation/manifest.yaml}"
LAUNCHER="experiments/hooke/studies/pair_validation/launch_submitit.py"
DEVICE="${DEVICE:-cuda}"
HYDRA_LAUNCHER="${HYDRA_LAUNCHER:-submitit_slurm}"

case "$DEVICE" in
  cuda)
    VENV="${VENV:-.venv-gpu}"
    EXTRA="${EXTRA:-cu126}"
    ;;
  cpu)
    VENV="${VENV:-.venv}"
    EXTRA="${EXTRA:-cpu}"
    ;;
  *)
    echo "Unsupported DEVICE=${DEVICE}; expected 'cuda' or 'cpu'." >&2
    exit 2
    ;;
esac

export UV_PROJECT_ENVIRONMENT="$VENV"

SYNC_EXTRAS=(--extra "$EXTRA" --extra submitit)
if [[ -n "${SPENN_EXTRA_EXTRAS:-}" ]]; then
  # Space-separated extra names, e.g. SPENN_EXTRA_EXTRAS="wandb".
  for extra in $SPENN_EXTRA_EXTRAS; do
    SYNC_EXTRAS+=(--extra "$extra")
  done
fi

uv sync "${SYNC_EXTRAS[@]}"
source "$VENV/bin/activate"

JOB_INDEX_SWEEP="$(python "$LAUNCHER" --manifest "$MANIFEST" --print-job-index-sweep)"
MANIFEST_HYDRA_OVERRIDES=()
if [[ "$HYDRA_LAUNCHER" == "submitit_slurm" ]]; then
  mapfile -t MANIFEST_HYDRA_OVERRIDES < <(
    python "$LAUNCHER" --manifest "$MANIFEST" --device "$DEVICE" --print-hydra-overrides
  )
else
  MANIFEST_HYDRA_OVERRIDES+=("hydra.sweep.dir=${HYDRA_SWEEP_DIR:-/tmp/rhu/spenn_submitit_local}")
fi

HYDRA_OVERRIDES=(
  "hydra/launcher=${HYDRA_LAUNCHER}"
  "${MANIFEST_HYDRA_OVERRIDES[@]}"
)

if [[ "$HYDRA_LAUNCHER" == "submitit_slurm" ]]; then
  HYDRA_OVERRIDES+=("hydra.launcher.setup=[\"cd ${REPO_ROOT}\",\"source ${REPO_ROOT}/${VENV}/bin/activate\"]")
fi

# Optional one-off resource overrides without editing the manifest.
if [[ "$HYDRA_LAUNCHER" == "submitit_slurm" ]]; then
  [[ -n "${PARTITION:-}" ]] && HYDRA_OVERRIDES+=("hydra.launcher.partition=${PARTITION}")
  if [[ "${GRES+x}" == "x" ]]; then
    if [[ -n "$GRES" ]]; then
      HYDRA_OVERRIDES+=("hydra.launcher.gres=${GRES}")
    else
      HYDRA_OVERRIDES+=("hydra.launcher.gres=null")
    fi
  fi
  [[ -n "${ARRAY_PARALLELISM:-}" ]] && HYDRA_OVERRIDES+=("hydra.launcher.array_parallelism=${ARRAY_PARALLELISM}")
fi
[[ -n "${CPUS_PER_TASK:-}" ]] && HYDRA_OVERRIDES+=("hydra.launcher.cpus_per_task=${CPUS_PER_TASK}")
[[ -n "${MEM_GB:-}" ]] && HYDRA_OVERRIDES+=("hydra.launcher.mem_gb=${MEM_GB}")
[[ -n "${TIMEOUT_MIN:-}" ]] && HYDRA_OVERRIDES+=("hydra.launcher.timeout_min=${TIMEOUT_MIN}")
[[ -n "${RUN_ROOT:-}" ]] && HYDRA_OVERRIDES+=("run_root=${RUN_ROOT}")

echo "manifest=${MANIFEST}"
echo "device=${DEVICE}"
echo "venv=${VENV}"
echo "hydra_launcher=${HYDRA_LAUNCHER}"
echo "job_index=${JOB_INDEX_SWEEP}"

HYDRA_FULL_ERROR=1 python "$LAUNCHER" \
  --multirun \
  "manifest=${MANIFEST}" \
  "device=${DEVICE}" \
  "job_index=${JOB_INDEX_SWEEP}" \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
