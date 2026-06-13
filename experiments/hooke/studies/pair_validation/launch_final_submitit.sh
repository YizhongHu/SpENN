#!/usr/bin/env bash
# =============================================================================
# Hydra Submitit launcher for Hooke pair final benchmark jobs
# =============================================================================
#
# Final benchmark launch is intentionally split into two phases:
#
#   STAGE=final_train bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh
#   STAGE=final_eval  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh
#
# `final_eval` should be launched only after final training checkpoints exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:-experiments/hooke/studies/pair_validation/manifest.yaml}"
INPUTS="${INPUTS:-experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv}"
LAUNCHER="experiments/hooke/studies/pair_validation/launch_final_submitit.py"
DEVICE="${DEVICE:-cuda}"
STAGE="${STAGE:-final_train}"
HYDRA_LAUNCHER="${HYDRA_LAUNCHER:-submitit_slurm}"

hydra_value() {
  local value="$1"
  value="${value//\\/\\\\}"
  printf '%s' "${value//,/\\,}"
}

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

case "$STAGE" in
  final_train|final_eval)
    ;;
  *)
    echo "Unsupported STAGE=${STAGE}; expected 'final_train' or 'final_eval'." >&2
    exit 2
    ;;
esac

export UV_PROJECT_ENVIRONMENT="$VENV"

RUN_SUFFIX="${RUN_SUFFIX:-}"
if [[ "$(basename "$INPUTS")" == "final_smoke_inputs.csv" ]]; then
  RUN_SUFFIX_FILE="${RUN_SUFFIX_FILE:-slurm_logs/hooke_pair_final_v1/smoke_${DEVICE}.run_suffix}"
  mkdir -p "$(dirname "$RUN_SUFFIX_FILE")"
  if [[ -z "$RUN_SUFFIX" ]]; then
    if [[ "$STAGE" == "final_eval" ]]; then
      if [[ -s "$RUN_SUFFIX_FILE" ]]; then
        RUN_SUFFIX="$(<"$RUN_SUFFIX_FILE")"
      else
        echo "No smoke RUN_SUFFIX found for final_eval. Run final_train first or set RUN_SUFFIX explicitly." >&2
        echo "Expected suffix file: ${RUN_SUFFIX_FILE}" >&2
        exit 2
      fi
    else
      GIT_HASH="$(git rev-parse --short=8 HEAD 2>/dev/null || printf 'nogit')"
      RUN_STAMP="$(TZ=America/New_York date +%Y%m%d_%H%M%S)"
      RUN_SUFFIX="${DEVICE}_${RUN_STAMP}_${GIT_HASH}"
    fi
  fi
  printf '%s\n' "$RUN_SUFFIX" > "$RUN_SUFFIX_FILE"
fi

SYNC_EXTRAS=(--extra "$EXTRA" --extra submitit)
if [[ -n "${SPENN_EXTRA_EXTRAS:-}" ]]; then
  for extra in $SPENN_EXTRA_EXTRAS; do
    SYNC_EXTRAS+=(--extra "$extra")
  done
fi

uv sync "${SYNC_EXTRAS[@]}"
source "$VENV/bin/activate"

JOB_INDEX_SWEEP="$(python "$LAUNCHER" --inputs "$INPUTS" --stage "$STAGE" --print-job-index-sweep)"
MANIFEST_HYDRA_OVERRIDES=()
if [[ "$HYDRA_LAUNCHER" == "submitit_slurm" ]]; then
  mapfile -t MANIFEST_HYDRA_OVERRIDES < <(
    python "$LAUNCHER" \
      --manifest "$MANIFEST" \
      --inputs "$INPUTS" \
      --stage "$STAGE" \
      --device "$DEVICE" \
      --print-hydra-overrides
  )
else
  MANIFEST_HYDRA_OVERRIDES+=("hydra.sweep.dir=${HYDRA_SWEEP_DIR:-/tmp/rhu/spenn_final_submitit_local}/${STAGE}")
fi

HYDRA_OVERRIDES=(
  "hydra/launcher=${HYDRA_LAUNCHER}"
  "${MANIFEST_HYDRA_OVERRIDES[@]}"
)

if [[ "$HYDRA_LAUNCHER" == "submitit_slurm" ]]; then
  HYDRA_OVERRIDES+=("hydra.launcher.setup=[\"cd ${REPO_ROOT}\",\"source ${REPO_ROOT}/${VENV}/bin/activate\"]")
fi

if [[ "$HYDRA_LAUNCHER" == "submitit_slurm" ]]; then
  [[ -n "${PARTITION:-}" ]] && HYDRA_OVERRIDES+=("hydra.launcher.partition=$(hydra_value "$PARTITION")")
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

echo "manifest=${MANIFEST}"
echo "inputs=${INPUTS}"
echo "stage=${STAGE}"
echo "device=${DEVICE}"
echo "venv=${VENV}"
echo "hydra_launcher=${HYDRA_LAUNCHER}"
echo "job_index=${JOB_INDEX_SWEEP}"
echo "run_suffix=${RUN_SUFFIX:-<none>}"

LAUNCH_OVERRIDES=(
  "inputs=${INPUTS}"
  "stage=${STAGE}"
  "device=${DEVICE}"
  "job_index=${JOB_INDEX_SWEEP}"
)
if [[ -n "$RUN_SUFFIX" ]]; then
  LAUNCH_OVERRIDES+=("run_suffix=${RUN_SUFFIX}")
fi

HYDRA_FULL_ERROR=1 python "$LAUNCHER" \
  --multirun \
  "${LAUNCH_OVERRIDES[@]}" \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
