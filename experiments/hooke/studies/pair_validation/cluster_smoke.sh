#!/usr/bin/env bash
# Submit one-job Slurm smokes for the Hooke pair-validation launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-both}"
SMOKE_MANIFEST="${SMOKE_MANIFEST:-experiments/hooke/studies/pair_validation/smoke_manifest.yaml}"
RUN_ROOT_PREFIX="${RUN_ROOT_PREFIX:-outputs/hooke_pair_validation_smoke}"
SMOKE_CPU_PARTITION="${SMOKE_CPU_PARTITION:-test}"
SMOKE_GPU_PARTITION="${SMOKE_GPU_PARTITION:-gpu_test}"
SMOKE_TIMEOUT_MIN="${SMOKE_TIMEOUT_MIN:-15}"
DRY_RUN="${DRY_RUN:-false}"
EXTRA_OVERRIDES=()

usage() {
  cat <<'USAGE'
Usage:
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh [--device cpu|cuda|both] [--dry-run] [-- HYDRA_OVERRIDES...]

Examples:
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh --device cpu
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh --device cuda
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh --dry-run

The default submits both CPU and GPU one-job Slurm execution smokes.
CPU smoke defaults to partition `test`; GPU smoke defaults to `gpu_test`.
Each smoke uses a 15-minute Slurm timeout unless SMOKE_TIMEOUT_MIN is set.
USAGE
}

while (($#)); do
  case "$1" in
    --device)
      if (($# < 2)); then
        echo "--device requires one of: cpu, cuda, both" >&2
        exit 2
      fi
      DEVICE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#--device=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --execute)
      DRY_RUN=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_OVERRIDES=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$DEVICE" in
  cpu)
    DEVICES=(cpu)
    ;;
  cuda|gpu)
    DEVICES=(cuda)
    ;;
  both)
    DEVICES=(cpu cuda)
    ;;
  *)
    echo "Unsupported device: ${DEVICE}; expected cpu, cuda, or both." >&2
    exit 2
    ;;
esac

for device in "${DEVICES[@]}"; do
  label="$device"
  partition="$SMOKE_CPU_PARTITION"
  if [[ "$device" == "cuda" ]]; then
    label="gpu"
    partition="$SMOKE_GPU_PARTITION"
  fi

  echo "Submitting ${device} validation smoke job to partition ${partition}..."
  DEVICE="$device" \
    MANIFEST="$SMOKE_MANIFEST" \
    HYDRA_LAUNCHER=submitit_slurm \
    ARRAY_PARALLELISM=1 \
    PARTITION="$partition" \
    TIMEOUT_MIN="$SMOKE_TIMEOUT_MIN" \
    RUN_ROOT="${RUN_ROOT_PREFIX}_${label}" \
    bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
    dry_run="${DRY_RUN}" \
    job_index=0 \
    "${EXTRA_OVERRIDES[@]}"
done
