#!/usr/bin/env bash
# Submit one-job Slurm dry runs for the Hooke pair-validation launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-both}"
RUN_ROOT_PREFIX="${RUN_ROOT_PREFIX:-outputs/hooke_pair_validation_v1}"
EXTRA_OVERRIDES=()

usage() {
  cat <<'USAGE'
Usage:
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh [--device cpu|cuda|both] [-- HYDRA_OVERRIDES...]

Examples:
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh --device cpu
  bash experiments/hooke/studies/pair_validation/cluster_smoke.sh --device cuda

The default submits both CPU and GPU one-job Slurm dry runs.
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
  if [[ "$device" == "cuda" ]]; then
    label="gpu"
  fi

  echo "Submitting ${device} validation smoke job..."
  DEVICE="$device" \
    HYDRA_LAUNCHER=submitit_slurm \
    ARRAY_PARALLELISM=1 \
    RUN_ROOT="${RUN_ROOT_PREFIX}_${label}_smoke" \
    bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
    dry_run=true \
    job_index=0 \
    "${EXTRA_OVERRIDES[@]}"
done
