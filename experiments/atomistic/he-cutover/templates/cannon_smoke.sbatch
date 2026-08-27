#!/bin/bash
#SBATCH --job-name=p2-cutover
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
# a100_mig smoke stratum
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
: "${TPEN_CHECKOUT:?}"
: "${TPEN_RESULTS_ROOT:?}"
: "${TPEN_PLAN_ATTEMPT_ID:?}"
: "${TPEN_UV:?absolute uv executable required}"
mkdir -p "${TPEN_RESULTS_ROOT:?}/scheduler"
exec >"${TPEN_RESULTS_ROOT:?}/scheduler/slurm-${SLURM_JOB_ID:?}.out" 2>"${TPEN_RESULTS_ROOT:?}/scheduler/slurm-${SLURM_JOB_ID:?}.err"
case "$TPEN_UV" in /*) ;; *) echo "TPEN_UV must be absolute" >&2; exit 2;; esac
export UV_CACHE_DIR="${TMPDIR:?}/uv-cache-${SLURM_JOB_ID:?}"
"$TPEN_UV" sync --project "$TPEN_CHECKOUT" --locked --extra cu126 --extra parsl
PYTHONPATH="${TPEN_CHECKOUT:?}${PYTHONPATH:+:$PYTHONPATH}" "$TPEN_UV" run --project "$TPEN_CHECKOUT" --locked --extra cu126 --extra parsl python "$TPEN_CHECKOUT/experiments/atomistic/he-cutover/pipeline.py" \
  --plan-dir "$TPEN_RESULTS_ROOT/00_plan/$TPEN_PLAN_ATTEMPT_ID" --facility cannon \
  --launch-dir "$TPEN_RESULTS_ROOT/01_launch/$TPEN_PLAN_ATTEMPT_ID" --admission-id "$TPEN_PLAN_ATTEMPT_ID"
