#!/bin/bash
# =============================================================================
#  SpENN | SLURM: Hooke pair smoke evaluation
# -----------------------------------------------------------------------------
#  Submit from the repository root:
#
#    sbatch experiments/hooke/slurm/eval_pair_smoke.sbatch [overrides...]
#
#  Extra arguments are forwarded to run.py as OmegaConf dotlist overrides:
#
#    # CPU smoke with a custom output root:
#    sbatch experiments/hooke/slurm/eval_pair_smoke.sbatch run.root=/scratch/$USER/spenn
#
#    # GPU smoke (also switch the partition/gres below or via sbatch flags):
#    sbatch --partition=kozinsky_gpu --gres=gpu:1 \
#        experiments/hooke/slurm/eval_pair_smoke.sbatch runtime.device=cuda
# =============================================================================
#SBATCH --job-name=spenn_hooke_pair_eval_smoke
#SBATCH --partition=kozinsky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/hooke/slurm/logs/%x_%j.out

set -euo pipefail

CONFIG=experiments/hooke/configs/smoke/pair_eval.yaml

# --- Job metadata prologue (line-oriented, grep-friendly) --------------------
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] array_task_id=${SLURM_ARRAY_TASK_ID:-}"
echo "[slurm] host=$(hostname)"
echo "[slurm] workdir=$(pwd)"
echo "[slurm] started_at=$(date --iso-8601=seconds)"
echo "[slurm] config=${CONFIG}"
echo "[slurm] command=$*"

# --- Environment --------------------------------------------------------------
# Unbuffered, line-oriented logs; no interactive progress bars in batch logs.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Plain logs by default. Export FORCE_COLOR=1 before sbatch for colored logs.
if [ -z "${FORCE_COLOR:-}" ]; then
    export NO_COLOR=1
fi

# This script must run from the repository root (run.py lives there).
if [ ! -f run.py ]; then
    echo "[slurm] error: run.py not found; submit from the repository root" >&2
    exit 1
fi

# --- Run ----------------------------------------------------------------------
# Capture the exit code so the epilogue still prints on failure, then preserve it.
set +e
uv run python -u run.py --config "${CONFIG}" "$@"
status=$?
set -e

echo "[slurm] finished_at=$(date --iso-8601=seconds)"
echo "[slurm] exit_code=${status}"
exit "${status}"
