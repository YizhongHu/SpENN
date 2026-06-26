#!/usr/bin/env bash
# =============================================================================
# SLURM array launcher for the Hooke pair validation scan (study v1)
# =============================================================================
#
# One array task per grid point of manifest.yaml (3 seeds x 3 lrs x 3 channel
# counts x 2 gates = 54 tasks). Submit from the repo root on FASRC:
#
#   mkdir -p slurm_logs
#   sbatch experiments/hooke/studies/pair_validation/launch_array.sh
#
# CPU partitions (sapphire, kozinsky, seas_compute) work too:
#
#   DEVICE=cpu sbatch -p sapphire --gres="" \
#     experiments/hooke/studies/pair_validation/launch_array.sh
#
# Keep the slurm_logs/ output files around for reproducibility.
#
#SBATCH --job-name=hooke-pv-v1
#SBATCH --partition=kozinsky_gpu,seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-53
#SBATCH --output=slurm_logs/hooke_pv_v1_%A_%a.out

set -euo pipefail

# Under sbatch the script runs from a copy in the slurmd spool dir, so the
# BASH_SOURCE-relative path math breaks; recover the repo root from the
# directory sbatch was invoked in instead.
if [[ "${BASH_SOURCE[0]}" == */slurmd/* && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR" && git rev-parse --show-toplevel)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"
if [[ "$DEVICE" == "cuda" ]]; then
  export UV_PROJECT_ENVIRONMENT=.venv-gpu
  EXTRA=cu126
else
  EXTRA=cpu
fi

# Grid axes; must mirror manifest.yaml. Changing the grid means a new study
# version (new manifest + new study.name).
SEEDS=(3 9 11)
LRS=(3e-4 1e-3 3e-3)
CHANNELS=(8 32 128)
GATES=(silu sigmoid)

i="${SLURM_ARRAY_TASK_ID:?run via sbatch --array}"
seed="${SEEDS[$(( i % ${#SEEDS[@]} ))]}";        i=$(( i / ${#SEEDS[@]} ))
lr="${LRS[$(( i % ${#LRS[@]} ))]}";              i=$(( i / ${#LRS[@]} ))
channels="${CHANNELS[$(( i % ${#CHANNELS[@]} ))]}"; i=$(( i / ${#CHANNELS[@]} ))
gate="${GATES[$(( i % ${#GATES[@]} ))]}"

echo "task=$SLURM_ARRAY_TASK_ID seed=$seed lr=$lr channels=$channels gate=$gate device=$DEVICE"

uv run --extra "$EXTRA" python run.py \
  --config experiments/hooke/configs/benchmark/pair_train.yaml \
  run.root=outputs/hooke_pair_validation_v1 \
  study.name=hooke_pair_validation_v1 \
  runtime.device="$DEVICE" \
  runtime.seed="$seed" \
  optimizer_params.lr="$lr" \
  model_params.channels="$channels" \
  model_params.gate_activation="$gate"
