#!/usr/bin/env bash
# =============================================================================
# Pair validation study end-to-end test run
# =============================================================================
#
# Runs a tiny 2x2 grid (2 seeds x 2 learning rates) of the benchmark training
# config at smoke scale, then collects and selects, and reports PASS/FAIL.
# Use this to confirm the whole train -> validation -> collect -> select
# pipeline works before launching the real scan.
#
# Local (workstation/WSL):
#   bash experiments/hooke/studies/pair_validation/test_run.sh
#
# SLURM (FASRC), GPU:
#   sbatch -p kozinsky_gpu --gres=gpu:1 -c 4 -t 00:30:00 \
#     experiments/hooke/studies/pair_validation/test_run.sh
#
# SLURM (FASRC), CPU:
#   DEVICE=cpu sbatch -p sapphire -c 4 -t 00:30:00 \
#     experiments/hooke/studies/pair_validation/test_run.sh
#
# Environment knobs:
#   DEVICE   cuda|cpu   (default: cuda if nvidia-smi works, else cpu)
#   SCRATCH  output dir (default: mktemp -d)
#
#SBATCH --job-name=spenn-pv-test-run
#SBATCH --output=pv_test_run_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

# Pick device + matching uv environment (see README.md: GPU uses .venv-gpu).
if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    DEVICE=cuda
  else
    DEVICE=cpu
  fi
fi
if [[ "$DEVICE" == "cuda" ]]; then
  export UV_PROJECT_ENVIRONMENT=.venv-gpu
  EXTRA=cu126
else
  EXTRA=cpu
fi

SCRATCH="${SCRATCH:-$(mktemp -d -t spenn_pv_test_run.XXXXXX)}"
RUN_ROOT="$SCRATCH/runs"
RESULTS="$SCRATCH/results"
echo "device=$DEVICE extra=$EXTRA scratch=$SCRATCH"

# Tiny manifest matching the tiny grid below (the real manifest declares the
# full 54-point protocol; this one only drives the pipeline check).
MANIFEST="$SCRATCH/manifest.yaml"
cat > "$MANIFEST" <<'EOF'
study:
  name: hooke_pair_validation_test_run
  purpose: validation_scan
  sector: singlet
train_config: experiments/hooke/configs/benchmark/pair_train.yaml
grid:
  runtime.seed: [3, 9]
  optimizer_params.lr: [1.0e-3, 3.0e-3]
  model_params.channels: [4]
  model_params.layers: [1]
  model_params.gate_activation: [silu]
seed_key: runtime.seed
validation:
  metric: validation/energy
  aggregate: median
  checkpoint: final
  failed_run_value: .inf
eligibility:
  require:
    - checks/data_integrity/passed
    - checks/gradient/passed
    - checks/equivariance/full_model/passed
  local_energy_finite_fraction: 1.0
tie_breakers:
  - validation/energy_variance
  - seed_energy_spread
  - model_params.channels
  - runtime/wall_time_sec
diagnostic_fields:
  sampler_geometry:
    - validation/sampler/radius_mean
    - validation/sampler/radius_q99
    - validation/sampler/radius_max
    - validation/sampler/electron_distance_q01
    - validation/sampler/electron_distance_min
    - validation/sampler/position_rms
geometry_flags:
  electron_distance_q01_min: 1.0e-3
EOF

# 1. Train the tiny grid at smoke scale (W&B stays off: loggers are CSV/JSONL).
for seed in 3 9; do
  for lr in 1e-3 3e-3; do
    echo "=== train seed=$seed lr=$lr ==="
    uv run --extra "$EXTRA" python run.py \
      --config experiments/hooke/configs/benchmark/pair_train.yaml \
      run.root="$RUN_ROOT" \
      study.name=hooke_pair_validation_test_run \
      runtime.device="$DEVICE" \
      runtime.seed="$seed" \
      optimizer_params.lr="$lr" \
      model_params.channels=4 \
      model_params.hidden_channels=8 \
      sampler_params.n_walkers=16 \
      sampler_params.burn_in=10 \
      sampler_params.n_steps=5 \
      validation_sampler_params.n_walkers=32 \
      validation_sampler_params.burn_in=10 \
      validation_sampler_params.n_steps=5 \
      training.max_steps=2 \
      checks.every_n_steps=1 \
      checkpoint.every_n_steps=1 \
      status.every_n_steps=1
  done
done

# 2. Collect run outputs into a normalized table.
uv run --extra "$EXTRA" python "$SCRIPT_DIR/collect.py" \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT" --output-dir "$RESULTS"

# 3. Apply the selection rule.
uv run --extra "$EXTRA" python "$SCRIPT_DIR/select.py" \
  --manifest "$MANIFEST" --runs "$RESULTS/runs.csv" --output-dir "$RESULTS"

# 4. Check the pipeline produced what the real scan needs.
fail() { echo "TEST RUN FAILED: $1"; exit 1; }
[[ -s "$RESULTS/runs.csv" ]] || fail "runs.csv missing/empty"
[[ -s "$RESULTS/selection.csv" ]] || fail "selection.csv missing/empty"
[[ -s "$RESULTS/selected_config.yaml" ]] || fail "selected_config.yaml missing/empty"
[[ -s "$RESULTS/selection_report.md" ]] || fail "selection_report.md missing/empty"
grep -q "validation/energy" "$RESULTS/runs.csv" || fail "no validation/energy column"
grep -q "radius_q99" "$RESULTS/runs.csv" || fail "no sampler geometry columns"
completed=$(awk -F, 'NR>1 && $2=="completed"' "$RESULTS/runs.csv" | wc -l)
[[ "$completed" -eq 4 ]] || fail "expected 4 completed runs, got $completed"
# n_failed_seeds (last column) must be 0 for every config group.
awk -F, 'NR>1 && $NF != 0 { bad = 1 } END { exit bad }' "$RESULTS/selection.csv" \
  || fail "selection has failed seeds"

echo
echo "TEST RUN PASSED on device=$DEVICE"
echo "  results: $RESULTS"
echo "  selected: $(grep -m1 config_id "$RESULTS/selected_config.yaml")"
echo "Everything is working as expected; the real scan can be launched."
