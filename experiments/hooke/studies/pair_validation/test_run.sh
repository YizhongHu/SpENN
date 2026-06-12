#!/usr/bin/env bash
# =============================================================================
# Pair validation study end-to-end test run
# =============================================================================
#
# Runs a tiny 2x2 grid (2 seeds x 2 learning rates) of the benchmark training
# config at smoke scale, then collects, selects, and dry-runs the final
# evaluator, and reports PASS/FAIL. Use this to confirm the whole
# train -> validation -> collect -> select -> final-eval-commands pipeline
# works before launching the real scan.
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
#   PV_SCRATCH  output dir; must be writable (default: a fresh dir under
#               /n/netscratch/kozinsky_lab/Everyone/$USER on FASRC, else
#               mktemp -d). Named PV_SCRATCH because FASRC exports
#               SCRATCH=/n/netscratch globally, which is not writable.
#
#SBATCH --job-name=spenn-pv-test-run
#SBATCH --output=pv_test_run_%j.out

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
SCRIPT_DIR="$REPO_ROOT/experiments/hooke/studies/pair_validation"
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

# Default scratch: lab netscratch on FASRC (writable per-user dir), else tmpdir.
# Knob is PV_SCRATCH, not SCRATCH: FASRC exports SCRATCH=/n/netscratch globally.
NETSCRATCH_BASE="/n/netscratch/kozinsky_lab/Everyone/$USER"
if [[ -z "${PV_SCRATCH:-}" ]]; then
  if [[ -d "$NETSCRATCH_BASE" && -w "$NETSCRATCH_BASE" ]]; then
    SCRATCH="$(mktemp -d -p "$NETSCRATCH_BASE" spenn_pv_test_run.XXXXXX)"
  else
    SCRATCH="$(mktemp -d -t spenn_pv_test_run.XXXXXX)"
  fi
elif [[ ! -d "$PV_SCRATCH" || ! -w "$PV_SCRATCH" ]]; then
  echo "ERROR: PV_SCRATCH=$PV_SCRATCH is not a writable directory" >&2
  echo "       (e.g. use a subdir of $NETSCRATCH_BASE)" >&2
  exit 1
else
  SCRATCH="$PV_SCRATCH"
fi
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
selection:
  absolute_energy_floor: 1.0e-4
  margin:
    stderr_multiplier: 2.0
    seed_iqr_fraction: 0.25
  require_all_seeds: true
  tie_breakers:
    - validation/energy_variance
    - validation_energy_iqr
    - validation/energy_stderr
    - geometry_warning_count
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
final_evaluation:
  study_name: hooke_pair_validation_test_run_final
  eval_config: experiments/hooke/configs/benchmark/pair_final_eval.yaml
  training_seeds: [100, 101]
  eval_seeds: [100000, 100001]
  allow_validation_seed_reuse: false
  sampler:
    n_walkers: 8192
    burn_in: 1000
    n_steps: 500
    proposal_scale: 0.35
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

# 4. Generate the final-benchmark commands (dry-run; nothing executes).
uv run --extra "$EXTRA" python "$SCRIPT_DIR/evaluate_selected.py" \
  --manifest "$MANIFEST" --selected-config "$RESULTS/selected_config.yaml" \
  --run-root "$RUN_ROOT" --output-dir "$RESULTS" --dry-run

# 5. Check the pipeline produced what the real scan needs.
fail() { echo "TEST RUN FAILED: $1"; exit 1; }
[[ -s "$RESULTS/runs.csv" ]] || fail "runs.csv missing/empty"
[[ -s "$RESULTS/selection.csv" ]] || fail "selection.csv missing/empty"
[[ -s "$RESULTS/selected_config.yaml" ]] || fail "selected_config.yaml missing/empty"
[[ -s "$RESULTS/selection_report.md" ]] || fail "selection_report.md missing/empty"
[[ -s "$RESULTS/final_eval_commands.sh" ]] || fail "final_eval_commands.sh missing/empty"
[[ -s "$RESULTS/final_eval_manifest.yaml" ]] || fail "final_eval_manifest.yaml missing/empty"
[[ -s "$RESULTS/final_eval_inputs.csv" ]] || fail "final_eval_inputs.csv missing/empty"
grep -q "validation/energy" "$RESULTS/runs.csv" || fail "no validation/energy column"
grep -q "radius_q99" "$RESULTS/runs.csv" || fail "no sampler geometry columns"
grep -q "pair_final_eval.yaml" "$RESULTS/final_eval_commands.sh" \
  || fail "final eval commands missing eval config"
completed=$(awk -F, 'NR>1 && $2=="completed"' "$RESULTS/runs.csv" | wc -l)
[[ "$completed" -eq 4 ]] || fail "expected 4 completed runs, got $completed"
# n_failed_seeds must be 0 for every config group (located by header name).
awk -F, 'NR==1 { for (i = 1; i <= NF; i++) if ($i == "n_failed_seeds") col = i }
         NR>1 && $col != 0 { bad = 1 } END { exit bad }' "$RESULTS/selection.csv" \
  || fail "selection has failed seeds"

echo
echo "TEST RUN PASSED on device=$DEVICE"
echo "  results: $RESULTS"
echo "  selected: $(grep -m1 config_id "$RESULTS/selected_config.yaml")"
echo "Everything is working as expected; the real scan can be launched."
