#!/usr/bin/env bash

# Scrappy run-and-forget controller for the complete pair-stability V3 stack.
# First invocation submits this script to Slurm. The controller then submits
# each GPU stage, waits for it, and runs the intervening CPU stages directly.

set -euo pipefail

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
STUDY="$(dirname "$SCRIPT")"
REPO="$(realpath "$STUDY/../../..")"

if [[ "${1:-}" != "--worker" ]]; then
  MODE="${1:-}"
  case "$MODE" in
    full|smoke|pilot|pilot-smoke) ;;
    *)
      echo "usage: $0 {full|smoke|pilot|pilot-smoke}"
      echo "logs:  $STUDY/results/stack/<stack_id>/"
      exit 2
      ;;
  esac

  command -v sbatch >/dev/null || { echo "sbatch not found" >&2; exit 1; }
  command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }

  RESULTS_ROOT="$(realpath -m "${RESULTS_ROOT:-$STUDY/results}")"
  STACK_ID="${STACK_ID:-$(TZ=America/New_York date +%Y%m%dT%H%M%S%z)-$RANDOM}"
  [[ "$STACK_ID" =~ ^[A-Za-z0-9._+-]+$ ]] || { echo "invalid STACK_ID" >&2; exit 1; }
  STACK_DIR="$RESULTS_ROOT/stack/$STACK_ID"
  mkdir -p "$RESULTS_ROOT/stack"
  mkdir "$STACK_DIR" || { echo "stack already exists: $STACK_DIR" >&2; exit 1; }
  cp "$SCRIPT" "$STACK_DIR/submit_stack.sh"
  {
    echo "mode=$MODE"
    echo "stack_id=$STACK_ID"
    echo "results_root=$RESULTS_ROOT"
    echo "git_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "git_branch=$(git -C "$REPO" branch --show-current)"
    echo "submitted_at=$(date --iso-8601=seconds)"
  } >"$STACK_DIR/manifest.txt"

  set +e
  OUTPUT=$(sbatch \
    --parsable \
    --job-name=pair-stability-v3-stack \
    --partition="${STACK_CONTROLLER_PARTITION:-sapphire}" \
    --time="${STACK_CONTROLLER_TIME:-7-00:00:00}" \
    --cpus-per-task=4 \
    --mem-per-cpu=8G \
    --chdir="$REPO" \
    --output="$STACK_DIR/controller-%j.log" \
    "$SCRIPT" --worker "$MODE" "$STACK_ID" "$RESULTS_ROOT" "$STUDY" "$REPO" 2>&1)
  RC=$?
  set -e
  printf '%s\n' "$OUTPUT" | tee "$STACK_DIR/submission.log"
  ((RC == 0)) || exit "$RC"
  JOB_ID="${OUTPUT##*$'\n'}"
  JOB_ID="${JOB_ID%%;*}"
  echo "controller_job_id=$JOB_ID" >>"$STACK_DIR/manifest.txt"
  echo "[stack] submitted controller job_id=$JOB_ID"
  echo "[stack] logs=$STACK_DIR"
  exit 0
fi

[[ $# == 6 ]] || { echo "invalid worker invocation" >&2; exit 2; }
MODE="$2"
STACK_ID="$3"
RESULTS_ROOT="$(realpath -m "$4")"
STUDY="$(realpath "$5")"
REPO="$(realpath "$6")"
[[ -d "$STUDY" && -d "$REPO" ]] || { echo "invalid worker source paths" >&2; exit 2; }
STACK_DIR="$RESULTS_ROOT/stack/$STACK_ID"
mkdir -p "$STACK_DIR"
exec > >(tee -a "$STACK_DIR/stack.log") 2>&1

STAGE=starting
trap 'RC=$?; echo "status=$([[ $RC == 0 ]] && echo completed || echo failed)" >"$STACK_DIR/status.txt"; echo "stage=$STAGE" >>"$STACK_DIR/status.txt"; echo "exit_code=$RC" >>"$STACK_DIR/status.txt"' EXIT
set -x

cd "$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

local_py() {
  env UV_PROJECT_ENVIRONMENT="$REPO/.venv" uv run python "$@"
}

submit_py() {
  env UV_PROJECT_ENVIRONMENT="$REPO/.venv-submitit" uv run --extra submitit python "$@"
}

wait_stage() {
  local name="$1"
  local records="$2"
  local csv states
  [[ -s "$records" ]] || { echo "missing $records" >&2; return 1; }
  mapfile -t JOB_IDS < <(
    jq -r '.launcher_job_id' "$records" \
      | tr ',' '\n' \
      | sed -E 's/^[^:]+://; s/_[0-9]+$//' \
      | awk 'NF' \
      | sort -u
  )
  ((${#JOB_IDS[@]} > 0)) || { echo "no job ids in $records" >&2; return 1; }
  printf '%s\n' "${JOB_IDS[@]}" | tee "$STACK_DIR/${name}_job_ids.txt"
  csv="$(IFS=,; echo "${JOB_IDS[*]}")"
  while [[ -n "$(squeue -h -j "$csv" -o '%i' 2>/dev/null)" ]]; do
    sleep "${STACK_POLL_SECONDS:-60}"
  done
  sleep 10
  states="$STACK_DIR/${name}_sacct.txt"
  sacct -n -X -j "$csv" --format=JobIDRaw,State -P >"$states"
  grep -q '|' "$states" || { echo "no accounting data for $name" >&2; return 1; }
  if awk -F'|' 'NF >= 2 && $2 !~ /^COMPLETED/ {bad=1} END {exit !bad}' "$states"; then
    echo "$name has failed Slurm jobs; see $states" >&2
    return 1
  fi
}

case "$MODE" in
  full)
    GRID=grid.yaml
    BLIND=(--blind --blind-seed 811)
    ;;
  smoke)
    GRID=smoke.yaml
    BLIND=(--blind --blind-seed 811)
    ;;
  pilot)
    GRID=pilot.yaml
    BLIND=(--no-blind)
    ;;
  pilot-smoke)
    GRID=pilot_smoke.yaml
    BLIND=(--no-blind)
    ;;
esac

if [[ "$MODE" == smoke ]]; then
  TRAIN=(--backend submitit --device cuda --chunk-size 32 --slurm-cpus 4 --slurm-partition gpu_test --slurm-mem-per-cpu-gb 8 --slurm-timeout-min 60)
  VALIDATE=(--backend submitit --device cuda --chunk-size 32 --slurm-cpus 4 --slurm-partition gpu_test --slurm-mem-per-cpu-gb 8 --slurm-timeout-min 120)
  FINAL_TRAIN=(--backend submitit --device cuda --chunk-size 8 --slurm-cpus 4 --slurm-cuda-partition gpu_test --slurm-mem-per-cpu-gb 8 --slurm-cuda-timeout-min 60)
  FINAL_EVAL=(--backend submitit --device cuda --chunk-size 8 --slurm-cpus 4 --slurm-partition gpu_test --slurm-mem-per-cpu-gb 8 --slurm-timeout-min 120)
else
  TRAIN=(--backend submitit --device cuda --chunk-size 18 --slurm-cpus 4 --slurm-mem-per-cpu-gb 8 --slurm-timeout-min 480)
  VALIDATE=(--backend submitit --device cuda --chunk-size 32 --slurm-cpus 4 --slurm-mem-per-cpu-gb 8 --slurm-timeout-min 120)
  FINAL_TRAIN=(--backend submitit --device cuda --chunk-size 18 --slurm-cpus 4 --slurm-mem-per-cpu-gb 8 --slurm-timeout-min 480)
  FINAL_EVAL=(--backend submitit --device cuda --chunk-size 32 --slurm-cpus 4 --slurm-mem-per-cpu-gb 8 --slurm-timeout-min 120)
fi

STAGE=plan
local_py "$STUDY/plan.py" --grid "$STUDY/configs/$GRID" --results-root "$RESULTS_ROOT" --attempt-id "$STACK_ID" "${BLIND[@]}"
EXPECTED=$(jq -r '.n_jobs' "$RESULTS_ROOT/00_grid/$STACK_ID/manifest.json")

STAGE=train
submit_py "$STUDY/train.py" --results-root "$RESULTS_ROOT" --grid-attempt-id "$STACK_ID" "${TRAIN[@]}"
wait_stage train "$RESULTS_ROOT/01_train/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=validation
submit_py "$STUDY/validate.py" --results-root "$RESULTS_ROOT" --grid-attempt-id "$STACK_ID" --train-attempt-id "$STACK_ID" --attempt-id "$STACK_ID" "${VALIDATE[@]}"
wait_stage validation "$RESULTS_ROOT/02_validation/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=collect
local_py "$STUDY/collect.py" --results-root "$RESULTS_ROOT" --grid-attempt-id "$STACK_ID" --attempt-id "$STACK_ID"
COLLECTED=$(jq -r '.n_collected' "$RESULTS_ROOT/03_collect/$STACK_ID/collection_report.json")
[[ "$COLLECTED" == "$EXPECTED" ]] || { echo "collected $COLLECTED/$EXPECTED validation runs" >&2; exit 1; }

STAGE=select
local_py "$STUDY/select_champions.py" --results-root "$RESULTS_ROOT" --collection-attempt-id "$STACK_ID" --attempt-id "$STACK_ID"

STAGE=final-plan
local_py "$STUDY/final_plan.py" --results-root "$RESULTS_ROOT" --selection-attempt-id "$STACK_ID" --attempt-id "$STACK_ID"
FINAL_EXPECTED=$(jq -r '.n_jobs' "$RESULTS_ROOT/05_final_grid/$STACK_ID/manifest.json")

STAGE=final-train
submit_py "$STUDY/final_train.py" --results-root "$RESULTS_ROOT" --final-grid-attempt-id "$STACK_ID" --attempt-id "$STACK_ID" "${FINAL_TRAIN[@]}"
wait_stage final_train "$RESULTS_ROOT/06_final_train/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=final-eval
submit_py "$STUDY/final_eval.py" --results-root "$RESULTS_ROOT" --final-grid-attempt-id "$STACK_ID" --final-train-attempt-id "$STACK_ID" --attempt-id "$STACK_ID" "${FINAL_EVAL[@]}"
wait_stage final_eval "$RESULTS_ROOT/07_final_eval/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=final-collect
local_py "$STUDY/final_collect.py" --results-root "$RESULTS_ROOT" --final-grid-attempt-id "$STACK_ID" --final-eval-attempt-id "$STACK_ID" --attempt-id "$STACK_ID"
FINAL_COLLECTED=$(tail -n +2 "$RESULTS_ROOT/08_final_collect/$STACK_ID/run_index.csv" | wc -l)
FINAL_COLLECTED="${FINAL_COLLECTED//[[:space:]]/}"
[[ "$FINAL_COLLECTED" == "$FINAL_EXPECTED" ]] || { echo "collected $FINAL_COLLECTED/$FINAL_EXPECTED final runs" >&2; exit 1; }

STAGE=final-report
local_py "$STUDY/final_report.py" --results-root "$RESULTS_ROOT" --final-collect-attempt-id "$STACK_ID" --attempt-id "$STACK_ID"
test -s "$RESULTS_ROOT/09_final_report/$STACK_ID/report.md"

STAGE=complete
echo "[stack] complete: $RESULTS_ROOT/09_final_report/$STACK_ID/report.md"
