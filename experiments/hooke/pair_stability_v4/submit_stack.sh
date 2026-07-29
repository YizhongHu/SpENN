#!/usr/bin/env bash

# V4-0 controller for the complete pair-stability smoke compatibility stack.

set -euo pipefail

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
STUDY="$(dirname "$SCRIPT")"
REPO="$(realpath "$STUDY/../../..")"
DISPATCH="$STUDY/dispatch.py"

if [[ "${1:-}" != "--worker" ]]; then
  MODE="${1:-}"
  if [[ "$MODE" != "smoke" ]]; then
    echo "usage: $0 smoke"
    echo "set RESULTS_ROOT to a new absolute root, or use the per-lineage default"
    exit 2
  fi

  command -v sbatch >/dev/null || { echo "sbatch not found" >&2; exit 1; }
  command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }
  [[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] || {
    echo "canonical V4-0 stack requires a clean checkout" >&2
    exit 1
  }

  STACK_ID="${STACK_ID:-$(TZ=America/New_York date +%Y%m%dT%H%M%S%z)-$RANDOM}"
  [[ "$STACK_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || {
    echo "invalid STACK_ID" >&2
    exit 1
  }
  RESULTS_ROOT="$(realpath -m "${RESULTS_ROOT:-$STUDY/results/$STACK_ID}")"
  [[ "$RESULTS_ROOT" = /* ]] || { echo "RESULTS_ROOT must be absolute" >&2; exit 1; }
  CONTROLLER_PARTITION="${STACK_CONTROLLER_PARTITION:-sapphire}"
  CONTROLLER_TIME="${STACK_CONTROLLER_TIME:-3-00:00:00}"
  [[ "$CONTROLLER_PARTITION" == "sapphire" ]] || {
    echo "canonical V4-0 controller must use sapphire" >&2
    exit 1
  }

  env UV_PROJECT_ENVIRONMENT="$REPO/.venv" \
    uv run python "$DISPATCH" initialize \
      --results-root "$RESULTS_ROOT" \
      --lineage-id "$STACK_ID" \
      --purpose experiment
  STACK_DIR="$RESULTS_ROOT/_v4/stack/$STACK_ID"
  mkdir -p "$STACK_DIR"
  cp "$SCRIPT" "$STACK_DIR/submit_stack.sh"
  printf '%q ' "$SCRIPT" smoke >"$STACK_DIR/controller_command.txt"
  printf '\n' >>"$STACK_DIR/controller_command.txt"
  {
    echo "mode=$MODE"
    echo "stack_id=$STACK_ID"
    echo "results_root=$RESULTS_ROOT"
    echo "results_root_policy=guarded-v4-root"
    echo "git_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "git_branch=$(git -C "$REPO" branch --show-current)"
    echo "git_dirty=false"
    echo "controller_partition=$CONTROLLER_PARTITION"
    echo "controller_time=$CONTROLLER_TIME"
    echo "controller_cpus=4"
    echo "controller_mem_per_cpu_gb=8"
    echo "scientific_partition=gpu_test"
    echo "submitted_at=$(TZ=America/New_York date --iso-8601=seconds)"
  } >"$STACK_DIR/manifest.txt"
  {
    echo "screen_train=submitit,cuda,chunk=32,cpus=4,mem_per_cpu_gb=8,partition=gpu_test,timeout_min=60"
    echo "screen_eval=submitit,cuda,chunk=32,cpus=4,mem_per_cpu_gb=8,partition=gpu_test,timeout_min=120"
    echo "confirm_train=submitit,cuda,chunk=8,cpus=4,mem_per_cpu_gb=8,partition=gpu_test,timeout_min=60"
    echo "confirm_eval=submitit,cuda,chunk=8,cpus=4,mem_per_cpu_gb=8,partition=gpu_test,timeout_min=120"
  } >"$STACK_DIR/profile.txt"
  env UV_PROJECT_ENVIRONMENT="$REPO/.venv" \
    uv run python "$DISPATCH" controller-request \
      --results-root "$RESULTS_ROOT" \
      --lineage-id "$STACK_ID" \
      --partition "$CONTROLLER_PARTITION" \
      --walltime "$CONTROLLER_TIME" \
      --cpus 4 \
      --mem-per-cpu-gb 8
  env UV_PROJECT_ENVIRONMENT="$REPO/.venv" \
    uv run python "$DISPATCH" stack-inventory \
      --results-root "$RESULTS_ROOT" \
      --lineage-id "$STACK_ID" \
      --phase pre
  # This is intentionally before the controller submission: gpu_test permits at
  # most two jobs/array tasks per submitted fan-out stage for this work.
  env UV_PROJECT_ENVIRONMENT="$REPO/.venv" \
    uv run python "$DISPATCH" profile-check

  set +e
  OUTPUT=$(sbatch \
    --parsable \
    --job-name=pair-stability-v4-stack \
    --partition="$CONTROLLER_PARTITION" \
    --time="$CONTROLLER_TIME" \
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
  echo "[v4-stack] submitted controller job_id=$JOB_ID"
  echo "[v4-stack] evidence=$STACK_DIR"
  exit 0
fi

[[ $# == 6 ]] || { echo "invalid worker invocation" >&2; exit 2; }
MODE="$2"
STACK_ID="$3"
RESULTS_ROOT="$(realpath -m "$4")"
STUDY="$(realpath "$5")"
REPO="$(realpath "$6")"
[[ "$MODE" == "smoke" ]] || { echo "worker supports smoke only" >&2; exit 2; }
[[ -d "$STUDY" && -d "$REPO" ]] || { echo "invalid worker source paths" >&2; exit 2; }
STACK_DIR="$RESULTS_ROOT/_v4/stack/$STACK_ID"
mkdir -p "$STACK_DIR"
exec > >(tee -a "$STACK_DIR/stack.log") 2>&1

local_dispatch() {
  env UV_PROJECT_ENVIRONMENT="$REPO/.venv" \
    uv run python "$DISPATCH" "$@"
}

fanout_dispatch() {
  env UV_PROJECT_ENVIRONMENT="$REPO/.venv-submitit" \
    uv run --extra submitit python "$DISPATCH" "$@"
}

# The controller worker, not the outer submitter, records the one immutable
# submission receipt.  This eliminates a race where a fast worker could reach
# finalization before the submitter had persisted the job identity.
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "worker has no SLURM_JOB_ID" >&2
  exit 1
}
local_dispatch controller-submission \
  --results-root "$RESULTS_ROOT" \
  --lineage-id "$STACK_ID" \
  --job-id "$SLURM_JOB_ID"

STAGE=starting
finalize_stack() {
  STAGE_RC=$?
  trap - EXIT
  set +e
  FINALIZATION_ERRORS=()
  FINAL_RC="$STAGE_RC"
  {
    echo "state=finalizing"
    echo "stage=$STAGE"
    echo "stage_exit_code=$STAGE_RC"
    echo "started_finalization_at=$(TZ=America/New_York date --iso-8601=seconds)"
  } >"$STACK_DIR/status.txt" || FINALIZATION_ERRORS+=("status_write_failed")
  local_dispatch stack-inventory \
    --results-root "$RESULTS_ROOT" \
    --lineage-id "$STACK_ID" \
    --phase post || FINALIZATION_ERRORS+=("post_inventory_failed")
  local_dispatch dispatch-manifest \
    --results-root "$RESULTS_ROOT" \
    --lineage-id "$STACK_ID" || FINALIZATION_ERRORS+=("dispatch_manifest_failed")
  # A successful route publishes V4-1A sidecars before, never after, the
  # immutable controller result.  The finalizer itself performs the public
  # pre-close control audit, completed-lineage audit, manifest-last publish,
  # and a fresh-process source verifier.
  if [[ "$STAGE" == "complete" && "$STAGE_RC" -eq 0 ]]; then
    local_dispatch contract-sidecars-finalize \
      --results-root "$RESULTS_ROOT" \
      --lineage-id "$STACK_ID" || FINALIZATION_ERRORS+=("contract_sidecars_finalize_failed")
  fi
  find "$RESULTS_ROOT/_v4/dispatch/$STACK_ID" \
    -type f -name result.json -print | sort \
    >"$STACK_DIR/dispatch_receipts.txt" || FINALIZATION_ERRORS+=("dispatch_receipt_list_failed")
  if ((${#FINALIZATION_ERRORS[@]} > 0)) && ((FINAL_RC == 0)); then
    FINAL_RC=1
  fi
  RESULT_ARGS=(
    controller-result
    --results-root "$RESULTS_ROOT"
    --lineage-id "$STACK_ID"
    --stage "$STAGE"
    --stage-exit-code "$STAGE_RC"
    --exit-code "$FINAL_RC"
    --worker-job-id "${SLURM_JOB_ID:-missing}"
    --worker-partition "${SLURM_JOB_PARTITION:-missing}"
    --effective-cpus-per-task "${SLURM_CPUS_PER_TASK:-missing}"
    --effective-mem-per-cpu-mb "${SLURM_MEM_PER_CPU:-missing}"
    --effective-time-limit "${SLURM_TIMELIMIT:-missing}"
  )
  for ERROR in "${FINALIZATION_ERRORS[@]}"; do
    RESULT_ARGS+=(--finalization-error "$ERROR")
  done
  local_dispatch "${RESULT_ARGS[@]}"
  RESULT_RC=$?
  if ((RESULT_RC != 0)); then
    FINAL_RC=1
  fi
  exit "$FINAL_RC"
}
trap finalize_stack EXIT

set -x
cd "$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

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

STAGE=screen_plan
local_dispatch run screen_plan \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID"
EXPECTED=$(jq -r '.n_jobs' "$RESULTS_ROOT/00_grid/$STACK_ID/manifest.json")
[[ "$EXPECTED" == "64" ]] || { echo "screen plan produced $EXPECTED jobs, expected 64" >&2; exit 1; }

STAGE=screen_train
fanout_dispatch run screen_train \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --grid-attempt "$STACK_ID"
wait_stage screen_train "$RESULTS_ROOT/01_train/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=screen_eval
fanout_dispatch run screen_eval \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --grid-attempt "$STACK_ID" \
  --train-attempt "$STACK_ID"
wait_stage screen_eval "$RESULTS_ROOT/02_validation/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=screen_collect
local_dispatch run screen_collect \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --grid-attempt "$STACK_ID"
COLLECTED=$(jq -r '.n_collected' "$RESULTS_ROOT/03_collect/$STACK_ID/collection_report.json")
[[ "$COLLECTED" == "64" ]] || { echo "collected $COLLECTED/64 validation runs" >&2; exit 1; }

STAGE=select
local_dispatch run select \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --collection-attempt "$STACK_ID"

STAGE=confirm_plan
local_dispatch run confirm_plan \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --grid-attempt "$STACK_ID" \
  --selection-attempt "$STACK_ID"
FINAL_EXPECTED=$(jq -r '.n_jobs' "$RESULTS_ROOT/05_final_grid/$STACK_ID/manifest.json")
[[ "$FINAL_EXPECTED" == "8" ]] || { echo "final plan produced $FINAL_EXPECTED jobs, expected 8" >&2; exit 1; }

STAGE=confirm_train
fanout_dispatch run confirm_train \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --final-grid-attempt "$STACK_ID"
wait_stage confirm_train "$RESULTS_ROOT/06_final_train/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=confirm_eval
fanout_dispatch run confirm_eval \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --final-grid-attempt "$STACK_ID" \
  --final-train-attempt "$STACK_ID"
wait_stage confirm_eval "$RESULTS_ROOT/07_final_eval/stage_plans/$STACK_ID/execution_records.jsonl"

STAGE=confirm_collect
local_dispatch run confirm_collect \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --final-grid-attempt "$STACK_ID" \
  --final-eval-attempt "$STACK_ID"
FINAL_COLLECTED=$(tail -n +2 "$RESULTS_ROOT/08_final_collect/$STACK_ID/run_index.csv" | wc -l)
FINAL_COLLECTED="${FINAL_COLLECTED//[[:space:]]/}"
[[ "$FINAL_COLLECTED" == "8" ]] || { echo "collected $FINAL_COLLECTED/8 final runs" >&2; exit 1; }

STAGE=report
local_dispatch run report \
  --results-root "$RESULTS_ROOT" \
  --output-attempt "$STACK_ID" \
  --final-collect-attempt "$STACK_ID"
test -s "$RESULTS_ROOT/09_final_report/$STACK_ID/report.md"

STAGE=complete
echo "[v4-stack] complete: $RESULTS_ROOT/09_final_report/$STACK_ID/report.md"
