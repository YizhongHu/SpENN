# Experiment Planning / Execution Split

This note records the proposed direction for separating experiment design from
execution orchestration in SpENN experiments. It is intentionally a design note,
not an implementation patch. The current `hooke/pair_stability_v2` run is live;
do not refactor its execution path until the run is finished or we have a tested
compatibility layer that can prove it preserves existing behavior.

## Motivation

The pair-stability runs exposed a structural issue: the code that decides which
scientific runs exist is intertwined with the code that decides how those runs
are submitted to Slurm. This makes every scheduling improvement risky because it
can accidentally change scientific lineage, naming, checkpoint selection,
result collection, or reproducibility.

The two concerns are different:

- Planning decides what to run.
- Execution decides where and how to run it.

They should communicate through an explicit, durable task table rather than
through stage-specific Python functions that both generate commands and submit
jobs.

This split matters because we want a reusable experiment toolkit for SpENN, not
a growing collection of bespoke launch scripts for each study.

## Current Pain Points

The current `pair_stability_v2` scripts have been useful and pragmatic, but they
bundle several responsibilities:

- Enumerating hyperparameter grids.
- Blinding semantic axes into routine labels.
- Assigning seeds and final replicates.
- Choosing champions.
- Writing per-run provenance files.
- Resolving previous-stage attempts and latest pointers.
- Resolving checkpoint resume.
- Selecting CPU or CUDA environments.
- Choosing Slurm partitions, memory, CPUs, walltime, and chunk sizes.
- Submitting Submitit arrays.
- Submitting dependent launcher jobs.
- Claiming rows in mixed CPU/CUDA mode.
- Skipping completed rows.
- Reclaiming failed/stopped rows.
- Handling local opportunistic workers.

This is too much surface area for a stage script. The stage scripts are becoming
a mix of scientific intent, filesystem layout policy, scheduler behavior, and
cluster-specific operational knowledge.

The failure modes reflect this coupling:

- A GPU partial checkpoint can collide with a CPU rerun unless the launcher knows
  resume semantics.
- A completed run can be resubmitted unless the launcher knows completion
  semantics.
- A CUDA checkpoint can fail on CPU unless restore code is device-aware.
- A CPU/GPU mixed run can produce immediate-exit loser jobs because row claiming
  is happening at job startup.
- Slurm dependency launchers are stage-specific rather than a generic workflow
  concept.
- Environment separation (`.venv`, `.venv-gpu`, `.venv-submitit`) is entangled
  with stage-specific command construction.

The goal is not to make all of this abstract for its own sake. The goal is to
make the important scientific objects stable and make execution replaceable.

## Core Distinction

There should be three layers.

### 1. Workflow Layer

The workflow layer defines the stage DAG:

```text
screen_plan
  -> screen_train
  -> screen_eval
  -> screen_collect
  -> screen_select
  -> confirm_plan
  -> confirm_train
  -> confirm_eval
  -> confirm_collect
  -> report
```

In the current directory names this is:

```text
00_grid
  -> 01_train
  -> 02_validation
  -> 03_collect
  -> 04_select
  -> 05_final_grid
  -> 06_final_train
  -> 07_final_eval
  -> 08_final_collect
  -> 09_final_report
```

The train/eval x2 shape is reasonable for QMC:

- A screening phase explores model/configuration choices.
- A validation/evaluation phase ranks noisy training outputs.
- A selection phase chooses candidates.
- A confirmation phase runs independent final seeds.
- A final evaluation phase uses a larger or more standardized diagnostic budget.

This is similar to common experimental practice: exploration, selection, and
confirmation should be separated.

The workflow layer should say which stage outputs depend on which upstream
outputs. It should not know Slurm partitions or uv environments.

### 2. Planning Layer

The planning layer creates immutable task plans.

For fixed-grid stages, this is deterministic:

```text
inputs: config, grid axes, seed policy, blinding policy
output: immutable task table
```

For adaptive or stochastic search, the task table is dynamic but should still be
append-only:

```text
observe completed trials
ask search policy for more trials
append pending tasks
execute tasks
tell search policy results
repeat
```

This note recommends postponing adaptive search for now. The current studies are
well served by fixed task tables, and fixed task tables are much easier to audit
and resume on a busy HPC cluster.

The planning layer owns:

- Axis definitions.
- Hyperparameter grids.
- Seed assignment.
- Blinding and unblinding.
- Replicate counts.
- Search policies.
- Champion selection criteria.
- Expansion from selected champions into confirmation tasks.
- Logical task identity.
- Logical dependencies.
- Expected output locations.

It should not own:

- Slurm partition names.
- CPUs, memory, walltime, or GPUs.
- Chunk sizes.
- Submitit details.
- Local workers.
- CPU/CUDA racing or worker pools.
- Whether a particular row is currently running.

### 3. Execution Layer

The execution layer consumes task plans and runs eligible tasks.

It owns:

- Backend choice: local, Submitit, Snakemake, TaskVine, Parsl, etc.
- Resource profiles: CPU, CUDA, smoke CPU, smoke CUDA.
- Slurm partitions and resource defaults.
- Job arrays or worker pools.
- Chunk size.
- Retry policy.
- Resume policy application.
- Completion checks.
- Submission records.
- Worker/job metadata.
- Queue/backfill strategy.

It should not decide which scientific rows exist. It should only decide how to
run existing tasks.

### 4. Run Layer

The run layer is the concrete single-run entrypoint, currently `run.py`.

It owns:

- Instantiating one resolved config.
- Running training or evaluation.
- Writing metrics.
- Writing checkpoints.
- Restoring checkpoints.
- Device-local model/sampler behavior.
- Final run status.

This layer should remain as small and stable as possible. The orchestration
system should invoke it; it should not know about whole-study scheduling.

## Proposed Data Model

The central abstraction should be a task plan. It can initially be JSONL or CSV,
and later SQLite if we need stronger querying/state semantics.

A task plan is immutable for a given attempt. If new tasks are added by an
adaptive controller, that should create an append-only extension with explicit
lineage rather than silently mutating historical rows.

### TaskSpec

Sketch:

```text
task_id: stable logical id
stage: train | eval | collect | select | final_train | final_eval | ...
attempt_id: attempt that owns this task
run_id: stable run/output id
config_path: base config path
overrides: ordered list or mapping of OmegaConf overrides
command_kind: run_py | collect_py | custom
dependencies: list of task ids or sentinel paths
outputs: expected outputs or sentinels
result_dir: canonical run directory
allowed_devices: cpu | cuda | [cpu, cuda]
preferred_device: optional
resume_policy: none | latest_complete_checkpoint | stage_specific
completion_policy: status_completed_with_checkpoint | status_completed | file_exists
metadata: study-specific scientific context
```

Important: the task should describe the logical run, not the Slurm command. A
task may later be run by Submitit, Snakemake, local shell, TaskVine, or Parsl.

### StagePlan

A stage plan is a collection of tasks plus stage-level metadata:

```text
study
stage
attempt_id
source_attempts
created_at
timezone
config_snapshots
tasks_path
n_tasks
smoke/full identity
schema_version
```

This is close to the current manifests, but the task table should be first-class
and backend-neutral.

### Task State

State should be separate from the plan. The plan is the scientific manifest.
State is execution bookkeeping.

Possible state fields:

```text
task_id
state: pending | running | completed | failed | stopped | skipped
attempt_id
backend
worker_id
slurm_job_id
array_task_id
device
hostname
pid
claim_owner
claim_time
lease_expires_at
start_time
end_time
returncode
checkpoint_used
checkpoint_produced
status_path
metrics_path
log_paths
error_type
error_message
```

This can be derived from existing files at first:

- `status.json`
- `launcher_status.json`
- `submission.json`
- checkpoint manifests
- Slurm logs

But if we later use dynamic worker pools, it may want to become SQLite or a
small durable state store.

## Completion and Resume

Completion and resume should be explicit concepts owned by `task_state.py` or a
similar module, not reimplemented in each stage.

For training:

```text
completed if:
  status.json says completed
  and at least one complete checkpoint exists
```

For evaluation:

```text
completed if:
  status.json says completed
  and expected metrics/report artifacts exist
```

For collect/report:

```text
completed if:
  expected table/report exists
  and manifest records consumed inputs
```

Resume policy for training:

```text
if completed:
  skip
elif complete checkpoints exist:
  run with load.path=<highest complete checkpoint>
  run with load.mode=train_resume
else:
  run fresh
```

Checkpoint selection should use complete checkpoint directories, not just
`latest.json`, because `latest.json` can be stale, partial, or point into a
failed write. The highest complete checkpoint is safer.

The execution layer should never delete partial checkpoints. Partial data is
scientific/debugging state and must be preserved.

Snakemake-specific note: if Snakemake is used, it should target small sentinel
files such as `completed.done`, not checkpoint directories. SpENN's Python code
should own checkpoint and metrics files. This avoids Snakemake cleanup behavior
interfering with partial QMC data.

## Resource Profiles

Hardware selection should be represented as a resource profile, not embedded in
stage logic.

Current practical profiles:

```text
cpu:
  uv environment: .venv
  uv extra: cpu
  runtime.device: cpu
  partitions: sapphire, kozinsky, seas_compute
  cpus: 16
  memory: 128G
  gpus: 0

cuda:
  uv environment: .venv-gpu
  uv extra: cu126
  runtime.device: cuda
  partitions: seas_gpu, kozinsky_gpu
  cpus: 8
  memory: 80G
  gpus: 1

submitit-launcher:
  uv environment: .venv-submitit
  purpose: light launcher only

smoke-cpu:
  partition: test

smoke-cuda:
  partition: gpu_test
```

This belongs in reusable execution configuration, not in every stage script.

The execution layer should own these thread exports for CPU workers:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
VECLIB_MAXIMUM_THREADS
```

Those should follow the allocated CPU count, not a hardcoded local default.

## Staged Workflow and Snakemake

Snakemake is attractive for the workflow layer if we simplify to one hardware
profile per workflow invocation.

It can naturally express:

```text
missing sentinel -> run job
existing sentinel -> skip
failed/incomplete sentinel -> rerun/resume via wrapper
all train sentinels -> enable eval
all eval sentinels -> collect
collect -> select
select -> final plan
```

A plausible Snakemake mapping:

```text
rule screen_plan:
  output: 00_grid/{attempt}/tasks.jsonl

rule screen_train:
  input: 00_grid/{attempt}/tasks.jsonl
  output: 01_train/{run_id}/{attempt}/completed.done

rule screen_eval:
  input: 01_train/{run_id}/{attempt}/completed.done
  output: 02_validation/{run_id}/{attempt}/completed.done

rule screen_collect:
  input: expand validation sentinels
  output: 03_collect/{attempt}/metrics.parquet

rule screen_select:
  input: 03_collect/{attempt}/metrics.parquet
  output: 04_select/{attempt}/champions.json

rule confirm_plan:
  input: 04_select/{attempt}/champions.json
  output: 05_final_grid/{attempt}/tasks.jsonl

rule confirm_train:
  input: 05_final_grid/{attempt}/tasks.jsonl
  output: 06_final_train/{run_id}/{attempt}/completed.done

rule confirm_eval:
  input: 06_final_train/{run_id}/{attempt}/completed.done
  output: 07_final_eval/{run_id}/{attempt}/completed.done
```

The rule commands should call Python wrappers that understand SpENN task plans
and resume semantics. Snakemake should not construct long OmegaConf override
commands itself.

Snakemake helps with:

- Stage DAGs.
- Missing output detection.
- Workflow-level resume.
- Slurm submission through profiles/executor plugins.
- Retry declarations.
- Rule resources.
- Per-stage dependencies.

Snakemake does not directly solve:

- CPU/GPU first-claim racing.
- Dynamic multi-hardware worker pools.
- Checkpoint resume unless our wrapper implements it.
- Scientific task planning.
- Adaptive search.

For the near term, Snakemake should be considered a future workflow backend, not
a prerequisite for the task-plan refactor.

## Optuna and Adaptive Search

Optuna is useful for search policy and trial bookkeeping, not for Slurm
orchestration.

The concern with Optuna is real: stochastic/adaptive scanners have dynamic task
tables. This is not incompatible with the split, but it changes the plan model.

For a fixed grid:

```text
plan once -> immutable task table -> execute missing rows
```

For Optuna:

```text
controller asks for trials
controller appends tasks
executor runs tasks
collector computes values
controller tells Optuna results
repeat
```

This means Optuna belongs in a controller/search layer, not in the execution
backend.

Recommendation:

- Keep fixed task tables for the current QMC screening and confirmation
  workflow.
- Optionally add an Optuna-backed adaptive refinement phase later.
- Treat Optuna trials as producers of task rows, not as direct Slurm jobs.
- Mirror completed SpENN tasks into Optuna user attrs/values only after the
  fixed-grid workflow is stable.

For now, avoid making the active study depend on Optuna. The reproducibility
benefit of a fixed manifest is more valuable.

## Dynamic Worker Pools

Dynamic heterogeneous assignment should stay on the backburner.

The clean version would be:

```text
submit CPU worker pool to CPU partitions
submit GPU worker pool to GPU partitions
workers pull compatible tasks from shared state
each worker runs tasks until walltime is nearly exhausted
```

This avoids loser jobs in CPU/GPU row racing, but it creates a real mini
scheduler. It requires durable task state, leases, heartbeats, stale lease
reclaim, resource-aware task assignment, and careful worker shutdown.

Tools that could help later:

- TaskVine / CCTools.
- Parsl HighThroughputExecutor.
- Parsl TaskVineExecutor.
- FireWorks for persistent workflow DB style execution.
- Dask-jobqueue for Python task pools, though it may be less natural for long
  external `run.py` commands.

Do not build this now unless queue latency becomes the dominant bottleneck after
the simpler plan/execution split.

## Suggested Module Boundaries

Initial target structure under `experiments/` or a reusable package:

```text
experiments/spenn_workflow/
  task_plan.py
  task_state.py
  resources.py
  execution/
    base.py
    local.py
    submitit.py
  workflow.py
  sentinels.py
  provenance.py
```

Or, if keeping it study-local first:

```text
experiments/hooke/pair_stability_v2/
  task_plan.py
  task_state.py
  resources.py
  execution.py
```

Long-term, common pieces should move out of a single study and become reusable
for SpENN experiments.

### task_plan.py

Owns:

- `TaskSpec`.
- `StagePlan`.
- Reading/writing task tables.
- Schema versions.
- Stable task IDs.
- Conversion from old manifests to task specs.
- Validation that task rows are complete and deterministic.

Does not own:

- Slurm.
- Submitit.
- Checkpoint status.
- Runtime device availability.

### task_state.py

Owns:

- Completion checks.
- Resume checkpoint resolution.
- Status file interpretation.
- Sentinel writing.
- Mapping run artifacts to task state.
- Failure classification.

Does not own:

- Which tasks exist.
- Which Slurm resource profile is used.

### resources.py

Owns:

- CPU/CUDA profile definitions.
- Smoke profile definitions.
- uv environment mapping.
- uv extra mapping.
- Slurm partitions.
- memory/CPU/GPU defaults.
- thread environment exports.

Does not own:

- Stage-specific task enumeration.

### execution/base.py

Owns:

- Execution backend interface.
- Common submission result model.
- Common task filtering rules.

Possible API sketch:

```python
class ExecutionBackend(Protocol):
    def submit(self, tasks: Sequence[TaskSpec], profile: ResourceProfile) -> SubmissionResult:
        ...
```

### execution/local.py

Owns:

- Debug/local execution.
- Running one task at a time.
- Optional local opportunistic claiming.

### execution/submitit.py

Owns:

- Submitit arrays.
- Chunking.
- Slurm executor parameters.
- Launcher environment re-exec.
- Job ID expansion.
- Submission metadata.

Initially this can be a refactor of the current `launch.py`.

## Migration Plan

The migration should be staged carefully because live experiments depend on the
current behavior.

### Phase 0: Freeze Active Behavior

Do not refactor active `pair_stability_v2` execution during the current run.

Allowed changes during active runs should be limited to:

- Bug fixes necessary to prevent wasted compute or data loss.
- Documentation.
- Read-only inspection scripts.
- Additive compatibility helpers that do not change default paths.

### Phase 1: Document and Add Tests Around Current Semantics

Before extracting abstractions, capture current behavior in tests:

- Completed final-train rows are skipped.
- Partial final-train rows resume from highest complete checkpoint.
- Failed/stopped claims are reclaimable.
- Existing claimed/running rows are skipped by competing submissions.
- CPU/CUDA profile resolution is deterministic.
- `.venv-submitit` launcher re-exec is preserved.
- Smoke and full latest pointers remain separate.
- Final eval only consumes ready final-train checkpoints.

These tests become the safety net for refactoring.

### Phase 2: Extract Task State Helpers

Move reusable completion/resume helpers out of stage scripts:

- complete checkpoint discovery
- latest complete checkpoint selection
- final train completion check
- eval readiness check
- stale claim classification

This is low risk because it does not yet change task generation.

### Phase 3: Introduce TaskSpec Without Changing Launch Behavior

Have `final_train.py` build `TaskSpec` objects internally, then convert them
back to the existing command/status path lists.

This allows tests to assert the task table without changing Submitit behavior.

### Phase 4: Write Task Tables as Stage Artifacts

In addition to current files, write:

```text
05_final_grid/<attempt>/tasks.final_train.jsonl
07_final_eval/<attempt>/tasks.final_eval.jsonl
```

or similar. Do this additively at first.

The existing scripts can continue to use legacy manifests until the new task
tables have been validated.

### Phase 5: Make Execution Consume Task Tables

Refactor launch submission to accept `TaskSpec` objects. The execution backend
then handles:

- command materialization
- environment wrapping
- status path selection
- chunking
- submission metadata

At this point, the scientific plan and execution mechanism are mostly separated.

### Phase 6: Generalize Across Scan Train / Validate

Apply the same pattern to:

- `train.py`
- `validate.py`
- `final_train.py`
- `final_eval.py`

Collect/select/report stages can follow later because they are lower fan-out.

### Phase 7: Consider Snakemake Wrapper

Once task specs and sentinel logic exist, prototype a Snakemake wrapper that
calls the same Python task runner. Do not make Snakemake the source of scientific
truth. It should consume task tables and sentinel outputs.

### Phase 8: Consider Adaptive Search or Worker Pools

Only after the fixed workflow is stable:

- Add Optuna/controller layer for adaptive task generation.
- Add TaskVine/Parsl dynamic workers if queue pressure justifies it.

## Sentinels

A sentinel file is a small file whose existence means a task completed according
to SpENN semantics.

Examples:

```text
01_train/<run_id>/<attempt>/completed.done
02_validation/<run_id>/<attempt>/completed.done
06_final_train/<final_run_id>/<attempt>/completed.done
07_final_eval/<final_run_id>/<attempt>/completed.done
```

Sentinel content should be JSON, not empty:

```json
{
  "task_id": "...",
  "stage": "06_final_train",
  "attempt_id": "...",
  "status": "completed",
  "status_path": ".../status.json",
  "checkpoint_dir": ".../checkpoints/step_001999",
  "completed_at": "..."
}
```

The sentinel should be written only after checking the real run artifacts. It is
a workflow-facing completion marker, not the source of truth for scientific
metrics.

## Naming

The current stage names are serviceable but can be clarified:

- `train` / `validation` for screening.
- `collect` / `select` for reduction and analysis.
- `final_train` / `final_eval` for confirmation.

Possible future names:

```text
screen_plan
screen_train
screen_eval
screen_collect
screen_select
confirm_plan
confirm_train
confirm_eval
confirm_collect
report
```

Do not rename active directories during the current run. If renaming happens,
it should be part of a new study layout version.

## Industry-Norm Shape

For ML-style experimental systems, a common separation is:

```text
search space / experiment design
trial/task registry
executor/scheduler
run artifact store
metrics collector
analysis/reporting
```

SpENN can follow this pattern without adopting a full MLOps stack.

Suggested mapping:

```text
search space / design:
  plan.py, final_plan.py, future Optuna controller

trial/task registry:
  task tables and stage manifests

executor/scheduler:
  Submitit now, maybe Snakemake or TaskVine later

artifact store:
  existing results directories, checkpoints, metrics

metrics collector:
  collect.py, final_collect.py

analysis/reporting:
  select_champions.py, final_report.py
```

The artifact store should remain file-based for now. It is transparent,
portable, and suitable for scratch/storage workflows on Cannon.

## Questions to Resolve Before Implementing

1. Should task tables be JSONL, CSV, or SQLite?
   - JSONL is easiest for nested metadata and append-only behavior.
   - CSV is easier for quick inspection.
   - SQLite is better for dynamic worker pools and leases.
   - Recommendation: JSONL for immutable plans now; SQLite only if dynamic
     workers become necessary.

2. Should task IDs include attempt IDs?
   - The logical task ID should probably not include attempt ID.
   - The attempt ID should identify the plan instance.
   - Output paths may include attempt ID.

3. What is the canonical completion artifact?
   - Existing `status.json` should remain authoritative for run completion.
   - Sentinels can be workflow-facing derived artifacts.

4. Should execution state be stored centrally?
   - For Submitit arrays, existing per-run files are enough.
   - For dynamic workers, central state becomes more important.

5. How much should be reusable across experiments?
   - Planning utilities may remain study-specific.
   - TaskSpec, task_state, resources, and execution backends should be reusable.

6. How should smoke runs relate to task plans?
   - Smoke should likely be a plan transform: select first N tasks and apply
     smoke overrides.
   - Smoke should never overwrite full latest pointers.

7. Should CPU/CUDA mixed racing survive?
   - It is useful under queue pressure but operationally awkward.
   - The simpler long-term mode is one hardware profile per workflow invocation.
   - If mixed execution remains important, prefer worker pools over duplicate
     per-row candidate jobs.

## Non-Goals for the First Refactor

- Do not rewrite the training loop.
- Do not change checkpoint file format.
- Do not change result directory layout for active runs.
- Do not adopt Optuna immediately.
- Do not adopt Snakemake immediately.
- Do not build dynamic worker pools immediately.
- Do not require a database.
- Do not delete or migrate existing result data.

## Near-Term Recommendation

After the current run is complete, implement the split in this order:

1. Extract task-state/checkpoint-resume helpers with tests.
2. Introduce `TaskSpec` internally for final train/eval.
3. Write additive task tables next to existing manifests.
4. Refactor Submitit execution to consume task specs.
5. Port scan train/validate to the same machinery.
6. Add sentinel generation.
7. Evaluate Snakemake as a wrapper for one-hardware-profile workflows.

The guiding rule: every step should be behavior-preserving unless explicitly
creating a new study layout version.

## Summary

The durable split should be:

```text
workflow:
  stage DAG

planning:
  immutable or append-only task tables

execution:
  backend-specific scheduling and resources

run:
  one concrete config, checkpoints, metrics
```

This lets SpENN keep the current transparent file-based experiment layout while
making orchestration less brittle. It also keeps future choices open:

- fixed grids,
- Optuna/adaptive controllers,
- Submitit arrays,
- Snakemake stage orchestration,
- TaskVine/Parsl worker pools.

The immediate priority is not choosing a tool. The immediate priority is making
the scientific task plan explicit and making execution consume that plan through
a stable interface.
