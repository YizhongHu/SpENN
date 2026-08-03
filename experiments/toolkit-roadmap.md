# Experiment Toolkit / V4 Restructuring Roadmap

> **Status:** canonical running document; current-surface refresh 2026-07-11.
> **Review state:** approved 2026-07-11. V4 implementation is intentionally
> paused until later explicit direction.

## Authority, scope, and review workflow

This file is the canonical, actively maintained record for:

- observed implementation surfaces;
- v3/v4 parity policy and evidence gaps;
- accepted sequencing, decisions, and deferred work; and
- the next reviewable restructuring slice.

It is deliberately not a second copy of an issue-body design essay.

- [Issue #113 — Experiment Stack Design](https://github.com/YizhongHu/SpENN/issues/113)
  retains the refactoring **manifesto**: motivation, ownership boundaries,
  immutable-plan/state separation, resource/execution rules, independently
  runnable artifact stages, and the reasons not to let a scheduler framework
  own scientific planning. It is the review and coordination record, not the
  mutable phase-status ledger.
- [Issue #114 — Experiment Refactor Plans](https://github.com/YizhongHu/SpENN/issues/114)
  remains supporting historical design/phase context. Its mutable status claims
  have been reconciled here; do not use it instead of this current-surface and
  parity ledger.
- `TODO.md` is a local scratch note. It may point here, but it is not a third
  roadmap.

When this document is published to the repository branch, issue #113 should
link to that reviewed revision. Until then, its GitHub text names the repository
path rather than pretending an unpublished blob is available remotely.

### Status vocabulary

Use these labels precisely.

| Label | Meaning |
| --- | --- |
| **Observed** | Implemented in the current repository surface. |
| **Verified gate** | Protected by a focused test or documented completed comparison. It does not imply a fresh end-to-end run happened today. |
| **Unverified** | Code exists, but the stated broader parity or live-run evidence was not exercised. |
| **Planned** | Intended work; no implementation claim. |
| **Deferred / parity-blocked** | Intentionally postponed because it would alter a protected behavior or artifact surface. |
| **Decision** | A reviewed policy that later phases must obey. |

## Manifesto-derived non-negotiables

The following principles are the operational form of issue #113's manifesto.
Every v4 design and review must satisfy them.

1. **Scientific planning and operational execution are different owners.**
   Planning owns scientific grid axes, blinding, seeds, candidates, logical run
   identity, dependencies, and expected outputs. Execution owns backend,
   resources, queueing, claims, retries, resume application, and submission
   provenance. An executor never decides which scientific rows exist.
2. **Durable artifacts, not in-memory orchestration, are the cross-stage
   contract.** A plan is immutable for an attempt; task state is separate;
   every granular stage remains independently runnable from selected upstream
   artifacts. Convenience orchestrators may call stage entrypoints but may not
   create a private workflow path.
3. **A task describes a logical run, not a scheduler command.** `TaskSpec` and
   later trial contracts must remain backend-neutral. Slurm, Submitit, local
   execution, or a future adapter are execution choices around the same logical
   task.
4. **`run.py` stays the single train/eval run entrypoint.** Experiment-layer
   Hydra builds experiment objects; it does not become the scheduler, replace
   the run layer, or make multirun the authoritative execution mechanism.
5. **Result data is evidence.** V4 uses a distinct results root and never
   rewrites, renames, deletes, repairs, or repoints live v3 results. Historical
   v2/v3 data remains readable. The 2026-07-10 path repair was limited to an
   archived v2 migration-backup symlink; it is not precedent for mutating an
   active study.
6. **Do not unify semantics merely because names look similar.** Completion,
   checkpoint readiness, attempt selection, smoke/full pointers, and result
   populations have distinct owners and observable contracts.
7. **Fixed-grid behavior comes first.** Adaptive search remains optional and
   must sit behind the same durable trial/execution artifact boundary. Do not
   add Optuna or another dependency without explicit approval and a reproducible
   lockfile update.
8. **Parity evidence is required before removal.** Structural/scientific
   mismatches require an explicit layout-version decision; broadening a
   normalizer to hide a mismatch is not an acceptable fix.

## Current implementation surface

### Study roles

| Surface | Status | Role |
| --- | --- | --- |
| `experiments/hooke/pair_stability_v2` | **Historical reference; retired as a live comparison target** | The pre-profiling v2/v3 end-to-end confirmation passed. Profiling fields deliberately added to v3 afterwards make a fresh v2 compare invalid by design. Preserve v2 data and code; do not use it as the current v4 gate. |
| `experiments/hooke/pair_stability_v3` | **Observed; live behavioral baseline** | Current runnable ten-stage study. Future v4 work compares back to a frozen completed v3 lineage. It is parity-sensitive and not a migration target. |
| `experiments/hooke/pair_stability_v4` | **Planned; absent** | Future isolated study surface. No v4 directory, config tree, runner, or result root is currently implemented. |
| `experiments/toolkit` | **Observed; reusable execution/provenance foundation** | Owns current durable planning, resources, executor adapters, task state, execution records, lineage, and generic compact-artifact utilities. It does not yet contain the proposed v4 trial contracts, generic stage runner, or fixed-grid strategy stack. |

### Observed toolkit contracts and owners

| Owner | Current surface | Boundary it owns |
| --- | --- | --- |
| `experiments/toolkit/specs.py` | `CompletionSpec`, `TaskSpec`, `StagePlan`, `experiment-toolkit/v1`, `stage_manifest.json`, `tasks.jsonl`, deterministic `<stage>:<run_id>:<attempt_id>` task ids | Immutable stage/task plans, validation, serialization, command-to-task materialization. |
| `experiments/toolkit/resources.py` | `ResourceSpec` | Backend-neutral profile, device, partition, threads, memory, GPUs, timeout, and uv environment intent. |
| `experiments/toolkit/executors.py` | `Executor`, `SubmissionRequest`, `LauncherExecutor`, `LocalExecutor`, `SubmititExecutor` | Submission request/task alignment and adapter dispatch. The current adapter deliberately delegates to the legacy pair-stability launcher callable to preserve behavior. |
| `experiments/toolkit/execution.py` | `ExecutionRecord`, `execution_records.jsonl` helpers | Per-task submitted-command, backend, claim-path, and job-id provenance. |
| `experiments/toolkit/task_state.py` | Claims, deadline guards, checkpoint discovery, completion/readiness, and resume helpers | Layout-agnostic state predicates. It must not choose which study attempt to inspect. |
| `experiments/toolkit/lineage.py` | `TaskLineageRow`, `task_lineage.jsonl` | Additive low-fan-out provenance from collected row through candidate and final job. |
| `experiments/toolkit/artifacts.py`, `selection.py`, `cost.py` | Generic compact artifact/metric, selection, and cost utilities | Reusable utilities only. They are not yet the proposed `MetricCollector`, `Selector`, or `Reporter` protocol/object layer. |

### Observed v3 stage DAG and durable boundaries

The live artifact graph is intentionally granular:

```text
00_grid plan.py
  -> 01_train train.py
  -> 02_validation validate.py
  -> 03_collect collect.py
  -> 04_select select_champions.py
  -> 05_final_grid final_plan.py
  -> 06_final_train final_train.py
  -> 07_final_eval final_eval.py
  -> 08_final_collect final_collect.py
  -> 09_final_report final_report.py
```

| Stage | Inputs / durable outputs | Current toolkit role | V4 protection rule |
| --- | --- | --- | --- |
| `00_grid` | Scientific grid, materialized configs, commands, `manifest.json`, `unblind.json` | None beyond downstream task materialization | Preserve planned population, ordered commands, axes, blind mapping, run ids, and smoke/full pointer behavior. |
| `01_train` | Grid manifest; per-run status, checkpoints, metrics, submission provenance | Persists a `StagePlan`, `tasks.jsonl`, and `execution_records.jsonl`; submits through an executor adapter | Preserve task rows, completion metadata, resources, claim/status paths, commands, and submissions. |
| `02_validation` | Train checkpoints plus grid lineage; validation metrics/diagnostics/status | Same fan-out plan/executor/record boundary as train | Preserve selected train attempt and checkpoint readiness semantics. |
| `03_collect` | Explicit validation/grid attempts; `summary.csv`, `failures.csv`, `collection_report.json` | Adds `task_lineage.jsonl` without changing compact outputs | Do not replace manifest population with launched task rows without an explicit layout/parity decision. |
| `04_select` | Collection compact artifacts; `champions.csv`, `selection_report.json` | Adds lineage and uses toolkit selection utilities | Preserve metric names, ranking inputs, row schema/order, and candidate source lineage. |
| `05_final_grid` | Champions; `source_champions.csv`, `final_jobs.csv`, `manifest.json` | Adds final-job lineage | Preserve candidate-to-final-seed expansion, run ids, and source attempts. |
| `06_final_train` | Final-grid jobs; final status/checkpoints/submission provenance | Fan-out plan, executor adapter, and execution records | Preserve completed-row exclusion and partial-checkpoint resume semantics. |
| `07_final_eval` | Ready final-train checkpoints; final evaluation artifacts | Fan-out plan, executor adapter, and execution records | Preserve stricter selected-checkpoint, `COMPLETE`, and checkpoint-manifest readiness gate. |
| `08_final_collect` | Raw final train/eval artifacts; compact tables and manifest | Study-local collection boundary | Preserve tables and one-pass consumption behavior. |
| `09_final_report` | Compact final-collection tables; Markdown, CSV tables, PNGs, JSON | Study-local reporting boundary | Preserve compact-input-only behavior and report artifact schemas. |

Fan-out plan directories live under `<stage>/stage_plans/<attempt_id>` and are
additive orchestration provenance. They do not replace the stage's scientific
output directories.

### Current routing status

**Observed:** all four fan-out stages (`01_train`, `02_validation`,
`06_final_train`, `07_final_eval`) construct/persist `StagePlan` rows, submit
selected tasks through `LocalExecutor` or `SubmititExecutor`, and write
execution records. The executor path still uses the legacy launcher as an
adapter implementation. This is an established boundary, not evidence that the
legacy launcher has been replaced or that v4 exists.

**Observed:** low-fan-out `collect`, `select`, and `final_plan` preserve their
public compact artifacts and add lineage sidecars. Generic artifact/selection
utilities exist, but collector/selector/reporter object contracts do not.

## Mandatory acceptance gate: frozen v3 non-pilot smoke → fresh v4 E2E

### Decision: every experiment-stack change requires v3/v4 E2E evidence

`pair_stability_v3` is the only current v4 behavioral reference. The retained
v2/v3 comparator is useful machinery, but its v2 target is retired: the final
pre-profiling v2/v3 confirmation passed, and intentional v3 profiling columns
make a new v2 comparison invalid by design. Do not run a fresh v2 comparison as
a v4 gate.

Any source, config, contract, launcher, collector, reporter, stage-runner, or
artifact-layout change that participates in the experiment stack requires a
fresh **v4 end-to-end smoke run** compared with the frozen v3 smoke reference
before acceptance. Unit, property, plan-diff, and dry-run tests remain
required, but they are supplementary evidence and never replace this gate.
Documentation-only changes do not require an E2E run.

If a proposed slice has no executable v4 smoke route that exercises the changed
surface, it is not ready to accept. Expand the slice to make that route
executable or defer the change; do not merge an unexercised stack abstraction
on the assumption that a later routing phase will prove it.

### Frozen deterministic non-pilot v3 smoke reference

The canonical reference configuration is the non-pilot:

```text
experiments/hooke/pair_stability_v3/configs/smoke.yaml
```

Do not substitute `pilot.yaml` or `pilot_smoke.yaml` unless an explicit
decision records why the non-pilot smoke cannot be managed. The smoke config
retains the full-study axis shape—basis, update normalization, feature
normalization, learning rate, channels, and activation—while fixing one named
scan seed row, two train steps, and one final replicate. Its current plan
contract is 64 scan jobs with `final_replicates: 1`. It must be planned with an
explicit `--blind --blind-seed 811` and explicit stage attempt ids.

V4-0 creates or selects one reviewed completed v3 smoke lineage and freezes a
compact, read-only reference snapshot. That snapshot is generated from v3 once;
subsequent v4 acceptance runs compare against it and **must not rerun v3**.
The planned owner is:

```text
experiments/hooke/pair_stability_v4/reference/v3_smoke/<reference-id>/
```

The snapshot contains only the comparison inventory and a `reference.json`;
it does not copy checkpoints or mutate `pair_stability_v3/results`. The
metadata must record:

- v3 commit and dirty-state/provenance identity;
- source results root and every explicit v3 stage attempt id;
- `smoke.yaml`, resolved train/validation config snapshots, and hashes;
- blind seed, named seed assignments, grid/job counts, and logical stage map;
- backend/device/profile, PyTorch/CUDA environment identity, submitted command
  manifest, and source artifact checksums; and
- the comparator schema, permitted volatile fields, and layout-map version.

The fixed inputs make the reference deterministic at the scientific/configuration
level. GPU and BLAS arithmetic can still vary at floating-point scale; that is
why only measured floating values use the documented tolerance below rather
than claiming bitwise GPU reproducibility.

### E2E execution profiles

| Profile | Default use | Acceptance rule |
| --- | --- | --- |
| `gpu_test` | Preferred full non-pilot smoke E2E. Follow the v3 smoke stack's Submitit/CUDA shape and submit all fan-out work to `gpu_test`. | V4 compares only with a frozen `gpu_test` v3 reference. Scheduler job ids, host data, and timing are volatile; device, dtype, scientific parameters, and logical resource intent are not. |
| `test` | Preferred CPU E2E where the changed surface is CPU-specific or a CPU profile is needed. | V4 compares only with a frozen `test` v3 reference. It may not treat CPU/GPU numerical or submission differences as a cross-profile normalizer. |
| `local` | Acceptable only when `test`/`gpu_test` cannot be used. | Record a matching local v3 reference once, then compare fresh local v4 output to it. Scheduler-only fields are not applicable; a later available scheduler E2E must backfill executor/resource evidence before cutover. |
| non-test production | Required for full scientific batches and production studies. CPU work uses `sapphire`, `kozinsky`, or `seas_compute`; GPU work uses `kozinsky_gpu` or `seas_gpu`, subject to the selected resource profile. | This is not an acceptance-test substitute. Persist its exact profile/provenance; do not let `test` or `gpu_test` become a full-run default. |

The comparison profile is part of `reference.json`. A v4 run never compares
against a baseline produced under a different profile.

### Layout map and comparison inventory

V4 may use different directory names, stage names, config filenames, and
artifact filenames. It may not use that freedom to omit, merge, or silently
reinterpret a v3 logical artifact. Each frozen reference owns a versioned,
reviewed `layout_map.json` with one-to-one logical roles:

```text
v3 00_grid                    -> v4 screen_plan (or declared equivalent)
v3 01_train                   -> v4 screen_train
v3 02_validation              -> v4 screen_eval
v3 03_collect                 -> v4 screen_collect
v3 04_select                  -> v4 select
v3 05_final_grid              -> v4 confirm_plan
v3 06_final_train             -> v4 confirm_train
v3 07_final_eval              -> v4 confirm_eval
v3 08_final_collect           -> v4 confirm_collect
v3 09_final_report            -> v4 report
```

For every logical role, the map names the v3 reference path, v4 candidate path,
comparison mode, and any approved path-token substitution. Changing the map is
a parity-policy change: it needs a review rationale and a negative comparator
test, not a broad ignore rule.

The mandatory inventory is:

- `00_grid` `manifest.json` and `unblind.json`;
- all four fan-out stages' `stage_manifest.json`, `tasks.jsonl`,
  `execution_records.jsonl`, and per-run `submission.json` records;
- `03_collect` `summary.csv`, `failures.csv`, and
  `collection_report.json`;
- `04_select` `champions.csv` and `selection_report.json`;
- `05_final_grid` `source_champions.csv`, `final_jobs.csv`, and
  `manifest.json`;
- `08_final_collect` `manifest.yaml` and every compact final-collection table;
- `09_final_report` `final_report.json`, `report.md`, and every protected
  report table; and
- planned job counts, per-stage artifact counts, and explicit upstream attempt
  lineage.

### Exact, mapped, and volatile comparison rules

| Surface | Must be exact after the reviewed layout map | May change only through the map or reference metadata | May be normalized |
| --- | --- | --- | --- |
| Stage graph and lineage | Stage dependencies, independently rerunnable boundaries, source-attempt selection, intended/launch populations, and per-stage counts | Directory/stage labels | Never infer lineage from newest directories. |
| Scientific plan | Grid axes/values, blind mapping, named seeds, resolved scientific config values, task/run semantics, ordered planned population, and candidate/confirmation expansion | Study/root/config path tokens and layout-specific ids | No scientific field, seed, axis, or ordering normalization. |
| Commands and execution | Ordered command argv after approved root/config-path substitutions; entrypoint, overrides, device intent, backend semantics, completion/resume behavior, claim/status paths, and logical resource request | Study/result/config paths and declared v4 wrapper path | Scheduler job ids, PIDs, hostnames, timestamps, and recorded wall-time fields. |
| Structured artifacts | JSON key sets/values, CSV headers, CSV row order, nonnumeric text, report content, artifact presence, and logical schemas | File/directory names and object-key serialization order | Only fields enumerated in the reference's volatile-field allowlist. |
| Numeric artifacts | Exact discrete/count/integer semantics; floating metric locations and meanings | None | `rel_tol=1e-9`, `abs_tol=1e-12` for numeric values only. |
| V4-only provenance | It must not replace a mapped reference artifact. | Additive sidecars under declared v4 metadata locations | Its existence is allowed; its declared schema is checked separately. |

The volatile allowlist begins with the current comparator's scheduler/runtime
fields: attempt/root/study tokens, `created_at`, `start_time`, `end_time`,
`submitted_at`, host/PID, scheduler job ids, remaining time, `wall_time_sec`,
and explicitly declared profiling timing fields. A volatile metric may suppress
only its directly derived metric-value/champion fields. CSV headers, JSON
schema, nonvolatile metrics, scientific text, and row order never become
volatile merely because they are inconvenient to compare.

### Per-slice evidence

| Slice | Mandatory fresh E2E evidence against frozen v3 `smoke.yaml` | Additional focused evidence |
| --- | --- | --- |
| **V4-0: reference harness and isolated root** | Generate/select the one reviewed v3 smoke reference; run the first isolated v4 smoke route and compare its mapped inventory. | Comparator self-compare; negative header/row/metric/map tests; root-ownership test. |
| **V4-1: trial/seed/candidate contracts** | Full v4 smoke E2E exercising the new serialized contract artifacts. | JSON/CSV round trips, validation failures, import independence. |
| **V4-2: generic runner/config composition** | Full v4 smoke E2E through the selected-stage runner. | Selective instantiation and dry-run/no-submission tests. |
| **V4-3: fixed-grid planning/materializer** | Full v4 smoke E2E from v4 plan through report. | Exact plan/command/run-id diff and smoke/full pointer tests. |
| **V4-4: low-fan-out entrypoints** | Full v4 smoke E2E with the changed stage route. | Stale/unrelated-lineage fixtures and policy rerun tests. |
| **V4-5: fan-out entrypoints/executor factory** | Full v4 smoke E2E on the matching scheduler profile; local is allowed only under the local-profile rule above. | CPU/CUDA/mixed claim, request/task alignment, resume, and failure-state tests. |
| **V4-6: cutover decision** | A final complete non-pilot v3-smoke/v4-smoke comparison on every required scheduler profile. | Independent-stage rerun and resume/requeue evidence. |

### Non-unification rules

1. **Launch completion:** a run is skip-complete only when `status.json` says
   `completed` and `checkpoints/latest.json` exists.
2. **Scan validation readiness:** validation can require the latest checkpoint
   pointer without the final-train sentinel criteria.
3. **Final-train completion/resume:** completion requires completed status plus
   the highest complete `step_*` checkpoint; partial rows resume from that
   checkpoint with the established `load.path` and `load.mode=train_resume`
   behavior.
4. **Final-eval readiness:** evaluation additionally requires the selected
   checkpoint record, a valid ready checkpoint, `COMPLETE`, and its checkpoint
   manifest.
5. **Attempt resolution is separate from readiness:** explicit attempt ids,
   `latest.json`, smoke/full matching, and sorted fallback select which attempt
   to inspect. They are not generic completion policies.
6. **Manifest population is separate from launched task population:** grid
   manifests represent intended jobs; `tasks.jsonl` represents launched,
   readiness-filtered fan-out jobs. Replacing one with the other changes
   collection semantics.
7. **Controller sequencing is separate from stage equivalence:** scheduler
   waits and expected-count guards do not replace independently invocable CLI
   stages or artifact contracts.

### Open design: batch checkpoint-candidate selection

The active discussion is
[`checkpoint-selection-options.md`](checkpoint-selection-options.md). Current
requirements are optimizer-initiated batches, selector visibility over the
complete published-checkpoint candidate set, JSONL-backed selected-candidate
references, periodic pruning only after terminal evaluation/selection, and
energy plus local-energy variance as the default objective vector.

Dynamic scheduling is explicitly deferred. The first future contract slice may
define only serializable checkpoint references/publications, retention intent,
batch/candidate/evaluation/selection records, and pure selector/objective
reducer interfaces. It must not add an event poller, worker loop, queue,
database, controller service, or runtime evaluation materializer.

The ownership split remains open: the proposed narrow core extension publishes
and retains generic completed-checkpoint facts; the experiment stack owns batch
identity, selection/Optuna policy, and any later evaluation scheduling,
collection, and optimizer `tell`. No checkpoint-selection runtime
implementation starts until that boundary is decided and added to this roadmap
as an accepted decision.

## Current evidence and historical landed work

| Area | Status | Current interpretation |
| --- | --- | --- |
| Historical reduced v2/v3 confirmation | **Verified gate; retired target** | The final pre-profiling comparison passed. Preserve it as provenance; do not re-run it after intentional v3 profiling-schema divergence. |
| Durable task/stage contracts | **Observed; verified by focused toolkit tests** | `StagePlan`/`TaskSpec`/`CompletionSpec`, resources, schema checks, deterministic ids, read/write, and execution-record validation are current reusable foundations. |
| Executor routing | **Observed; verified by focused v3/toolkit tests** | All four v3 fan-out stages use executor adapters. The legacy launcher remains behavior-preserving implementation detail behind the adapter. |
| Task-state relocation | **Observed; verified by focused tests** | Claim, deadline, checkpoint, and resume helpers moved into toolkit without semantic unification. Smoke/full pointer interpretation and sentinel derivation remain separate planned questions. |
| Additive lineage | **Observed; verified by focused tests** | Collect/select/final-plan write `task_lineage.jsonl` while protected compact outputs remain unchanged. |
| Generic compact utilities | **Observed, partial** | Artifact/metric, selection, and cost utilities are reusable. Collector/selector/reporter protocolization and cross-study proof are still planned. |
| End-to-end v3/v4 parity | **Planned** | No v4 surface exists; no v3/v4 comparison has been run. |

## Restructuring plan

Each implementation slice is a small reviewable change. Every stack slice must
finish with a fresh full ten-stage v4 run of the non-pilot `smoke.yaml` shape
against the frozen matching-profile v3 reference. Before coding, add its exact
source paths, expected artifacts, layout-map entries, and E2E evidence here;
after verification, replace **Planned** with the appropriate observed status
and record any evidence gap. A phase label, unit test, plan diff, or dry-run is
not proof without the mandatory E2E comparison.

### V4-0: establish the non-pilot smoke reference, comparator, and v4 bootstrap

**Status:** Planned. This is the first implementation prerequisite, not a
documentation-only self-comparison.

1. Generate or select one completed v3 `configs/smoke.yaml` lineage with fixed
   `--blind-seed 811`, explicit attempt ids, and a recorded matching execution
   profile; freeze its compact reference snapshot without touching v3 results.
2. Refactor the retained comparator into an explicit v3-reference/v4-candidate
   tool driven by `reference.json` and `layout_map.json`.
3. Add negative tests for an altered CSV header, row order, nonvolatile metric,
   artifact-map entry, scientific seed, and v4 root violation.
4. Build the smallest isolated **real v4** smoke route that produces the mapped
   ten-stage inventory. It must execute the v4 path, not merely compare v3
   artifacts to themselves.

**Acceptance:** the v3 reference snapshot has recorded checksums and profile;
the fresh v4 non-pilot smoke E2E passes the mapped comparison; every negative
case fails; and root-ownership tests prove v4 cannot write below the v3 root.

### V4-1: add experiment-stack contracts above task plans

**Status:** Planned. First additive candidate after an explicit resume of
implementation; its new contract artifacts must be exercised by the v4 smoke
route, not merely unit-tested in isolation.

Add toolkit-owned, serializable contracts:

- `TrialProposal`, `SeedAssignment`, `TrialSpec`, `SeedResult`, `TrialResult`,
  `CandidateSpec`, `StageContext`, and `StageResult`;
- `SearchSpace`, `SearchStrategy`, `SeedPolicy`, `OverrideMapper`,
  `TrialMaterializer`, `MetricCollector`, `Objective`, `Selector`, `Reporter`,
  and `StageEntrypoint` protocols;
- compact names `trial_proposals.jsonl`, `trials.jsonl`, `seed_results.csv`,
  `trial_results.csv`, `candidates.csv`, `stage_result.json`, and report
  manifests; and
- required-metric validation that a future selector/reporter can declare and a
  collector can reject early.

**Constraints:** immutable/serializable records; no `spenn`, pair-stability,
Optuna, or Hydra-runtime import; no change to v3 scripts, protected compact
artifacts, `StagePlan`, `TaskSpec`, or execution semantics.

**Acceptance:** every record has versioned JSON/CSV round trips and validation
failures; fresh-process toolkit imports stay independent; **and a fresh full
v4 non-pilot smoke E2E that writes/reads the new contract artifacts passes the
frozen-v3 comparison.**

### V4-2: complete generic compact-artifact interfaces and stage composition

**Status:** Planned.

1. Finish reusable metric projection, trial aggregation, selector-input, and
   report-input boundaries on top of V4-1 contracts.
2. Keep pair-stability plots, metric names, and scientific defaults study-local.
3. Add experiment-layer Hydra config under
   `experiments/hooke/pair_stability_v4/experiment_configs/`, separate from
   single-run SpENN configs.
4. Add `experiments.toolkit.run_stage` to compose one selected stage,
   construct `StageContext`, instantiate only that `StageEntrypoint`, invoke
   `run(context)`, and write a resolved stage manifest.
5. Begin with dry-run `select`/`report`; they must not instantiate an executor,
   search strategy, or seed policy.

**Acceptance:** deterministic compact-table fixtures; required-metric errors;
selective-instantiation tests; dry-run manifest with zero submission; **and a
fresh full v4 non-pilot smoke E2E passes the frozen-v3 comparison.**

### V4-3: express fixed-grid planning through v4 stack objects

**Status:** Planned.

Implement `GridSearchSpace`, `GridStrategy`, `NamedSeedPolicy`,
`AxisOverrideMapper`, and a pair-stability materializer. Feed them the readable
scientific grid, blinding policy, named seeds, base configs, and smoke transform
that v3 uses. Emit v4-only trial/candidate sidecars plus current-compatible
planning artifacts in the isolated v4 root.

**Acceptance:** reduced-grid commands, overrides, run ids, task counts,
source-attempt lineage, and plan artifacts compare exactly with frozen v3 after
only reviewed layout-map substitutions; smoke never moves a full latest
pointer; **and a fresh full v4 non-pilot smoke E2E passes.**

### V4-4: route low-fan-out v4 stages through durable entrypoints

**Status:** Planned. Routing order: `select` → `report` → `screen_collect` →
`confirm_collect`.

Each entrypoint reads explicit upstream artifact references, writes its own
v4 stage manifest/provenance, and remains rerunnable without train/eval. It
must preserve compact schemas and use additive v4 sidecars for new data.

**Acceptance:** each stage rejects a stale or unrelated lineage fixture,
reruns independently, produces comparator-equivalent compact output, **and a
fresh full v4 non-pilot smoke E2E passes the complete mapped inventory.**

### V4-5: route v4 planning and fan-out through stage entrypoints

**Status:** Planned. Routing order: `screen_plan` → `confirm_plan` →
`screen_train` → `screen_eval` → `confirm_train` → `confirm_eval`.

Extract v4-owned resource/executor construction into an `ExecutorFactory`, but
retain the legacy launcher adapter until an independently reviewed backend
change exists. Construct commands and apply readiness/resume selection before
executor dispatch. Preserve the distinct rules in **Non-unification rules**.

**Acceptance:** CPU, CUDA, mixed CPU/CUDA, fresh, completed, partial, failed,
stopped, and missing-state cases preserve task selection, commands, resources,
claim paths, status paths, completion metadata, and execution records. **Every
accepted fan-out change also requires a fresh full v4 non-pilot smoke E2E on
the matching profile; local follows the documented fallback rule.**

### V4-6: prove reduced end-to-end parity before any cutover

**Status:** Planned.

Run the independently invocable v4 stage DAG using the non-pilot
`configs/smoke.yaml` reference shape and compare the complete frozen surface
through `09_final_report`. Keep scheduler dependency/wait behavior close to v3;
reduce only the grid scale already encoded by that smoke config.

**Acceptance:** all protected artifacts, submission/provenance records, and
stage-plan checks pass under the documented mapped comparator; resume/requeue
cases pass; every v4 stage is rerunnable from durable artifacts; and the final
required `test`/`gpu_test` profile comparisons are clean.

### V4-7: retire v4-internal duplication only after proof

**Status:** Deferred / parity-blocked.

After V4-6, remove only v4-internal helpers proven unreachable. Keep v3 as a
readable behavioral reference until an explicit project decision says
otherwise. Do not rename active result directories in place; any conceptual
stage-name layout change requires a distinct layout version and compatibility
readers.

### V4-8: optional adaptive search

**Status:** Optional; blocked on stable fixed-grid stack and explicit dependency
approval.

An `OptunaStrategy` or another adaptive strategy may implement the same trial
contracts only when a concrete study needs it. It must not submit jobs directly,
be imported by fixed-grid studies, or replace the executor/artifact boundary.

## Explicitly deferred

- Snakemake, TaskVine, Parsl, Ray Tune, W&B Sweeps, Kubernetes controllers, or
  other workflow platforms as primary orchestration.
- Central SQLite task state for fixed-grid workflows.
- Active pruning or early stopping.
- Migration, deletion, or in-place renaming of historical results.
- A second study as a prerequisite for the v4 foundation.
- Any v3 refactor that changes its live result surface while v4 is being
  established.

### Archive workflow extraction

**Status:** Deferred. `pair_stability_v3` may retain a bounded `10_sync` stage
for one completed lineage: trace explicit provenance, account for bytes before
copying, omit checkpoints, and verify the immutable archive. Do not generalize
it now. If a second study needs archival transfer, first extract a
source-independent archive-plan/verification contract with explicit lineage and
size limits; keep Slurm submission as a thin study-facing adapter.

## Running decision log

| Date | Decision | Reason / consequence |
| --- | --- | --- |
| 2026-07-11 | Make this file the canonical running roadmap. | Issue #113 retains the manifesto and review discussion; issue #114 remains supporting historical context. |
| 2026-07-11 | Treat v3, not v2, as the v4 behavioral reference. | V2/v3 end-to-end parity was confirmed before intentional v3 profiling-schema additions; the retained v2 comparator has no valid current target. |
| 2026-07-11 | Keep v4 isolated from live v3 results. | All v4 writes require a separate root and frozen-reference comparison. |
| 2026-07-11 | Approve roadmap; keep implementation paused. | The first candidate when explicitly resumed is V4-1; V4-0 remains mandatory before v4 scientific-output work. |

## Review queue

Review this document before implementation against these questions:

1. Does the current-surface map accurately distinguish landed v3/toolkit
   behavior from absent v4 stack components?
2. Are the manifesto-derived non-negotiables sufficient to constrain the first
   v4 design review?
3. Is frozen-v3 → fresh-v4 the correct parity authority and comparison scope?
4. Is V4-1 the correct first additive code slice, with V4-0 required before
   any v4 scientific artifacts?
5. Are there any deliberate v4 layout changes that should be recorded now
   rather than discovered during implementation?

Do not start v4 implementation until review resolves these questions. On every
accepted decision or verified slice, update this file first, then add the
corresponding issue #113 review link/evidence reference.
