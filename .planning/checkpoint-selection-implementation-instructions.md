# V4 Checkpoint-Candidate Contracts: Future Implementer Instructions

> **Status:** implementation handoff, 2026-07-24. No implementation is
> authorized by this document.
>
> **Scope:** serializable v4 checkpoint-candidate contracts, strict
> experiment-side checkpoint observation, pure retry/selection/objective
> policy, and retention decisions. Dynamic scheduling and physical custody are
> later slices.
>
> **Authority:** this document operationalizes
> [`checkpoint-selection-boundary-decision.md`](checkpoint-selection-boundary-decision.md)
> and
> [`checkpoint-artifact-systems-decision.md`](checkpoint-artifact-systems-decision.md).
> [`toolkit-roadmap.md`](../experiments/toolkit-roadmap.md) remains the
> canonical v4 sequence and acceptance ledger. If repository reality changes,
> update the decisions and roadmap before changing these instructions.

## Intended outcome

Build an additive, experiment-owned contract layer that can answer all of the
following without consulting mutable pointers or result-arrival order:

1. Which optimizer batch and trial/seed producers were planned?
2. Which checkpoint steps were expected from each producer?
3. Which exact completed checkpoint bytes filled each expected slot?
4. Which slots were terminally absent, and why?
5. Which evaluation attempts occurred for each candidate?
6. How were retry and terminal-result decisions made?
7. What complete immutable population did the selector see?
8. Which checkpoint or checkpoint set was selected for each trial?
9. Which one objective vector or explicit failed outcome is returned for each
   trial proposed by `SearchStrategy.ask`?
10. Which artifacts should be retained or are release-eligible?

The first implementation is deliberately not a live checkpoint scheduler. It
may materialize these records after fixed-grid stages complete so the real v4
smoke route exercises the contracts. Later runtime work may consume the same
records while training and evaluation overlap.

## Current stop condition

As of 2026-07-24, feature code must not start. Three prerequisites are absent:

1. **V4-0:** frozen v3 non-pilot smoke reference, reviewed layout map,
   comparator, and a real isolated ten-stage v4 smoke route.
2. **Foundational V4-1 contracts:** one authoritative definition of trial,
   seed, run, producer/task, metric, and execution-profile identity.
3. **First-study policy configuration:** representable checkpoint cadence,
   seed grouping, objective metrics/aggregation, coverage rule, retry budgets,
   equivalent profiles, and v4 artifact-root identifiers.

Do not work around these gaps by creating checkpoint-local copies of
`TrialSpec`, `SeedAssignment`, metric schemas, or execution profiles. Complete
the prerequisites in separate reviewed slices. The roadmap forbids merging
unexercised experiment abstractions; unit tests alone are not an acceptance
route.

## Read these surfaces before editing

- [`experiments/README.md`](../experiments/README.md): experiment code cannot
  import `spenn`, except through the sanctioned launcher.
- [`toolkit-roadmap.md`](../experiments/toolkit-roadmap.md): ownership,
  immutable-stage rules, V4-0/V4-1 sequencing, and frozen-v3 parity gate.
- [`checkpoint-selection-boundary-decision.md`](checkpoint-selection-boundary-decision.md):
  accepted Level-0 experiment adapter and corrected contract model.
- [`checkpoint-artifact-systems-decision.md`](checkpoint-artifact-systems-decision.md):
  custody, receipt, and durable-storage boundary.
- [`spenn/checkpoint/manifest.py`](../spenn/checkpoint/manifest.py),
  [`save.py`](../spenn/checkpoint/save.py), and
  [`callback/checkpoint.py`](../spenn/callback/checkpoint.py): foreign wire
  format and currently representable callback schedule.
- [`experiments/toolkit/specs.py`](../experiments/toolkit/specs.py),
  [`jsonio.py`](../experiments/toolkit/jsonio.py),
  [`task_state.py`](../experiments/toolkit/task_state.py), and
  [`selection.py`](../experiments/toolkit/selection.py): established
  precedents and protected semantics that must not be silently reused.

Treat current code according to this hierarchy:

| Surface | Status for this feature |
| --- | --- |
| Planning/execution/run ownership in roadmap and issue #113 | Non-negotiable architecture |
| `StagePlan`, `TaskSpec`, `ResourceSpec`, and `ExecutionRecord` | Established generic foundations |
| Core `step_*` directory, manifest v1, and `COMPLETE` | Established foreign wire artifact |
| `task_state.py` checkpoint predicates | Established v3 behavior; wrong abstraction for candidate observation |
| Existing toolkit champion `selection.py` | Protected v3 table-selection behavior; not a checkpoint selector |
| `latest.json` | Mutable convenience pointer; never checkpoint identity |
| Old core-publisher and mutable-state proposals in the options discussion | Superseded by 2026-07-24 decisions |

## Non-negotiable boundaries

### Core run layer

Core continues to own:

- writing checkpoint payload files;
- manifest schema and `COMPLETE`;
- atomic publication of an exact `step_*` directory;
- restore/load semantics; and
- current callback cadence and `latest.json`.

Do not change `spenn/checkpoint/*` or add core batch, candidate, selection,
retention, or optimizer concepts in this slice.

### Experiment planning

Experiment planning owns:

- pre-`ask` capacity admission;
- exact batch, trial, seed, producer, and scheduled-step population;
- evaluator, retry, objective, and selection policy specifications;
- candidate cap and storage estimate; and
- immutable expected-slot identity.

An executor cannot shrink this population after `ask`.

### Experiment execution and collection

Experiment execution/collection owns:

- resolving authorized training attempts;
- observing and verifying exact completed step directories;
- deriving evaluation tasks later;
- recording attempts and failure evidence;
- retry decisions and terminal result resolution;
- producer and evaluation closure snapshots; and
- retention execution in a separate future custody slice.

### Pure policy

`CheckpointSelector`, `ObjectiveReducer`, and `RetryResolver` consume complete
serialized inputs. They do not read filesystems, resolve `latest.json`, submit
jobs, mutate records, or depend on arrival time.

## Terms

| Term | Meaning |
| --- | --- |
| **Producer** | One planned trial/seed/run training lineage expected to publish scheduled checkpoint steps. |
| **Producer attempt** | One authorized execution or resume attempt belonging to a producer. It is execution lineage, not a new scientific trial. |
| **Checkpoint slot** | One expected `(batch, producer, scheduled step)` position. |
| **Artifact reference** | Payload-fixed description of one exact completed core checkpoint directory. |
| **Candidate** | Binding from one checkpoint slot to one artifact reference from the authorized producer lineage. |
| **Absence** | Terminal evidence that one expected slot has no usable candidate. |
| **Candidate-set snapshot** | Immutable exact census of candidates and absences after every producer is terminal. |
| **Evaluation attempt result** | Observed success metrics or structured failure evidence for one candidate/spec/profile attempt. |
| **Retry decision** | Pure policy result choosing another authorized attempt or declaring terminal failure. |
| **Resolved evaluation** | Exactly one terminal successful result or failed result for a candidate. |
| **Batch-closure snapshot** | Immutable selector input containing the candidate census and exactly one resolved evaluation per candidate. |
| **Selection snapshot** | Replayable policy output naming selected candidates, computed vectors, exclusions, and deterministic tie trace. |
| **Trial objective outcome** | Exactly one objective vector or explicit failed outcome for one asked trial. |
| **Retention decision** | Scientific intent to retain or mark release-eligible. It is not proof that bytes were copied or deleted. |
| **Archive receipt** | Future custody evidence proving verified durable bytes. It is not implemented in this slice. |

## Required planning and artifact flow

```text
versioned search space, seed policy, and realizable checkpoint schedule
  → pure capacity preflight
  → BatchCapacityAdmissionV1
  → SearchStrategy.ask(admitted_trial_count) exactly once
  → validate returned proposals against admission envelope
  → atomically publish BatchPlanV1
  → resolve terminal producer attempt lineage
  → observe exact checkpoint artifacts
  → bind candidate or absence for every expected slot
  → CandidateSetSnapshotV1
  → evaluation attempts and structured failure evidence
  → pure retry decisions and terminal evaluation resolutions
  → BatchClosureSnapshotV1
  → pure CheckpointSelector
  → CheckpointSelectionSnapshotV1
  → pure ObjectiveReducer
  → one TrialObjectiveOutcomeV1 per asked trial
  → RetentionDecisionV1
  → future custody request, archive receipt, and receipt binding
  → only then may later GC consider source pruning
```

No work may launch between `ask` and durable `BatchPlanV1` publication. If
publication fails, optimizer state and the unrecorded proposals require
explicit recovery; never call `ask` again and pretend the first call did not
happen.

## Capacity admission must precede `ask`

`SearchStrategy.ask` opens a scientific batch. Capacity checks therefore run
before it, using a conservative proposal-independent envelope:

```text
candidates_per_trial = seed_count × scheduled_step_count
admitted_trial_count = min(
    requested_trial_count,
    candidate_cap // candidates_per_trial,
    usable_storage_bytes // bounded_bytes_per_trial,
)
```

`usable_storage_bytes` must already subtract protected data and a configured
safety margin. If no defensible checkpoint-size upper bound exists, or
`admitted_trial_count == 0`, reject before `ask`. Do not ask for a larger
population and discard proposals afterward.

`BatchCapacityAdmissionV1` should record:

```text
kind and schema version
admission_id
search-space, seed-policy, and checkpoint-schedule IDs
requested and admitted trial counts
seed and scheduled-step counts
candidate cap
checkpoint-size estimator identity and version
bounded bytes per candidate/trial
observed capacity, already-protected bytes, and safety margin
worst-case candidate count and storage demand
decision and explanatory trace
observation time outside semantic identity
```

This is deterministic planning evidence, not a reservation, lease, monitor,
queue, or controller. `BatchPlanV1` references the accepted admission and
records exact proposals and exact derived slots. Proposal shape/count must fit
the envelope. A later observed size overrun never permits candidate deletion;
record the operational failure, protect all opened-batch candidates, and block
new admission until resolved.

## Checkpoint schedule must match existing core behavior

Core currently supports periodic `every_n_steps`/`start_step` filtering plus a
separate `train_end` checkpoint callback. It does not support an arbitrary
step list. V1 planning must therefore persist:

```text
max_steps
every_n_steps
start_step
include_train_end
explicit expected_steps after expansion and terminal-step de-duplication
```

For example, a cadence of 1,000 updates and `max_steps=3,500` yields expected
steps `(1_000, 2_000, 3_000, 3_500)`. Validate all of the following:

- positive cadence and nonnegative, strictly increasing expected steps;
- `probability == 1`;
- no `max_calls` truncation;
- deterministic triggers only;
- the periodic and terminal callback configuration expands to the persisted
  expected steps;
- candidate mode uses `keep_last=None`; and
- the sum of expected slots across all producers is less than or equal to the
  admitted candidate cap. Unused admitted capacity is valid.

Unexpected published steps are conflicts, not bonus candidates. Missing steps
become absences only after terminal producer evidence. Irregular schedules
require a later generic core quality-of-life enhancement and a new reviewed
schedule contract; do not emulate them through polling or ad hoc callback
lists.

## Training-attempt resolution

Never select a training attempt through directory ordering, modification time,
arrival time, or mutable `latest.json`.

At minimum, add `ProducerAttemptResolutionV1` before candidate binding. It
records:

```text
kind and schema version
resolution_id
batch and producer IDs
trial, seed, run, and train-task IDs
ordered authorized/considered attempt IDs
attempt-selection policy ID and digest
terminal evidence references
selected attempt lineage or terminal no-usable-attempt outcome
deterministic reason and tie trace
resolution time outside semantic identity
```

One material detail depends on foundational V4-1 retry semantics and must be
resolved there:

- If every retry reuses one logical attempt directory/identity, one
  producer-level selected attempt is sufficient.
- If resumes create an ordered chain of attempt IDs and different attempts can
  legitimately hold different scheduled steps, the producer resolution must
  select an authorized resume lineage and a separate
  `CheckpointSlotResolutionV1` must deterministically select zero or one
  artifact for each slot from that lineage.
- If an attempt restarts training as a new stochastic trajectory, do not mix
  its steps with another attempt. Resolve one complete scientific lineage or
  represent it as a different producer/trial.

Do not implement until V4-1 states which model applies. In every model,
duplicate identical observations are idempotent; divergent artifacts for the
same resolved slot are conflicts; late attempts after closure require an
explicit corrective record or new snapshot, never in-place mutation.

## Recommended package ownership

After prerequisites exist, create a focused subpackage rather than enlarging
`specs.py` or legacy `selection.py`:

```text
experiments/toolkit/checkpoint_candidates/
  __init__.py
  _codec.py
  io.py
  artifacts.py
  batches.py
  evaluations.py
  policies.py
  retention.py
```

| Module | Owner |
| --- | --- |
| `_codec.py` | Type-specific version dispatcher, strict field sets, canonical semantic projections, deterministic IDs |
| `io.py` | Immutable JSON/JSONL bundle reading, validation, file digests, destination-local temporary publication |
| `artifacts.py` | Foreign manifest-v1 parser, path validation, streaming payload hashing, observation and verification |
| `batches.py` | Capacity admission, realizable schedules, producers/slots, attempt resolution, candidates/absences, producer closure |
| `evaluations.py` | Evaluator specs, attempts/results, failure evidence, retry decisions, terminal resolution, evaluation closure |
| `policies.py` | Objective and selection specs, pure protocols, slow reference implementations, decision traces |
| `retention.py` | Retention decisions only; no filesystem or archive backend |

Keep package out of top-level `experiments.toolkit` exports until its public
surface is exercised and reviewed. Production modules under `experiments/`
must not import `spenn`.

Tests should live beside this package under `experiments/toolkit`, following
the existing experiment-test convention. Use a tiny checked-in core-manifest-v1
wire fixture plus a cross-boundary integration test that creates a real
checkpoint through the sanctioned run entrypoint and observes it through the
experiment adapter.

## Common contract rules

### Type-specific versioning

Every record carries `kind` and its own schema version, for example:

```text
spenn.experiments.checkpoint-artifact-ref/v1
spenn.experiments.checkpoint-batch-plan/v1
spenn.experiments.candidate-set-snapshot/v1
```

Implementation classes should be suffixed `V1`. Public reader functions
dispatch by exact `(kind, schema_version)` and reject unknown combinations.
Never bump or reinterpret global `experiment-toolkit/v1`.

V1 field meaning never changes. A future incompatible schema adds a V2 reader
and writer while V1 remains readable. Reject unexpected top-level fields;
future optional data belongs in an explicit `extensions` mapping with defined
identity treatment.

### Logical immutability

Use frozen dataclasses and typed frozen nested records. Prefer tuples over
lists and arbitrary mappings. `frozen=True` alone is insufficient if fields
contain mutable dictionaries; copy and normalize allowed configuration data at
construction, or represent identity-bearing data as typed tuples.

No record contains mutable `status`, `pin_state`, or `retention_state`.
Later facts reference earlier IDs. A correction is a new record with explicit
supersession/conflict lineage.

### Deterministic identities

Document each record's semantic identity projection beside its class.
Timestamps, absolute paths, storage locators, hostnames, free-form diagnostics,
and arrival order stay outside identity.

For identity projections:

- use UTF-8, sorted object keys, compact separators, and `allow_nan=False`;
- reject non-string mapping keys and unsupported values;
- avoid floats where possible; use integers or validated finite decimal text
  for identity-bearing numeric configuration;
- prefix or length-delimit components before SHA-256 hashing; and
- include policy/configuration digests wherever changing policy changes
  meaning.

Do not change or reuse current `jsonio.py` for semantic identities. Its
behavior is part of existing toolkit/v3 compatibility.

### Immutable publication

One stage/finalizer writes each complete JSON or JSONL artifact once. Workers
must not append concurrently to a shared JSONL file on NFS.

Writer sequence:

1. normalize and validate all records;
2. sort by the contract's canonical order;
3. write to a unique destination-local temporary file;
4. flush and close;
5. compute row count and file SHA-256;
6. atomically replace the final path on the same filesystem; and
7. publish the snapshot manifest last.

Readers validate kind/version, exact field sets, row count, ordered IDs, and
file digest. A partially written bundle is absent or invalid, never a smaller
valid population.

## Artifact adapter contract

### Records

`PayloadFileV1`:

```text
manifest role
normalized checkpoint-relative path
size in bytes
SHA-256
```

`CheckpointArtifactRefV1`:

```text
kind and schema version
artifact_id
core checkpoint kind and schema version
step
raw manifest size and SHA-256
canonical sorted payload file table
payload-table SHA-256
trusted storage-root ID
root-relative checkpoint locator
observation time outside identity
```

Artifact identity includes core kind/schema, step, raw manifest digest, and
canonical payload-table digest. It excludes storage root and locator. Moving
identical bytes between trusted roots changes custody/location, not artifact
identity.

### Public operations

Target narrow operations:

```text
observe_checkpoint_v1(exact_step_dir, trusted_root, trusted_root_id)
  → CheckpointArtifactRefV1

verify_checkpoint_artifact(ref, trusted_root_mapping)
  → verification success or explicit error
```

Do not accept a checkpoint root, `latest.json`, URI, or arbitrary mapping in
the V1 observer.

### Observation algorithm

Implement a slow, readable, streaming reference:

1. Require exact `step_<digits>` directory, not `.tmp`.
2. Require `manifest.json` and `COMPLETE` to be regular files.
3. Parse JSON without importing `spenn`.
4. Require exact core kind `spenn.checkpoint`, schema version `1`, integral
   nonnegative step, and step/directory agreement. Reject coercive values such
   as numeric strings or booleans.
5. Validate exact manifest-v1 fields and `files` mapping shape.
6. For every declared payload, require one normalized relative path contained
   below the step directory. Reject absolute paths, empty components, `.`,
   `..`, symlinks, duplicate targets, missing files, and non-regular files.
7. In V1, reject unmanifested entries other than `manifest.json` and
   `COMPLETE`. This makes the verified payload set exact; future additive core
   files require a new adapter version or explicit ignore rule.
8. Open each file without following symlinks, stream it through a bounded
   buffer, and record size/SHA-256. Never deserialize `.pt` payloads.
9. Compare file metadata before and after hashing; fail if size, inode, or
   modification evidence changes. Recheck manifest and `COMPLETE` before
   publication.
10. Sort file rows by normalized relative path, compute their semantic digest,
    compute raw manifest digest separately, then compute artifact ID.
11. Verify locator containment under the trusted root and store only root ID
    plus root-relative path.

Payload or manifest mutation after observation makes verification fail.
Changing `latest.json` has no effect. Hash using constant memory and persist
the digest once; later candidate and policy operations must not rehash payloads.

## Batch and producer-closure contracts

Use foundational V4-1 identities rather than redefining them.

Minimum records:

- `CheckpointScheduleV1`
- `PlannedCheckpointProducerV1`
- `BatchCapacityAdmissionV1`
- `CheckpointBatchPlanV1`
- `ProducerAttemptResolutionV1`
- optional `CheckpointSlotResolutionV1`, depending on retry model
- `CheckpointCandidateV1`
- `CheckpointAbsenceV1`
- `CandidateSetSnapshotV1`

`CheckpointBatchPlanV1` records:

```text
batch and strategy-snapshot IDs
batch index
capacity-admission ID
exact ordered trial proposals
seed/run/producer memberships
exact ordered expected slots
evaluation, retry, objective, and selection policy IDs
allowed execution-profile references
candidate cap and estimated byte demand
```

Candidate ID binds exactly one slot, artifact ID, and producer/slot resolution.
Absence ID binds exactly one slot to structured terminal producer evidence.
Initial absence reasons should distinguish:

- producer failed;
- producer stopped;
- scheduled checkpoint missing after terminal success; and
- checkpoint integrity failure.

Producer closure requires exact set equality:

```text
expected slots == candidate slots ∪ absence slots
candidate slots ∩ absence slots == ∅
```

Reject duplicate, foreign, unexpected, or late slot bindings. Snapshot identity
is independent of input row order. Snapshot existence means closure; do not add
an `open|closing|closed` status field.

## Evaluation, failure, and retry contracts

Keep observation separate from policy.

### Evaluation specification

`CheckpointEvaluationSpecV1` fixes:

- evaluator policy key and version;
- evaluator configuration digest;
- input data references and digests;
- required metric schema, units, and numeric contract;
- accepted numerical execution-profile equivalence class; and
- expected artifact outputs.

Scientific input, evaluator code/configuration, data, metric meaning, or
numerical-contract changes create a new evaluation spec, not a retry.

### Attempts and observed failure evidence

`CheckpointEvaluationAttemptV1` binds candidate, evaluation spec, attempt
ordinal, execution-profile digest, and later task/execution-record IDs.

`CheckpointEvaluationAttemptResultV1` is exactly one of:

- success with all required finite metrics; or
- `EvaluationFailureEvidenceV1`.

Failure evidence records a stable diagnostic code and evidence, such as
preemption, timeout, out-of-memory, infrastructure/environment failure,
checkpoint integrity failure, evaluator failure, or invalid/missing metrics.
Workers report evidence; they do not decide whether another attempt is
scientifically valid.

### Retry policy and terminal resolution

`RetryPolicySpecV1` defines:

- same-profile and alternate-profile attempt budgets;
- exact profile digests considered scientifically equivalent;
- diagnostic-code-to-action rules; and
- deterministic exhaustion behavior.

Pure `RetryResolver` emits `RetryDecisionV1`:

```text
retry same profile
retry one explicitly equivalent alternate profile
terminate with failed evaluation
```

Profile labels alone are insufficient because their contents may change.
Retry decisions reference complete profile/spec digests and considered attempt
IDs.

`ResolvedCheckpointEvaluationV1` records exactly one terminal leaf:

- one explicitly selected successful attempt result; or
- explicit terminal failure after policy resolution/exhaustion.

It includes every considered attempt/result/decision ID and deterministic
reason trace. Never choose “latest.” Duplicate identical results are
idempotent; divergent duplicates, multiple unauthorized successes, or
conflicting terminal leaves fail closure.

`BatchClosureSnapshotV1` requires exactly one resolved evaluation for every
real candidate in `CandidateSetSnapshotV1`. Absences do not receive fabricated
evaluations; they remain inputs to coverage and failed-trial policy.
`pruned_before_evaluation` is forbidden because candidate payloads must survive
through selection.

## Selection and objective contracts

Do not alias checkpoint selection, objective reduction, and optimizer strategy.

```text
CheckpointSelector:
  complete batch closure + policy specs
  → CheckpointSelectionSnapshotV1

ObjectiveReducer:
  complete batch closure + selection + objective spec
  → one TrialObjectiveOutcomeV1 per asked trial
```

Both protocols are pure and operate only on typed records.

`ObjectiveComponentV1` should record metric name, unit, minimize/maximize
direction, aggregation, optional scale, and constraint role.

`ObjectiveSpecV1` should record:

- ordered components;
- seed aggregation and minimum successful coverage;
- constraints;
- optional explicit scalarization;
- comparison tolerance;
- failed-trial policy; and
- deterministic projection from a multi-checkpoint selection to one optimizer
  vector per trial.

`CheckpointSelectionPolicySpecV1` should record:

- grouping: individual candidate or aligned trial-step seed group;
- output: one, top-K, or Pareto;
- `K` where applicable;
- total deterministic tie-break order; and
- Pareto/frontier anchor or collapse rule needed for one `tell` per trial.

Energy and local-energy variance belong in study configuration as the default
objective, not in toolkit schema defaults.

Provide slow reference implementations before optimization:

- configured single winner;
- deterministic top-K;
- deterministic Pareto frontier;
- weighted scalarization when explicitly configured;
- aligned-step seed aggregation;
- minimum-coverage and constraint filtering; and
- explicit failed-trial outcomes.

Input permutation must not change selected IDs, objective results, or decision
trace. Tie-breaking ends with a stable total order over typed candidate
identity, never row position or arrival time. Missing/non-finite metrics and
failed trials remain structured failures; never substitute `±infinity`.

`CheckpointSelectionSnapshotV1` records all considered candidate/evaluation
IDs, policy/objective IDs, computed vectors, exclusions, selected IDs grouped
by trial, and deterministic tie trace.

`TrialObjectiveOutcomeV1` records exactly one success vector or explicit failed
outcome for every trial returned by `ask`. Assert cardinality equality against
`CheckpointBatchPlanV1`.

## Retention boundary

This slice implements only `RetentionDecisionV1`:

```text
selection snapshot ID
candidate and artifact IDs
intent = retain | release_eligible
policy ID
reason
```

`release_eligible` is not deletion permission. Before physical custody exists:

- every candidate source remains protected through selection;
- selected sources remain protected indefinitely;
- `keep_last=None` remains required; and
- no code removes source or archive bytes.

Defer `RetentionRequestV1`, `ArtifactArchive`, `ArchiveReceiptV1`, receipt
binding, verified-copy/restic backends, and all pruning to one separately
reviewed custody/finalizer slice. That slice must implement and exercise the
receipt contract against real bytes. Expected receipt fields remain normative
in
[`checkpoint-artifact-systems-decision.md`](checkpoint-artifact-systems-decision.md).
No archive interface exposes `delete` until independent GC design exists.

## V4-only serialized layout

Final paths belong to reviewed v4 stage ownership and layout mapping. A
reasonable logical bundle is:

```text
checkpoint_batches/<batch_id>/
  capacity_admission.json
  batch_plan.json
  producer_attempt_resolutions.jsonl
  checkpoint_artifacts.jsonl
  candidates.jsonl
  absences.jsonl
  candidate_set_snapshot.json
  evaluation_specs.jsonl
  evaluation_attempts.jsonl
  evaluation_results.jsonl
  retry_decisions.jsonl
  resolved_evaluations.jsonl
  batch_closure_snapshot.json
  checkpoint_selection_snapshot.json
  trial_objectives.jsonl
  retention_decisions.jsonl
```

These are immutable stage outputs, not a shared event log. The concrete v4
layout may distribute them across independently runnable stages. Each
downstream snapshot must reference explicit upstream attempt/artifact IDs;
never search for a newest directory. Declare every file as a reviewed v4-only
sidecar in `layout_map.json`; it may not replace a mapped v3 artifact.

## Implementation sequence

Prefer small PRs, but never merge a dormant abstraction. Every accepted code
slice must be exercised by the real v4 smoke route.

### Prerequisite PRs

1. **V4-0 reference and executable bootstrap**
   - freeze one matching-profile v3 non-pilot smoke lineage;
   - implement reviewed comparator/layout map;
   - run a real isolated ten-stage v4 smoke route.
2. **Foundational V4-1 contracts**
   - land canonical trial/seed/run/metric/profile identities;
   - define producer-attempt/resume semantics;
   - exercise them in v4 smoke.

### Feature PR 1 — strict codec and artifact adapter

- Add type-specific dispatcher and immutable I/O.
- Add manifest-v1 golden fixture and exact path validation.
- Add streaming observation/verification and deterministic artifact identity.
- Make v4 smoke observe actual completed `step_*` directories and write
  additive artifact-reference sidecars.

### Feature PR 2 — admission, batch, attempt, and candidate census

- Add realizable schedule expansion and capacity admission before `ask`.
- Add exact batch/producers/slots and attempt resolution.
- Add candidate/absence binding and producer closure.
- Make v4 smoke emit a complete post-training census.

### Feature PR 3 — evaluation and terminal closure

- Add evaluator specs, attempts/results, failure evidence, retry policies and
  decisions, terminal resolution, and batch closure.
- Use fixed existing v4 evaluation results; do not add dynamic materialization.
- Exercise successful and failed terminal paths in focused tests and real
  successful path in v4 smoke.

### Feature PR 4 — selection, objectives, and retention decisions

- Add configurable pure policies and slow reference implementations.
- Add deterministic selection snapshot and one trial objective outcome per
  asked trial.
- Add retention decisions with no physical effect.
- Exercise complete contract path in v4 smoke.

### Later custody/finalizer PR

- Select durable storage and operator.
- Add retention requests, `ArtifactArchive`, verified archive receipts, crash
  reconciliation, materialization, and restoration tests.
- Keep GC and deletion separate even then.

If one feature PR cannot be exercised without the next, combine them into the
smallest executable review slice instead of weakening the E2E gate.

## Required focused tests

### Codec and serialization

- Round-trip every V1 record.
- Reject unknown kind/version, missing and extra fields, scalar-for-sequence
  coercion, booleans as integers, NaN/Inf, unsupported mappings, and invalid
  UTF-8/path values.
- Prove deterministic IDs under mapping/input-order permutations.
- Prove locator/timestamp changes do not alter semantic identity.
- Tamper with JSONL row count, ordered IDs, and file digest.
- Simulate interruption before and after final-path publication.

### Artifact adapter

- Core-produced manifest-v1 fixture and real configured-run checkpoint.
- Fresh-process import firewall proving experiment package does not load
  `spenn`.
- Missing/non-regular `COMPLETE` or manifest; wrong kind/schema/step.
- Exact step-directory mismatch and rejection of root/`latest.json` input.
- Missing, duplicate, absolute, traversing, symlinked, non-regular, and
  unmanifested payload paths.
- Concurrent mutation/deletion during hashing.
- Large-file streaming with bounded memory.
- Moving `latest.json` cannot alter captured identity.
- Relocation under another trusted root preserves artifact ID.
- Payload or manifest mutation fails verification.

### Schedule, admission, and producer closure

- Cadence landing and not landing on terminal `max_steps`.
- Invalid, duplicate, unsorted, impossible, probabilistic, or truncated
  schedules.
- Candidate cap and storage admission, including zero admission.
- Prove `ask` is not called on rejected admission and called exactly once on
  accepted admission.
- Proposal count/shape outside envelope fails before work materialization.
- Multiple producer attempts, resume lineage, divergent duplicate slot
  artifacts, and deterministic resolution.
- Exact candidate/absence set equality; duplicate, missing, foreign,
  unexpected, and late slots.
- Producer failure/stopped/integrity absence reasons.
- Input permutations produce identical snapshot ID and row order.

### Evaluation and retry

- Deterministic attempt identity and ordinal/profile validation.
- Success requires every declared metric, correct unit/schema, and finite
  value.
- Failure evidence for preemption, timeout, OOM, environment, integrity,
  evaluator, and invalid-metric cases.
- Same-profile and explicitly equivalent-profile budgets.
- Profile-content change invalidates equivalence.
- Scientific-input change creates a new evaluation spec.
- Exhausted retry produces terminal failed resolution.
- Duplicate identical result is idempotent; divergent duplicate conflicts.
- Missing, duplicate, pending, or conflicting resolutions prevent closure.
- Result-arrival permutations produce identical terminal closure.

### Selection and objective reduction

- Configurable minimize, maximize, and mixed vectors.
- Seed aggregation, aligned-step grouping, minimum coverage, and constraints.
- Explicit scalarization, single winner, top-K, and Pareto frontier.
- Deterministic ties and Pareto/frontier anchor for optimizer feedback.
- Missing/non-finite metrics and producer/evaluation failures.
- Reordered input produces byte-equivalent IDs and traces.
- Every asked trial yields exactly one successful or failed objective outcome.
- Failed science never becomes an infinite numeric objective.

### Retention and architecture

- Every selected artifact gets retain intent.
- Only terminal non-selected artifacts become release-eligible.
- No receipt means no pruning authority.
- Static and fresh-process checks enforce no `spenn` import.
- Existing toolkit, core checkpoint, and v3 experiment tests remain unchanged
  and green.

Use repository tooling:

```text
uv run pytest <focused experiment tests>
uv run pytest experiments/toolkit
uv run pytest tests/unit/training/test_checkpoint_callback.py
uv run pytest tests/unit/callback/test_checkpoint.py
uv run pytest experiments/hooke/pair_stability_v3
```

If environment setup fails, stop and resolve it interactively per project
rules. Do not install substitute packages or use `uv run --nosync`.

## Mandatory E2E evidence

Every accepted experiment-stack code slice requires:

1. matching frozen v3 non-pilot `smoke.yaml` reference;
2. fresh real v4 non-pilot smoke route;
3. explicit `--blind --blind-seed 811` and attempt IDs;
4. matching `test` or `gpu_test` execution profile;
5. exact protected mapped v3/v4 artifacts under reviewed substitutions;
6. separately validated v4-only checkpoint sidecars; and
7. retained command, profile, task-count, artifact-count, and comparator
   evidence.

Do not rerun v3 after its reference is frozen. Local-only evidence follows the
roadmap fallback and must later be backfilled on the required scheduler
profile. Full science uses production partitions, never test partitions.

## Compatibility and migration

- Do not modify live v3 scripts, output schemas, paths, pointers, or result
  data.
- Do not modify global `experiment-toolkit/v1`, `StagePlan`, `TaskSpec`,
  `task_state.py`, current `selection.py`, or current `jsonio.py` semantics.
- New artifacts are v4-only additive sidecars.
- No automatic backfill of historical checkpoints.
- Unknown future schemas fail closed; old V1 readers stay available.
- A moved identical payload keeps artifact identity but receives a new locator
  or future custody receipt.
- Never rewrite immutable records during migration; append versioned
  supersession/mapping records.
- No feature branch may write below a v3 results root.

## Performance and operational limits

- Stream SHA-256 with a bounded buffer; never load checkpoint files into
  memory or deserialize them for identity.
- Hash each observed payload once and reuse persisted digests.
- Start with one observer/finalizer writer on NFSv3.
- Bound policy work by persisted candidate cap.
- A readable `O(n² × objectives)` Pareto reference is acceptable for bounded
  batches; optimize only after profiling and equivalence tests.
- Benchmark representative checkpoint hashing, sidecar size, selection time,
  and E2E overhead without flaky wall-clock assertions.
- Tests log in UTC; experiment records use `America/New_York` where the study
  convention requires human-readable time.

## Explicit anti-goals

Do not add:

- core publication events, holds, pins, or experiment callbacks;
- `latest.json` identity or fallback;
- arbitrary checkpoint-container probing;
- polling, queues, cursors, leases, claims, worker pools, controller services,
  or dynamic task materialization;
- concurrent shared JSONL append;
- mutable record state;
- Optuna, Ray, ASHA, Hyperband, PBT, pruning, promotion, or early stopping;
- artifact database, MLflow, MLMD, restic, DataLad, DVC, or another dependency;
- physical archive, source deletion, garbage collection, or delete-capable
  archive API;
- hard-coded energy/variance schema, checkpoint cadence, seed grouping, retry
  budget, execution partition, candidate cap, or storage budget; or
- changes to v3 readiness/completion predicates.

## Decisions required when first study is configured

These are configuration choices, not reasons to reopen generic ownership:

1. checkpoint cadence representable by existing callback and whether terminal
   step is always included;
2. seed grouping: independent candidates or aligned trial-step groups;
3. ordered objective metric keys, units, directions, aggregation, constraints,
   and scalarization/frontier anchor;
4. minimum successful seed/evaluation coverage and failed-trial handling;
5. retry budgets, diagnostic mapping, and exact equivalent-profile digests;
6. conservative checkpoint-size estimator, candidate cap, capacity margin, and
   batch-size request;
7. v4 storage-root IDs and concrete sidecar stage layout; and
8. later durable archive destination, operator, key policy, and replica count.

Record every choice in versioned experiment configuration and resolved
`BatchPlanV1` provenance.

## Handoff checklist

Before requesting review, future implementer must be able to answer yes:

- [ ] V4-0 and foundational V4-1 prerequisites exist and are referenced.
- [ ] Roadmap and layout map describe every new sidecar.
- [ ] Core and v3 files are untouched.
- [ ] Experiment production code imports no `spenn`.
- [ ] Exact representable schedule and pre-`ask` admission are persisted.
- [ ] Training attempt/resume semantics are explicit.
- [ ] Every expected slot closes as one candidate or absence.
- [ ] Every candidate closes as one successful or failed evaluation.
- [ ] Arrival order and mutable pointers cannot affect any decision.
- [ ] Every asked trial yields exactly one objective outcome.
- [ ] Retention decisions claim no physical custody or deletion authority.
- [ ] Focused tests, protected existing suites, and full matching-profile v4
      smoke parity pass.
- [ ] Validation commands, artifacts, performance observations, and remaining
      policy choices are recorded for review.

Full planner, implementation-representative, and critic discussions remain
available through Paseo.
