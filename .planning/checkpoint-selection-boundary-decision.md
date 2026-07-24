# Checkpoint-Candidate Boundary: Human Decision Memo

**Status:** decision record, updated 2026-07-24; no implementation authorized.

**Scope:** future v4 experiment stack only. Live v3 behavior remains unchanged.

**Purpose:** record chosen ownership and policy boundaries, explain remaining
planning parameters, and define the first contract slice for
optimizer-initiated batch checkpoint selection.

## Decision frame

Current v3 plans training, waits for every run, plans validation, collects a
stable result population, then selects champions. This is reproducible but
cannot evaluate intermediate checkpoints while training continues.

Future v4 should support bounded optimizer batches. Training runs publish
scheduled checkpoint candidates; evaluations may overlap training; next
optimizer decision waits for a complete, replayable batch snapshot.

Six requirements are already fixed:

| ID | Requirement |
| --- | --- |
| D1 | Every successfully published checkpoint in the batch's fixed, persisted checkpoint schedule is a candidate. Unscheduled optimizer steps are not candidates. |
| D2 | Selection sees every published candidate with a terminal evaluation outcome. Result arrival order cannot change scientific selection. |
| D3 | `SearchStrategy.ask` opens a batch; `tell` happens only after checkpoint selection and objective reduction close that batch. |
| D4 | Every candidate payload survives through selection. Selected payloads receive a physical pin; only terminal non-selected payloads may later be pruned, with an appended outcome record. |
| D5 | Default objective vector minimizes energy and local-energy variance. One, top-K, Pareto, scalarization, constraints, aggregation, and tie-breaking remain configured policy. |
| D6 | Acceptance uses `test`/`gpu_test`; full science uses production CPU/GPU profiles and records the exact execution profile. |

Present scope is narrower than target runtime: define serializable references,
batch/candidate/evaluation/selection records, and pure policies. Defer polling,
queues, worker loops, controller services, databases, dynamic task
materialization, promotion, and early stopping. Full discussion history remains
in [`checkpoint-selection-options.md`](../experiments/checkpoint-selection-options.md),
but this memo does not require it.

## Recommendation

Adopt an **artifact-hybrid boundary, using an experiment-side adapter first**.
This supports target behavior without adding an unused core publication API
now.

1. Core `spenn` continues to own checkpoint creation, completeness, schema,
   restore/load semantics, and its existing atomic wire artifact:
   `step_*/manifest.json` plus `step_*/COMPLETE`.
2. `experiments/toolkit` snapshots an exact completed step directory into a
   versioned `CheckpointArtifactRef`. It never resolves `latest.json`.
3. Experiment contracts add batch, trial, seed, task, evaluation, selection,
   objective, and retention meaning around that foreign reference.
4. Candidate-producing v4 runs use a bounded checkpoint schedule and
   `keep_last=None` through batch selection. No candidate payload may be pruned
   before selection.
5. A later experiment execution component may copy or archive selected
   checkpoints into experiment-owned durable storage and emit a
   digest-verified `RetentionReceipt`. A JSONL row, symlink, or hardlink on the
   source filesystem is not independently durable custody.

Core publication must contain no batch, trial, seed, task, selection, or
retention meaning. Add a core hold protocol only when evaluation must overlap
core pruning or another independent consumer needs it.

## Why this fits the v4 restructure

Issue [#113](https://github.com/YizhongHu/SpENN/issues/113) and the
[v4 roadmap](../experiments/toolkit-roadmap.md) separate scientific planning,
operational execution, and one-run production. Durable artifacts connect them;
`experiments/` cannot import `spenn` except through the sanctioned launcher.

Core already assembles a checkpoint under a temporary directory with its
manifest and `COMPLETE`, then atomically renames it. It next updates mutable
`latest.json` and applies `keep_last`. Exact completed step paths are therefore
suitable facts; `latest.json` is not identity; finite `keep_last` cannot safely
race an external pin. Candidate mode must disable pruning until selection.
Current manifest hashes configuration metadata, not payload bytes.

External precedents agree with ownership direction: Lightning
[`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
owns run-local checkpointing; Optuna
[`ask`/`tell`](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)
keeps trial lifetime in the optimization layer.

## Boundary options

Here, **boundary** means the ownership and serialization seam between one
SpENN run and the experiment system. A **wire artifact** is the durable
filesystem representation exchanged across that seam; it does not imply a
network service. An **adapter** is experiment-side code that validates and
translates the wire artifact without importing or controlling the producer.

### Level 0 — Existing artifact plus experiment adapter

Core keeps writing its current exact checkpoint directory, manifest, and
`COMPLETE` marker. Experiment code validates those files, rejects mutable
`latest.json` as identity, and emits its own versioned
`CheckpointArtifactRef`.

This is recommended now because it introduces no second producer format and no
core migration. Cost: experiment code must explicitly support each accepted
core manifest version. Candidate-producing runs must also disable automatic
pruning until selection finishes, because validation after publication cannot
prevent an earlier core deletion.

### Level 1 — Narrow core reference plus hold protocol

Core would publish a generic stable reference, possibly through a sidecar or
run-local index. A **hold protocol** would additionally let a consumer acquire
a deletion barrier before core pruning and later release it. The acknowledgement
must participate in save/prune ordering; a reference file alone does not keep
payload bytes alive.

This level becomes useful when evaluation overlaps finite retention, or when
multiple non-experiment consumers need the same publication contract. It adds
core schema, recovery, compatibility, and crash-reconciliation work, so current
contract-only scope does not justify it.

### Level 2 — Neutral shared contract package

A third package would own checkpoint-reference schemas and parsers. Both core
and experiment code would depend on it, avoiding duplicate validation logic.
This creates a new architectural owner and couples two otherwise separated
repositories through Python dependencies and releases. Consider it only after
several independent producers or consumers demonstrate repeated schema logic.

### Level 3 — Rich core lifecycle/controller

Core would own batch membership, trial/seed joins, evaluation state, selection,
optimizer feedback, or retention policy. This may look convenient because one
system sees everything, but it makes scientific-policy changes require core
checkpoint changes. It also collapses run, planning, and execution ownership.
Reject this option. Queues, polling, leases, and workflow services belong to
future experiment execution, not core checkpoint semantics.

## Corrected contract model

The candidate schema under consideration has two ownership leaks:
`CheckpointPublished` contains
experiment fields, and immutable records contain mutable `status`,
`retention_state`, or `pin_state`. Replace those with immutable facts and
append-only decisions:

A **contract** below is a versioned serialized record, not necessarily a
Python class or database table. **Immutable** means a record is never edited
after publication; later knowledge creates another record referencing it.
A **snapshot** is an immutable census of a bounded population at one declared
closure point. **Terminal** means no further automatic work or retry can change
that item's outcome.

```text
BatchCapacityAdmission
  → SearchStrategy.ask
  → BatchPlan
  → ProducerAttemptResolution
  → completed checkpoint artifact
  → CheckpointArtifactRef
  → CheckpointCandidate
  → CandidateSetSnapshot
  → resolved evaluation results
  → BatchClosureSnapshot
  → CheckpointSelector
  → ObjectiveReducer
  → selection and retention records
  → SearchStrategy.tell
```

| Contract | Owner | Meaning |
| --- | --- | --- |
| `BatchCapacityAdmission` | experiment planning | Proves a conservative trial/seed/schedule population fits candidate and storage bounds before `SearchStrategy.ask`. It is planning evidence, not a storage reservation or controller. |
| `BatchPlan` | experiment planning | Freezes what optimizer asked for: ordered trials, seed/run membership, expected checkpoint steps, evaluator and policy versions, and execution profile. It is intention, not progress state. |
| `ProducerAttemptResolution` | experiment collection | Selects one authorized terminal training attempt or resume lineage for each planned producer by deterministic policy. It prevents “newest attempt” from deciding which checkpoint artifacts may fill slots. |
| `CheckpointArtifactRef` | experiment adapter | Describes one exact completed core artifact. Stable identity uses schema and digest fields; a root-relative locator says where bytes currently live. Moving an identical retained copy changes locator, not identity. |
| `CheckpointCandidate` | experiment planning | Binds one expected `(batch, trial, seed/run, step)` slot to one artifact reference. This binding gives a core checkpoint experiment meaning. |
| `CandidateSetSnapshot` | experiment collection | Freezes producer census after every training producer terminates. Each expected slot is either a candidate or an explicit terminal absence. Any later publication conflicts with closure. |
| Evaluation spec, attempt, and result | experiment planning/execution | Evaluation spec defines intended measurement. Every attempt records success or structured failure evidence. Retry policy may create another attempt; resolution chooses exactly one terminal result for each candidate by declared rules, never “newest file wins.” |
| `BatchClosureSnapshot` | experiment collection | Freezes selector input: exact candidate census plus one resolved terminal evaluation per candidate. |
| `CheckpointSelector` | experiment policy | Pure function choosing checkpoint(s) from closed input. Pure means no filesystem mutation, job submission, or dependence on result arrival order. It records canonical ordering and tie-break trace. |
| `ObjectiveReducer` | experiment policy | Converts selected checkpoint results into one objective vector or failed outcome per trial. This is optimizer feedback, distinct from choosing checkpoints. |
| Selection snapshot | experiment policy | Records all selector inputs, policy identity, chosen candidate IDs, objective values, exclusions, and reasoning trace. |
| Retention decision and receipt | experiment policy/execution | Decision says which payloads must survive or may be released. Receipt proves an execution action produced a verifiable retained copy. Only verified receipts permit later source pruning. |

Two deterministic barriers result: producer closure freezes candidates;
evaluation closure freezes selector input. Evaluations may overlap training,
but selection cannot observe arrival order.

Separating **identity** from **locator** matters because archives may move.
Absolute paths identify storage positions, not checkpoint content. Current
manifest digest identifies manifest metadata only; the accepted experiment
adapter therefore adds the payload-file fixity described below.

Selector answers “which evaluated checkpoint satisfies policy?” Reducer
answers “what trial outcome reaches optimizer?” Retention execution answers
“which bytes were preserved?” Separation keeps each replayable and testable.

`pruned_before_evaluation` should not be a routine evaluation status: D4
requires every candidate to survive through selection. Missing source bytes are
an integrity failure. Exhausted evaluation retries may close as failed, but the
system must record a failed trial outcome rather than invent an infinite
scientific measurement.

Checkpoint choice should produce a result **per trial**, so every `ask` receives
one `tell`. RPlan favors selecting an aligned scheduled step across a trial's
seed replicas when schedules match; independent per-seed choice is simpler but
can introduce optimistic selection bias. Batch-global top-K/Pareto graduation
belongs to the later experiment `Selector`, not checkpoint-to-trial objective
reduction.

## Evidence and sequence before runtime implementation

1. Record accepted boundary and unresolved scientific defaults in the roadmap.
2. Complete the v4 foundation prerequisite and frozen-v3 reference route.
3. Add only versioned experiment contracts, a strict manifest-v1 adapter,
   core-produced golden fixtures, round-trip validation, and v4-only sidecars.
4. Test unknown versions, moving `latest.json`, deterministic identity,
   reordered arrival, missing/duplicate/late candidates, producer failure,
   retry resolution, missing/NaN metrics, deterministic ties/Pareto output, and
   failed-trial `tell`.
5. Before enabling pruning, test physical retention by removing the source and
   proving every selected retained checkpoint remains verifiable and loadable.
6. Run the full non-pilot frozen-v3 to fresh-v4 mapped E2E on `test`/`gpu_test`;
   retain production execution profiles for full science runs.

## Recorded decisions and consequences

### Ownership

Core SpENN interfaces stay as they are. Core owns complete checkpoint creation,
but not experiment artifact tracking. The experiment adapter must observe an
exact completed `step_*` directory and record it correctly. Future core
quality-of-life additions may make observation cheaper, but must not acquire
batch, selection, or retention responsibility.

### Customizable objectives

Objective shape and checkpoint-selection policy remain configurable. Energy
and local-energy variance may be the default `ObjectiveSpec`, but are not
hard-coded into contract schemas. Versioned policy configuration declares
metric names, directions, aggregation, constraints, scalarization or
Pareto/top-K behavior, and deterministic tie-breaking. `CheckpointSelector`
and `ObjectiveReducer` consume that configuration as pure policies.

### Failure-aware retries

An evaluation attempt must report structured failure evidence rather than only
`failed`. Evidence distinguishes conditions such as preemption, timeout,
out-of-memory, infrastructure failure, evaluator failure, invalid metrics, and
checkpoint-integrity failure. A pure retry policy—not the worker—maps that
evidence to same-profile retry, scientifically equivalent alternate-profile
retry, or terminal failure under an evaluator-specific budget.

Changing partition, memory, wall time, or equivalent hardware may be an
alternate-profile retry when evaluator inputs and numerical contract remain
unchanged. Changing evaluator code, data, metrics, or scientific configuration
creates a new `EvaluationSpec`, not a retry. Exhausted or non-retryable failures
become explicit terminal outcomes, allowing deterministic batch closure.

### Schedule, cap, storage, and retention

A **fixed schedule** is the persisted set of checkpoint steps that a run will
publish. With current core interfaces it must expand from periodic
`every_n_steps`/`start_step` behavior plus the terminal `train_end` checkpoint;
for example, cadence 1,000 with `max_steps=3,500` produces
`[1_000, 2_000, 3_000, 3_500]`. Freezing those explicit steps in `BatchPlan`
prevents runtime timing or storage pressure from changing selector population.
A **candidate cap** is a pre-`ask` admission ceiling; exact candidate count is
derived from admitted trials, seeds, and scheduled steps and must fit that
ceiling. It never truncates an opened batch. A **storage budget** estimates
whether all candidate bytes can remain protected through selection.
Insufficient capacity reduces or rejects admission before `ask`; it never
silently drops candidates afterward.

These are per-experiment planning inputs, not global architecture decisions,
and no numeric values are required for the contract-only slice. A physical
retention backend is needed only before source pruning is enabled; component
options and durability requirements are in
[`checkpoint-artifact-systems-decision.md`](checkpoint-artifact-systems-decision.md).

### Payload identity and fixity

Existing
[`CheckpointManifest`](../spenn/checkpoint/manifest.py) v1 names checkpoint
files and hashes resolved configuration, while `COMPLETE` proves publication
finished. Neither proves that payload bytes remained unchanged. Established
artifact formats keep metadata identity separate from payload fixity:
[BagIt RFC 8493](https://www.rfc-editor.org/rfc/rfc8493.html) requires every
payload file to appear in a checksum manifest, and
[OCI descriptors](https://specs.opencontainers.org/image-spec/descriptor/?v=v1.1.0)
carry content digest and byte size for consumer verification.

Resolution: leave core manifest v1 unchanged. Experiment-side
`CheckpointArtifactRef` records SHA-256 and size for every manifest-declared
payload file, plus a digest over the canonical sorted file table and a separate
SHA-256 of `manifest.json`. Retention verification uses the same table. A
manifest digest alone is insufficient for selected-checkpoint custody or
restore verification.

## Remaining configuration work

First concrete experiment still must choose seed grouping, default objective
policy, checkpoint schedule, retry budgets and alternate-profile rules, minimum
successful coverage, and durable retention destination. These choices belong
in versioned experiment configuration; they do not reopen ownership or
contract-shape decisions above.

Detailed staging, schemas, adapter checks, tests, and deferrals are recorded in
[`checkpoint-selection-implementation-instructions.md`](checkpoint-selection-implementation-instructions.md).

Full planner, implementation-representative, and critic discussions remain
available through Paseo.
