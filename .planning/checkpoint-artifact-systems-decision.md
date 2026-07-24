# Checkpoint Artifact Systems: Human Decision Memo

**Status:** decision record, updated 2026-07-24; no package installation or
implementation authorized.

**Scope:** future v4 checkpoint candidates and selected-checkpoint retention.
Live v3 `train → validate → collect → select` behavior remains unchanged.

**Decision:** use a composable architecture and a small proof of concept rather
than make one workflow or artifact-database product authoritative.

## Bottom line

No single package needs to own checkpoint creation, scientific identity,
selection, storage, lineage queries, and dashboards. Existing tools cover
useful parts reliably, but none should become the authority for every part.

Recommended composition:

```text
v4 experiment immutable records
  scientific identity, lineage, selection, and retain/release intent
        |
single finalizer/reconciler
  archive, verify, publish receipt, and prove when source may become prune-safe
        |
restic on named durable storage
  checkpoint bytes, deduplication, integrity checks, and restoration
        |
optional rebuildable projections
  PostgreSQL or ML Metadata for queries; MLflow for human browsing
```

When the separate custody/finalizer slice begins, define its small
backend-neutral archive receipt together with a plain verified-copy adapter so
the contract is exercised against real bytes. Then test **restic** as the first
external custody backend. Adopt it if realistic checkpoint tests meet restore,
failure-recovery, storage, and operational gates. Do not add an unused receipt
schema to the earlier checkpoint-candidate contract slice.

Use **DataLad/git-annex** instead when independently managed replicas, safe
drop, or offline exchange are primary requirements. Use **DVC** only if
retention should deliberately follow Git/DVC revisions. Treat
**disk-objectstore** as an embedded content-addressed-store component, not a
complete artifact system. Add a database or MLflow only after a query or UI
need appears; both remain disposable projections of canonical records.

Durable tiers are now identified: `$HOME` hosts personal scripts and control
material, `/n/holystore01/LABS/kozinsky_lab/Lab/User/rhu` permanently preserves
potentially useful artifacts and reusable databases, and `/n/netscratch` holds
experimental source/staging data. Before a custody proof of concept, a
read-only inventory must still establish exact archive path, quota/capacity
owner, restore procedure, deletion operator, and measured acceptance limits.
A restic repository located only on `/n/netscratch` is still temporary. FASRC
currently documents paid Lab Storage as having daily/weekly snapshots and
disaster recovery, while complimentary Lab Directory storage has neither; the
exact allocation and policy of the chosen path must be confirmed before the
proof of concept treats it as durable evidence.
([FASRC storage offerings](https://docs.rc.fas.harvard.edu/kb/data-storage-workflow-rdm/),
[scratch policy](https://docs.rc.fas.harvard.edu/kb/policy-scratch/))

## Capability split

The terms below name different responsibilities. Keeping them separate is what
makes several partial tools useful together.

| Capability | Meaning | Recommended owner |
| --- | --- | --- |
| Scientific authority | Defines which run, attempt, step, batch, and selection a checkpoint belongs to. | Versioned v4 experiment records |
| Payload custody | Preserves exact checkpoint bytes and can restore them after the source disappears. | Archive backend on durable storage |
| Finalization | Connects a completed source checkpoint to custody, verifies it, and publishes proof. | One experiment-side actor |
| Retention policy | Decides which scientifically meaningful artifacts must survive. | Experiment policy records |
| Query index | Answers cross-run lineage and status queries efficiently. | Optional rebuildable database |
| Human UI | Shows metrics, runs, and artifact links. | Optional MLflow mirror |
| Garbage collection | Deletes only payloads unreachable from a complete, reviewed set of retention roots. | Separate future audited process |

An archive locator is not scientific identity. For example, a restic snapshot
ID, git-annex key, DVC hash, filesystem path, or database row says where a
backend can find bytes. SpENN identity says what those bytes mean. Migrating
the same payload to another backend should create another custody receipt
without changing checkpoint identity.

Likewise, a JSONL reference or database row does not retain bytes. A valid
receipt must prove that an independently restorable copy exists. Hardlinks are
not independent custody because source and archive share one inode. Reflinks
or copies on the same purgeable filesystem do not protect against loss of that
filesystem.

## Component assessment

| Component | Part it can supply | Decision |
| --- | --- | --- |
| Plain verified copy | Simple payload custody and a test oracle for archive semantics. | Build the smallest reference adapter first: copy to destination-local staging, verify, atomically publish, then write receipt. No deduplication, so it may not remain the production backend. |
| [restic](https://restic.readthedocs.io/en/latest/040_backup.html) | Encrypted, content-addressed, deduplicated snapshots; local/SFTP/REST/object-store backends; concurrent backups; restore and repository checking. | Best first external proof of concept. Store the full snapshot ID as a locator, but verify restored files against SpENN's independent digest. Key recovery and maintenance ownership are real operational costs. |
| [DataLad/git-annex](https://git-annex.branchable.com/copies/) | Checksum-addressed content, replica-count policy, required content, verification, and guarded removal. | Strong choice when replica placement and safe drop matter more than a simple central archive. Git/annex branch coordination and operator complexity are unnecessary for the first central-custody use case. |
| [DVC](https://dvc.org/doc/command-reference/gc) | Content-addressed cache/remotes tied to Git-versioned data declarations. | Conditional choice for a deliberately Git-rooted artifact workflow. DVC garbage-collection roots come from selected Git/DVC revisions; SpENN selection records would not automatically protect content. Shared-cache GC requires special care. |
| [disk-objectstore](https://disk-objectstore.readthedocs.io/en/latest/pages/design.html) | Embedded SHA-256 content-addressed objects, deduplication, streaming, and concurrent loose-object writes. | Useful narrow library or design precedent. It does not supply checkpoint trees, scientific lineage, retention roots, replicas, receipts, or domain-aware GC. Packing introduces a SQLite index and exclusive-maintenance constraints that need filesystem and Python-version qualification. |
| [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) plus PostgreSQL | Artifact/execution/context/event lineage and central queries; artifacts refer to payload URIs. | Possible later projection, not payload storage or authority. Current MLMD binary availability must be qualified against the selected runtime; a separately pinned service is acceptable. Plain project-specific PostgreSQL tables may be smaller if generic MLMD concepts add little. |
| [MLflow](https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-gc) | Familiar run, metric, and artifact-link browsing UI. | Optional asynchronous mirror only. MLflow GC does not interpret SpENN selection roots and explicitly does not use pins, tags, or registered-model associations as retention authority. |
| AiiDA, Metaflow, Rucio, or iRODS | Larger workflow/provenance systems or institution-scale data-management services. | Consider only as an intentional platform migration or when an already operated institutional service is available. Their operational and ownership scope is larger than this boundary. |

PostgreSQL is preferable to concurrently written SQLite when a shared,
multi-node catalog becomes necessary. SQLite documents network-filesystem
locking and synchronization risks and recommends a client/server database when
data is separated from the application by a network.
([SQLite network guidance](https://sqlite.org/useovernet.html))

Core SpENN checkpoint interfaces remain unchanged. The experiment-side
finalizer observes an exact completed `step_*` artifact and records it; core
does not track experiment artifacts. Future core quality-of-life features may
make observation cheaper, but they must not acquire experiment identity,
selection, or retention responsibility.

The current workspace is NFSv3. Initial ingestion should therefore use one
controller/finalizer. Workers can publish immutable per-attempt records, but
should not append concurrently to one JSONL file or share a mutable SQLite
catalog. This restriction can be relaxed only after representative multi-node
tests.

## Correct archive contract

The smallest useful backend-neutral interface is:

```text
ArtifactArchive.archive(source, expected_digest) -> backend locator
ArtifactArchive.verify(locator, expected_digest) -> verification evidence
ArtifactArchive.materialize(locator, destination) -> restored checkpoint
```

It should not expose deletion yet. Retention and garbage collection have not
been specified strongly enough for a safe generic `delete`.

`ArchiveReceiptV1` should contain:

```text
receipt schema version
checkpoint scientific ID
source checkpoint manifest digest
canonical sorted file table: relative path, size, SHA-256
backend-independent payload digest
storage tier and repository identity
backend type, version, and immutable locator
verification method, result, and timestamp
retention-request ID
```

Define canonical path, number, and serialization rules. Either hash a
schema-defined semantic projection or use a canonical JSON serialization such
as [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html); ordinary
`json.dumps()` output is not a cross-version identity rule.

Publication order matters because no transaction spans scratch storage,
archive storage, and experiment records. For any retention request—whether it
protects a pre-selection candidate or a checkpoint already selected—the
custody sequence is:

```text
completed checkpoint and COMPLETE marker
  → immutable RetentionRequested record
  → archive payload created
  → every file verified against canonical digest
  → immutable ArchiveReceiptV1 published
  → receipt bound to candidate or selection by another immutable record
  → source becomes eligible for later pruning
```

Receipt publication is the commit point. A crash before it leaves the source
protected; reconciliation repeats or completes archive work. A crash after
backend publication but before receipt may leave an orphaned archive object,
which is a safe storage leak until reconciliation, not permission to delete
the source. Failure of PostgreSQL, MLMD, or MLflow projections cannot invalidate
canonical records or custody.

Scientific selection stays pure and may precede archiving a selected payload,
provided the source remains protected. A later binding record connects that
selection to a verified receipt. Archive execution never changes which
checkpoint won.

Selected checkpoints without verified durable receipts remain protected.
Source pruning, archive retention, and later GC must be separate actions.
Future GC needs immutable root closure, mark and dry-run reports, a quarantine
or grace period, sweep receipts, and tested recovery. Restic `forget` removes
snapshot references and `prune` removes unreferenced data, so generic
time-based restic retention must not replace experiment retention policy.
([restic retention behavior](https://restic.readthedocs.io/en/latest/060_forget.html))

## Recommended proof of concept

1. **Name storage and operators.** Record destination policy, quota/capacity
   owner, archive writer, separate deletion authority, credential/key escrow,
   expected replica count, and restore service level.
2. **Establish reference behavior.** Test plain verified copy on actual source
   and destination filesystems. Inject process death, quota exhaustion,
   checksum mismatch, source mutation, path traversal/symlinks, and duplicate
   requests. No failure may publish a valid receipt; retry must converge.
3. **Test restic against the reference.** Use representative checkpoint sizes
   and change patterns. Measure archive time, restore time, stored size, and
   deduplication. Test simultaneous backups, process/node death, stale-lock
   recovery, source deletion, exact restore, and `restic check --read-data`.
   Accept only complete backup status; do not automate `forget` or `prune`.
4. **Test failure-domain recovery.** Make the primary source unavailable and
   restore from the designated durable copy or second repository. Compare
   every restored path, size, and SHA-256 with the receipt.
5. **Add other components only on evidence.** Evaluate DataLad/git-annex when
   replica rules or offline exchange are required; disk-objectstore when direct
   content-addressed reads materially improve measured restore cost; a database
   when immutable-record scans miss a query-latency or concurrency target; and
   MLflow when users need a dashboard.

No candidate tool is currently on the repository's declared dependency path,
and no installation was performed during this exploration. Environment
qualification belongs in the approved proof of concept, using the repository's
normal reproducible tooling.

## Recorded human decisions and remaining work

1. **Storage tiers — decided.** `$HOME` and
   `/n/holystore01/LABS/kozinsky_lab/Lab/User/rhu` are durable tiers;
   `netscratch` is experimental source/staging. `holystore01` is the natural
   permanent checkpoint-payload tier. Read-only scouting may inventory these
   locations and account for future use of other clusters and `glob` transfer.
2. **Retention priority — decided; exact replica count remains open.** Data
   retention and reproducibility outrank storage economy unless retained data
   threatens capacity for roughly the next dozen experiments. At least one
   verified durable receipt is required before scratch pruning. A later
   quantitative capacity decision must choose whether selected checkpoints
   require one or two durable copies.
3. **Encryption — decided.** Encryption is not a design goal. Artifact lineage
   and reproducibility have priority. Any backend credential still requires an
   operational recovery owner.
4. **Proof-of-concept success thresholds — open.** Produce a separate
   decision memo covering archive and restore latency, storage growth,
   deduplication, failure recovery, representative capacity, replica count,
   restore-test cadence, and pass/fail thresholds.
5. **Replica-placement tooling — decided for now.** DataLad/git-annex and
   offline-exchange policy are not current requirements. Keep the
   backend-neutral receipt boundary so another cluster or transfer system can
   be added later.
6. **Metadata query/UI surface — decided for now.** Immutable records are
   sufficient. Do not add PostgreSQL, ML Metadata, or MLflow until a measured
   query or user-interface need appears.

The adjacent
[`checkpoint-selection-boundary-decision.md`](checkpoint-selection-boundary-decision.md)
now explains the run/experiment boundary and corrected immutable contract model
in full. This memo narrows the separate question of how existing components can
provide payload custody, metadata indexing, and user-facing views.
Detailed checkpoint-contract staging is in
[`checkpoint-selection-implementation-instructions.md`](checkpoint-selection-implementation-instructions.md);
it deliberately defers archive receipts until a custody/finalizer slice can
exercise them against real bytes.

Full scout, implementation-representative, and critic discussions remain
available through Paseo.
