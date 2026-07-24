# V4-0 Bootstrap: Human Decision Memo

> **Status:** accepted decision record, 2026-07-24. No V4 implementation or
> experiment submission was performed while producing this memo.
>
> **Decision:** all three recommendations under
> [Approved decisions](#approved-decisions) were approved 2026-07-24. No P0
> architecture choice remains; disposable-root audit and fresh reference run
> are implementation evidence rather than new design decisions.
>
> **Scope:** first executable slice of the experiment restructure on
> `codex/experiment-v4-restructure`, based on current `dev` at `4eadb55`.
> Detailed execution plan:
> [`v4-0-implementation-instructions.md`](v4-0-implementation-instructions.md).

## Bottom line

Build V4-0 as a v4-owned, versioned compatibility route that invokes the
existing v3 stage CLIs by subprocess. V4 owns its study identity, configs,
results root, stage selection, explicit attempt lineage, invocation records,
layout map, comparator, and stack controller. V3 appears only as a named legacy
implementation backend.

Keep the existing numbered physical layout for this bootstrap. Track the frozen
comparison inventory in Git: small files remain raw and large tabular files use
deterministic per-file gzip. Preserve the full source run separately when
useful, but do not make routine parity checks depend on a Holystore mount.

Use a sealed source-only snapshot only if a disposable-root audit proves that
the subprocess route cannot meet the ownership rules. Do not patch individual
stages to force the preferred route to pass.

## V4-0 versus V4-1

The two slices answer different questions.

| Slice | Question answered | Work included | Work excluded |
| --- | --- | --- | --- |
| **V4-0: migration safety harness** | Can an isolated v4 study run the complete current workflow and prove that behavior did not drift? | V4 identity and results root, compatibility dispatcher, v4 smoke/config snapshots, frozen v3 reference, layout map, comparator, ten-stage `gpu_test` smoke route | New trial/search abstractions, checkpoint-candidate contracts, generic stage protocols, scheduler redesign |
| **V4-1: semantic experiment contracts** | What immutable, versioned experiment objects does v4 mean? | Only trial, named-seed, producer/attempt, metric, execution-profile, and stage-result records exercised by the live v4 route | Dormant protocols or future policy objects without a current consumer |

Boundary rule:

```text
needed only to run or verify the legacy-equivalent route  -> V4-0
defines durable scientific identity or future policy     -> V4-1
```

V4-0 therefore remains deliberately narrow. It creates the evidence harness
that makes V4-1 reviewable; it does not prebuild V4-1's data model.

## Decisions already fixed

These P0 choices no longer need review:

- planning authority is tracked under `.planning/`, with
  `.planning/TODO.md` as priority ledger and
  `experiments/toolkit-roadmap.md` as canonical v4 roadmap;
- implementation uses dedicated branch
  `codex/experiment-v4-restructure`, created from `dev` at `4eadb55`;
- v4 restructuring is resumed, with V4-0 before V4-1;
- reference and acceptance shape is non-pilot v3 `configs/smoke.yaml`;
- reference and acceptance profile is Submitit/CUDA on `gpu_test`;
- fixed scientific shape is blind seed `811`, 64 scan jobs, and one final
  replicate; and
- existing completed smoke lineages are evidence of feasibility, not suitable
  canonical references. One fresh, clean, current-source v3 lineage is needed.

Exact attempt IDs, checksums, Slurm job IDs, and audit receipts are generated
evidence. They are not design choices.

## Audit result

### Existing v3 lineages

The complete lineages
`20260708T003541-0400-smoke` and
`20260708T112116-0400-smoke-rerun` have the expected population and
`gpu_test` evidence. Neither is suitable as the frozen oracle:

- source provenance is incomplete or includes a dirty tree;
- the lineages predate later v3 behavior and artifact changes;
- the retained resource shape differs from the current smoke stack; and
- the exact historical dirty patch cannot be reconstructed.

V4-0 should run exactly one fresh bounded v3 reference from a clean source
revision. It must use an isolated staging root, not mutate the live v3 results
tree, and record every explicit stage attempt.

### Ten-stage routing

Static CLI inspection supports one generic subprocess dispatcher:

| V4 logical role | Legacy CLI | Required explicit ownership |
| --- | --- | --- |
| `screen_plan` | `plan.py` | v4 grid/config, root, attempt, timezone, blind seed |
| `screen_train` | `train.py` | root, grid attempt, backend and resource profile |
| `screen_eval` | `validate.py` | root, grid/train/eval attempts, config and profile |
| `screen_collect` | `collect.py` | root, grid/collection attempts |
| `select` | `select_champions.py` | root, collection/selection attempts |
| `confirm_plan` | `final_plan.py` | root, selection/final-grid attempts and configs |
| `confirm_train` | `final_train.py` | root, final-grid/train attempts and profile |
| `confirm_eval` | `final_eval.py` | root, final-grid/train/eval attempts and profile |
| `confirm_collect` | `final_collect.py` | root, final-grid/eval/collection attempts |
| `report` | `final_report.py` | root, final-collection/report attempts |

Known coupling remains in study-local imports, `Path(__file__)` defaults,
Submitit callable import paths, and optional launcher re-execution. The
dispatcher must neutralize defaults through explicit arguments and the correct
Submitit environment. A disposable-root plan-stage and fan-out audit remains
mandatory before the bridge is accepted as implemented.

## Recommended contract

### V4 owns

- `pair_stability_v4` study and config identity;
- an absolute, guarded v4 results root with a v4 sentinel;
- logical stage selection and explicit upstream attempt IDs;
- a versioned route table and exact subprocess argv;
- stack ordering, waits, expected-count guards, and resource profile;
- additive dispatch provenance;
- reference freezing, layout mapping, and comparison policy; and
- all future replacement entrypoints.

### Legacy v3 owns temporarily

- implementation of each stage CLI;
- established stage-local semantics and artifact production; and
- existing Submitit worker callables reached by those CLIs.

This dependency is visible and replaceable. It is not allowed to become an
implicit import API.

### Required bridge gates

The dispatcher qualifies as a real V4-0 route only if all gates pass:

1. Every scientific identity, config path, results path, profile, and attempt
   ID is supplied by v4.
2. Every resolved write stays below the guarded v4 root. V3 root, children,
   symlink escapes, and traversal are rejected before launch.
3. Candidate artifacts identify as `pair_stability_v4`; v3 names appear only
   in explicit compatibility provenance.
4. One declarative mechanism handles all ten stages. No output rewriting,
   monkeypatching, direct imports, v3 edits, or stage-specific corrective
   wrappers are permitted.
5. Every stage remains independently invocable from explicit upstream
   artifacts.
6. Exact argv, source hashes, config hashes, environment identity, commit/dirty
   identity, timestamps, and exit status are recorded.
7. Pre/post inventories prove the run did not change v3 source or results.
8. Source digest mismatch and unexpected legacy launcher re-execution fail
   closed.

Pin the v3 study source closure used by the route: ten stage scripts plus
study-local launch, utility, statistics, plotting, and config files. Record
digests of runtime dependencies such as `experiments/toolkit`, `run.py`,
`spenn/`, and `uv.lock` in every E2E. Do not freeze those evolving runtime
modules into the dispatcher contract; behavior changes are instead exposed by
the parity gate.

If any gate fails because behavior is irreducibly source-location-owned, copy
the minimum source dependency closure into a sealed legacy snapshot, record
origin hashes, copy no results, and schedule removal by V4-7.

## Initial layout and configs

Keep physical directories and protected artifact names unchanged in V4-0:

```text
00_grid -> 01_train -> 02_validation -> 03_collect -> 04_select
        -> 05_final_grid -> 06_final_train -> 07_final_eval
        -> 08_final_collect -> 09_final_report
```

`layout_map.json` records conceptual v4 names, but initially maps them to the
same numbered paths. This separates logical vocabulary from a risky layout
rename. Later physical renames require a new reviewed map version and negative
comparison tests.

Copy the minimal smoke, train, and validation configs into the v4 study. Change
only v4-owned identity and path fields. Record source hashes and test that all
scientific values remain equal to the v3 source. Do not introduce a shared
config abstraction during V4-0.

## Frozen reference form

Keep mandatory comparison data portable and reviewable:

- track `reference.json`, `layout_map.json`, and small structured/text files
  raw;
- deterministically gzip large tabular files above one reviewed threshold
  (recommended: 1 MiB), with `mtime=0` and empty filename/header;
- record raw and stored SHA-256, raw and stored sizes, encoding, CSV header, and
  row count;
- stream decompressed logical content during comparison; and
- keep full raw lineage and checkpoints outside Git when useful, preferably on
  the named durable tier, but never require that external copy for routine
  parity.

Observed stale inventories are roughly 60–65 MiB raw. Their large CSVs compress
enough to make a tracked fixture practical. A Holystore-only payload would make
review and mandatory E2E checks depend on cluster mount availability and
permissions. No artifact database, custody abstraction, checkpoint copy, or
new package belongs in V4-0.

`reference.json` must include source commit/dirty identity, explicit attempts,
config and source hashes, blind seed, populations, profile/resources,
environment, submitted command manifest, inventory hashes, comparator version,
layout-map digest, volatile allowlist, and numeric tolerances.

## Smallest implementation sequence

1. Reconcile and commit the planning rename/memo on the feature branch.
2. Add v4-owned configs, root sentinel/guard, identity layout map, and legacy
   source manifest.
3. Add immutable `LegacyStageRoute` records and pure command rendering.
4. Audit representative low- and fan-out stages in a disposable v4 root. Use
   the sealed-snapshot fallback if any ownership gate fails.
5. Add one subprocess dispatch path and additive per-stage provenance.
6. Add a v4 stack controller mirroring the ten-stage waits, count checks,
   flags, and dependency shape.
7. Add lineage auditor, reference freezer/verifier, and streaming comparator.
8. Run focused tests.
9. From a clean recorded source revision, run exactly one fresh v3
   `smoke.yaml` stack on `gpu_test` in isolated staging; audit and freeze it.
10. Run the fresh v4 ten-stage smoke on `gpu_test`; compare the complete mapped
    inventory.
11. Record reference ID, source revision, attempts, Slurm evidence, dispatcher
    verdict, and comparison result in roadmap and TODO.

V4-0 stays study-local. Do not extract a toolkit abstraction until a second
consumer establishes the common contract.

## Required evidence

Focused tests must cover:

- root ownership, traversal, symlink escape, and v3-tree immutability;
- exact explicit argv for all ten routes and rejection of reserved overrides;
- source-digest mismatch, subprocess failure, and launcher re-execution;
- v4 identity in artifacts and legacy identity only in provenance;
- independently runnable stage commands and exact attempt propagation;
- reference rejection for incomplete stages, wrong seed/count/profile/source,
  ambiguous latest resolution, and mixed lineages;
- comparator self-comparison plus rejection of same-root candidate/reference;
- changed header, row order, metric, seed, integer count, map entry,
  missing/extra artifact, and candidate v3 identity;
- float tolerance only at declared typed metric fields; and
- deterministic gzip round trip and descriptor-integrity failures.

Acceptance still requires one real fresh ten-stage v4 `gpu_test` E2E against
the frozen matching v3 reference. Unit tests, dry runs, and plan diffs do not
replace it.

## Approved decisions

Approved together by human review on 2026-07-24:

1. **Bootstrap owner:** accept v4-owned, versioned dispatcher over pinned v3
   subprocess CLIs as a real V4-0 route when every bridge gate passes.
2. **Initial layout:** keep numbered physical stages and artifact names for
   V4-0; expose conceptual names only through versioned `layout_map.json`.
3. **Reference storage:** track portable comparison inventory; store small
   files raw and large tables as deterministic per-file gzip. External durable
   raw lineage is optional supporting evidence, not the parity authority.

P0 architecture is closed. Remaining work is evidence collection:
disposable-root audit, clean fresh v3 reference ID/checksums, and subsequent v4
E2E result.

## Anti-goals

No v3 edits, result-data mutation or deletion, direct legacy imports,
monkeypatching, output rewriting, physical stage rename, broad comparator
normalization, V4-1 contracts, checkpoint feature work, dynamic scheduler,
artifact database, new dependency, commit, push, or experiment submission is
part of this planning step.
