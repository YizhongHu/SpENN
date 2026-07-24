# V4-0 Experiment Restructure: Implementation Instructions

> **Status:** approved implementation plan, 2026-07-24.
>
> **Authority:** implements approved decisions in
> [`v4-0-bootstrap-decision-memo.md`](v4-0-bootstrap-decision-memo.md) and the
> V4-0 slice in
> [`experiments/toolkit-roadmap.md`](../experiments/toolkit-roadmap.md).
>
> **Branch:** `codex/experiment-v4-restructure`, based on `dev` at `4eadb55`.
>
> **Scope:** V4-0 only. This record authorizes planning and later implementation
> on the feature branch; it does not record completed code, tests, reference
> runs, or E2E evidence.

## Outcome

Build a study-local compatibility harness:

```text
v4 identity/config/root/controller
  -> versioned dispatcher
  -> pinned v3 CLI subprocess
  -> numbered v4 artifact tree
  -> frozen v3 reference comparison
```

V4-0 answers whether isolated v4 can run the complete current experiment and
prove behavior parity. It does not add V4-1 trial, seed, producer, metric,
search, or generic stage contracts.

No implementer-blocking design decision remains. Remaining gates are
operational evidence: clean source, disposable ownership audit, fresh v3
reference, fresh v4 E2E, and zero-difference comparison.

## Terms

| Term | Meaning |
| --- | --- |
| **Legacy bridge** | V4-owned dispatcher invoking an existing v3 stage CLI by subprocess. |
| **Route** | Versioned declarative mapping from one v4 logical role to one v3 script and exact owned arguments. |
| **Protected inventory** | Frozen artifacts whose presence, structure, ordering, values, and lineage define parity. |
| **Reference** | One audited, immutable, tracked comparison inventory derived from a fresh completed v3 smoke lineage. |
| **Candidate** | One fresh completed v4 smoke lineage compared with the reference. |
| **Study closure** | V3 stage-local source and configs whose exact hashes are fail-closed by the dispatcher. |
| **Runtime closure** | Toolkit, `run.py`, `spenn`, lockfile, and environment used by workers. Its digest is recorded, not dispatcher-pinned. |
| **Experiment write** | Study artifact, pointer, state, plan, receipt, or report owned by v4 and required to stay below the guarded root. |
| **Operational write** | Recorded uv cache/environment, Slurm state, or OS temporary write outside experiment artifacts. These are allowed and do not weaken root ownership. |

## Approved decisions

1. V4-owned, versioned dispatcher over pinned v3 subprocess CLIs qualifies as
   real V4-0 when every ownership gate passes.
2. Physical stages remain `00_grid` through `09_final_report`; conceptual names
   live in versioned layout map.
3. Portable reference inventory is tracked. Small artifacts stay raw; large
   tables use deterministic per-file gzip.
4. Reference and candidate use non-pilot `smoke.yaml`, blind seed `811`,
   Submitit/CUDA `gpu_test`, 64 scan jobs, and one final replicate.
5. Existing July lineages are not canonical. Run exactly one fresh v3
   reference from clean current source.
6. Sealed source-only snapshot is fail-closed fallback, not default.

## Non-negotiable invariants

1. Do not edit v3 source or mutate live v3 results.
2. Invoke v3 only through subprocess. No direct imports, monkeypatching, output
   rewriting, or stage-specific repair wrappers.
3. One new absolute, guarded results root belongs to one stack lineage. Never
   reuse it for another reference or candidate.
4. All experiment writes stay below guarded v4 root. Reject v3 roots,
   descendants, broad roots, traversal, and symlink escapes.
5. V4 owns study/config identity, route, profile, root, output attempt, and every
   upstream attempt input exposed by legacy CLI.
6. Legacy `collect.py` cannot accept exact validation attempt IDs and resolves
   per-run latest validation attempts. Determinism therefore comes from fresh
   single-lineage root plus post-stage audit. Do not claim stronger explicit
   control.
7. No arbitrary trailing argv reaches legacy CLI. Dispatcher accepts typed v4
   inputs and renders complete argv itself.
8. Fan-out dispatch starts inside `.venv-submitit`. V4-0 forbids legacy
   `--wait-job`, because dependent re-launch would invoke v3 outside dispatcher.
9. Preserve legacy `StagePlan.smoke == false`. Smoke identity comes from grid
   digest, blind seed, population, and execution profile.
10. Runtime dependency drift remains visible through provenance and parity; do
    not freeze evolving toolkit/core modules into bridge source manifest.
11. Unit tests, dry rendering, and plan diffs supplement but never replace real
    v3-reference/v4-candidate E2E.
12. Preserve all experiment outputs and failure evidence. Never remove them
    automatically.

## Planned source layout

Keep implementation study-local:

```text
experiments/hooke/pair_stability_v4/
  README.md
  configs/
    smoke.yaml
    pair_stability.yaml
    pair_validation.yaml
  legacy_routes_v1.json
  legacy_source_v1.json
  roots.py
  routes.py
  dispatch.py
  audit.py
  reference.py
  compare.py
  submit_stack.sh
  reference/
    layout_maps/
      v1.json
    v3_smoke/
      <reference-id>/
        reference.json
        inventory/...
  test_routes.py
  test_reference.py
  test_pair_stability_v4.py
```

Do not change `experiments/toolkit` in V4-0. Do not add `__init__.py` unless
fresh-process execution proves package semantics necessary. Existing study
scripts use study-local top-level imports.

## Work package 0 — close planning state

Before feature code:

1. Verify human approval is recorded in decision memo, TODO, and roadmap.
2. Keep this implementation record linked from both planning ledgers.
3. Reconcile staged planning moves without deleting unrelated user work.
4. Create one clean local planning commit on feature branch.
5. Keep branch unpushed until later explicit instruction.

Canonical reference evidence must never originate from dirty source. If current
checkout contains unrelated user files, preserve them and use a safe clean
checkout for evidence rather than deleting or moving them.

## Work package 1 — v4 configs, root, and source identity

### Config copies

Copy current v3 smoke, train, and validation configs into v4. Change only:

- `pair_stability_v3` to `pair_stability_v4`;
- v3 config paths to v4 equivalents;
- v3 results paths to v4 equivalents; and
- comments/examples that identify v3.

Add semantic equality test after these reviewed substitutions. All scientific
values, axis order, seeds, choices, metrics, workloads, and config structure
must remain exact. Do not extract shared config abstraction in V4-0.

### Root owner

`roots.py` owns root semantics:

```python
def initialize_root(path: Path, *, lineage_id: str) -> Path:
    """Create and identify one new pair-stability-v4 results root."""

def require_v4_root(path: Path, *, lineage_id: str | None = None) -> Path:
    """Resolve and validate one existing pair-stability-v4 results root."""

def require_beneath_root(path: Path, root: Path) -> Path:
    """Resolve a candidate artifact path and reject root or symlink escape."""

def validate_root_links(root: Path) -> tuple[str, ...]:
    """Return unsafe internal links or pointers that resolve outside the root."""
```

Sentinel path:

```text
<results-root>/.pair_stability_v4-root.json
```

Minimum sentinel:

```text
schema_version = pair-stability-v4/root/v1
study = pair_stability_v4
canonical_root
lineage_id
created_at
```

Initialization accepts only absent root or empty explicitly selected directory.
Reject `/`, repository root, v3 study/root or descendants, root symlink,
nonempty unsentinelled directory, mismatched sentinel, and canonical path
mismatch.

### Legacy source pin

`legacy_source_v1.json` records repository-relative path and SHA-256 for:

- ten v3 stage scripts;
- `launch.py`, `plot.py`, and `stats.py`;
- all runtime `utils/*.py`; and
- v3 smoke, train, and validation source configs.

Exclude v3 results, caches, tests, `parity.py`, `sync.py`, and README. Unrelated
docs/tests must not block dispatcher.

Record but do not fail-pin:

- tracked `experiments/toolkit/**/*.py`;
- `run.py`;
- tracked `spenn/**/*.py`;
- `pyproject.toml` and `uv.lock`;
- Python, PyTorch, CUDA, executable, and selected uv environment identity.

Reference and candidate commits will differ after reference fixture is
committed. Compare relevant source/runtime closure digests, not whole commit
equality.

## Work package 2 — versioned routes and dispatcher

### Route schema

`legacy_routes_v1.json` owns:

```text
schema_version
logical_role
physical_stage
legacy_script
kind = local | fanout
required_input_attempts
output_attempt_rule
config arguments
fixed smoke-gpu-test arguments
```

`routes.py` owns validation and rendering:

```python
@dataclass(frozen=True)
class LegacyStageRoute:
    """Describe one versioned V4-0 subprocess route without importing v3."""

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LegacyStageRoute":
        """Build and validate one route from serialized route data."""

def load_routes(path: Path) -> Mapping[str, LegacyStageRoute]:
    """Load exactly one unique route for every V4-0 logical stage."""

def render_legacy_argv(
    route: LegacyStageRoute,
    *,
    results_root: Path,
    output_attempt: str,
    input_attempts: Mapping[str, str],
    repo_root: Path,
) -> tuple[str, ...]:
    """Render complete pinned legacy argv from typed V4-owned inputs."""

def verify_legacy_source_manifest(repo_root: Path) -> tuple[str, ...]:
    """Return missing or digest-mismatched files in pinned v3 study closure."""

def require_launcher_environment(route: LegacyStageRoute, repo_root: Path) -> None:
    """Reject fan-out dispatch outside expected Submitit launcher environment."""
```

Do not add generic `StageEntrypoint`, `StageContext`, profile protocol, search
vocabulary, or executor factory.

### Dispatch execution

`dispatch.py` owns CLI and subprocess:

```python
def dispatch_stage(
    role: str,
    *,
    results_root: Path,
    output_attempt: str,
    input_attempts: Mapping[str, str],
) -> int:
    """Validate ownership, persist receipts, and invoke one pinned v3 CLI."""

def main(argv: Sequence[str] | None = None) -> int:
    """Render or run one independently invocable V4-0 stage."""
```

Rules:

- use argv sequence, never shell;
- child command begins with current expected Python, `-B`, and legacy script;
- set `PYTHONDONTWRITEBYTECODE=1`;
- run from repository root;
- inherit stdout/stderr so independent CLI and controller logs remain readable;
- validate attempt syntax, root sentinel, source hashes, configs, and upstream
  artifact paths before launch;
- no arbitrary pass-through options;
- preserve exact subprocess return code and signal outcome;
- never rewrite child output.

Persist separate immutable receipts:

```text
<results-root>/_v4/dispatch/<output-attempt>/<logical-role>/<invocation-id>/
  request.json
  result.json
```

`request.json` commits invocation intent before process start. `result.json`
records terminal outcome, timestamps, exact argv, cwd, source/runtime receipts,
and post-run root audit. Crash may leave request without result; never convert
that into successful completion.

### Exact route matrix

| Logical role | Legacy CLI | V4-owned arguments |
| --- | --- | --- |
| `screen_plan` | `plan.py` | v4 smoke grid, v4 train config, root, output attempt, `America/New_York`, `--blind`, seed `811`, recorded Python token |
| `screen_train` | `train.py` | root, grid attempt, repo root, Submitit/CUDA, chunk `32`, CPUs `4`, `gpu_test`, `8 GB/CPU`, timeout `60` |
| `screen_eval` | `validate.py` | root, grid/train/output attempts, materialized validation config, repo root, same profile, chunk `32`, timeout `120` |
| `screen_collect` | `collect.py` | root, grid attempt, output attempt; audit per-run resolved validation attempts |
| `select` | `select_champions.py` | root, collection attempt, output attempt |
| `confirm_plan` | `final_plan.py` | root, selection/output attempts, materialized train/eval configs, `--replicates 1` |
| `confirm_train` | `final_train.py` | root, final-grid/output attempts, config, repo root, Submitit/CUDA, chunk `8`, CPUs `4`, `gpu_test`, `8 GB/CPU`, timeout `60` |
| `confirm_eval` | `final_eval.py` | root, final-grid/final-train/output attempts, config, repo root, same profile, chunk `8`, timeout `120` |
| `confirm_collect` | `final_collect.py` | root, final-grid/final-eval/output attempts |
| `report` | `final_report.py` | root, final-collection/output attempts |

Legacy scan-train attempt equals grid attempt because `train.py` has no
separate `--attempt-id`. Preserve this; producer-attempt redesign belongs to
V4-1.

Materialized config paths come from guarded grid/final-grid manifests. Resolve
them and prove containment before use. `confirm_plan` must never reach v3
`DEFAULT_GRID` fallback.

## Work package 3 — v4 stack controller

Implement v4 `submit_stack.sh` by copying behavior, not source ownership, from
current v3 controller.

Initial supported mode: `smoke`.

Requirements:

- CPU controller keeps current `sapphire` controller shape and records it
  separately from scientific `gpu_test` profile;
- low-fanout roles use normal uv environment;
- four fan-out roles enter `.venv-submitit` before calling v4 dispatcher;
- controller calls no v3 script directly;
- no `--wait-job`;
- one fixed stack/lineage ID propagates through every stage;
- preserve execution-record job extraction, `squeue` wait, `sacct` terminal
  check, expected-count guards, and report existence check;
- enforce 64 collected screening rows and eight final rows;
- persist controller script, command, commit/branch/dirty receipt, profile,
  Slurm IDs, dispatch receipt paths, stage status, and pre/post legacy-tree
  inventories below `_v4/stack/<lineage-id>/`.

Shell tests require `bash -n`, no-submit usage path, exact role order, every
role once, exact profile flags, explicit attempt propagation, waits, and count
guards.

## Work package 4 — disposable ownership audit

Complete before canonical v3 submission.

Expose this only as a typed `dispatch.py audit` subcommand. It supports exactly
one plan row and one local CPU train row, writes `purpose=ownership_audit`, and
cannot select arbitrary legacy flags. It is not a second science profile or a
route-table entry.

1. Snapshot pinned v3 source content and v3 results path/type/size/mtime/link
   metadata. Do not hash historical checkpoint payloads.
2. Initialize disposable absolute v4 root with `purpose=ownership_audit`.
3. Dry-render all ten routes.
4. Execute one-row `screen_plan` through audit subcommand with fixed
   `--limit 1`.
5. Execute one representative `screen_train` locally on CPU through same
   subprocess/receipt machinery, fixed to chunk size one.
6. Verify every experiment path stays under audit root and candidate artifacts
   identify as v4.
7. Verify v3 source and live results metadata are unchanged.
8. Persist audit receipt.

Reference freezer must reject audit-purpose roots.

Environment failure stops for interactive resolution under repository rules.
It does not activate source snapshot fallback.

### Snapshot fallback triggers

Switch entire route version to sealed minimum source snapshot only when actual
audit proves one or more:

- experiment write escapes guarded root;
- v4 identity cannot be supplied without output rewriting;
- source-location/import behavior cannot survive subprocess or Submitit;
- isolated root cannot neutralize legacy latest/default resolution;
- stage requires v3 edit, direct import, monkeypatch, or corrective wrapper; or
- stage cannot remain independently invoked.

Do not mix live-v3 and snapshot routes by stage. Do not activate fallback for
numerical mismatch, Slurm unavailability, uv environment trouble, or ordinary
recorded operational writes.

## Work package 5 — lineage audit and reference freezer

`audit.py` owns read-only receipts:

```python
def inventory_source_tree(root: Path) -> tuple[dict[str, object], ...]:
    """Return deterministic content receipt for pinned source files."""

def inventory_results_tree(root: Path) -> tuple[dict[str, object], ...]:
    """Return deterministic metadata receipt without hashing result payloads."""

def audit_completed_lineage(
    results_root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[str, ...]:
    """Return every reason one lineage cannot become V4-0 reference."""

def audit_identity(results_root: Path, *, attempts: Mapping[str, str]) -> tuple[str, ...]:
    """Return candidate artifacts that retain wrong scientific identity."""
```

Reference audit requires:

- clean captured source;
- one explicit stage chain;
- blind seed `811`;
- exact v4-approved smoke grid digest;
- 64 planned, launched, and collected scan rows;
- one configured final replicate and eight final rows;
- 64/64/8/8 fan-out task populations;
- terminal execution records and required status/checkpoint/result evidence;
- CUDA on `gpu_test` for every fan-out stage with exact logical resources;
- zero missing protected artifacts;
- exact low-stage source chain;
- resolved collect pointers all target expected validation lineage;
- no mixed commits, dirty workers, fallback defaults, or unexpected latest
  resolution.

`reference.py` owns storage:

```python
@dataclass(frozen=True)
class ReferenceArtifact:
    """Describe stored encoding and verified logical bytes for one artifact."""

def enumerate_inventory(
    results_root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[Path, ...]:
    """Return ordered protected paths from explicit lineage manifests."""

def freeze_reference(
    results_root: Path,
    destination: Path,
    *,
    attempts: Mapping[str, str],
) -> Path:
    """Atomically create one immutable reference from audited source lineage."""

def verify_reference(reference_dir: Path) -> tuple[str, ...]:
    """Verify descriptor, stored bytes, decoded bytes, schemas, and counts."""

def open_logical_content(entry: ReferenceArtifact) -> BinaryIO:
    """Open verified raw logical bytes regardless of stored encoding."""
```

Enumerate fan-out submissions through stage plans/manifests and explicit run
IDs. Do not discover reference inputs through recursive latest/glob fallback.
Audit latest pointers separately.

### Protected inventory

Include:

- `00_grid`: manifest, unblind map, commands, grid/config snapshots;
- all four fan-out stages: stage manifest, tasks JSONL, execution records JSONL,
  and every per-run submission record;
- `03_collect`: summary, failures, collection report, cost tables,
  source-attempt records, task lineage;
- `04_select`: champions, selection report, task lineage;
- `05_final_grid`: source champions, final jobs, manifest, task lineage;
- `08_final_collect`: manifest and every compact table;
- `09_final_report`: final report JSON, Markdown, and every protected table;
- expected figure filenames and nonempty-size metadata;
- terminal counts, explicit source chain, pointer audit, environment, and
  profile summary in `reference.json`.

Exclude checkpoints, raw train/eval metrics, diagnostics, Slurm logs, latest
pointer bytes, and PNG byte equality.

### Deterministic encoding

Policy:

```text
raw table size <= 1_048_576 bytes -> raw
raw CSV/TSV size > 1_048_576 bytes -> gzip
gzip compresslevel = 9
gzip mtime = 0
gzip filename = empty
large non-tabular files -> raw unless separately reviewed
```

Each entry records logical role/path, stored path, media type, encoding, raw and
stored SHA-256, raw and stored size, and—when tabular—header, row count, and
declared column types.

Build under new sibling temporary directory, verify every entry, then atomically
rename. Reject symlinks, duplicate logical paths, source mutation, existing
destination, and partial output. Never overwrite reference ID.

Full raw lineage remains ignored external evidence. Optional Holystore copy is
useful but not required for routine comparison. Never delete source lineage.

## Work package 6 — layout map and comparator

Do not import or edit v3 `parity.py`. Its tolerance and structured comparison
are precedent; its hard-coded v2/v3 paths, incomplete inventory, broad string
replacement, and numeric-string coercion are unsuitable.

`reference/layout_maps/v1.json` classifies every protected artifact:

```text
schema_version
logical role
reference logical path
candidate logical path
format
approved token substitutions
volatile JSON pointers or CSV columns
float-tolerant JSON pointers or CSV columns
presence-only metadata where applicable
```

`compare.py` owns:

```python
@dataclass(frozen=True)
class Difference:
    """Describe one machine-readable mismatch in a mapped artifact."""

def load_layout_map(path: Path) -> Mapping[str, object]:
    """Validate complete one-to-one protected artifact mapping."""

def compare_reference(
    reference_dir: Path,
    candidate_root: Path,
    *,
    candidate_attempts: Mapping[str, str],
) -> tuple[Difference, ...]:
    """Compare verified frozen v3 inventory with one isolated v4 lineage."""

def write_comparison_report(
    destination: Path,
    differences: Sequence[Difference],
    *,
    provenance: Mapping[str, object],
) -> Path:
    """Write stable comparison inputs, counts, outcome, and differences."""
```

Comparison rules:

- verify reference before reading candidate;
- acceptance CLI rejects same resolved root/inode or same lineage;
- low-level comparison may self-compare only for unit sanity;
- candidate sentinel and v4 identity required;
- layout map must be one-to-one and root-safe;
- JSON key order may differ; key sets, structure, and values remain exact;
- JSONL and CSV row order remain exact;
- CSV header names and order remain exact;
- booleans, integers, counts, IDs, seeds, config/task strings, and unclassified
  values remain exact;
- float tolerance applies only at explicitly declared fields:
  `rel_tol=1e-9`, `abs_tol=1e-12`;
- handle NaN, infinity, and sign explicitly; never let parser coercion hide
  mismatch;
- volatile fields use exact JSON pointers/CSV columns, never substring patterns
  or wildcards;
- commands normalize only reviewed study/root/config/attempt tokens;
- missing and unexpected protected artifacts fail;
- `_v4/**` metadata is allowed only as declared additive surface; and
- large tables stream-decompress and compare rowwise.

If Markdown exposes floating formatting drift, inspect source and add narrow
parsed rules. Never introduce broad numeric regular-expression normalization.

## Work package 7 — tests

Use `uv run pytest`; never `--nosync`. Environment failure stops for interactive
resolution rather than prompting workaround or package installation.

### Routes, roots, and dispatch

- route schema/version and ten unique logical/physical roles;
- exact argv golden test for every role;
- missing, duplicate, unknown, or malformed attempt input;
- arbitrary/reserved override rejection;
- broad root, v3 root/child, traversal, and symlink escape;
- wrong/missing/mismatched sentinel;
- source missing/digest mismatch before launch;
- wrong fan-out interpreter and forbidden `--wait-job`;
- request/result receipts on success, nonzero exit, signal, and interrupted
  invocation;
- real legacy one-row plan writes only audit root and emits v4 identity;
- v3 source/results metadata unchanged;
- every role independently renderable from recorded upstream attempts;
- fresh-process v4 imports do not import v3 modules.

### Config and controller

- resolved config equality after approved identity/path substitutions;
- exact axis order/cardinality, seed rows, blind seed, and final replicate;
- shell syntax and usage path submit nothing;
- exact `gpu_test` resource flags, controller partition, waits, role order,
  count guards, and attempt propagation.

### Lineage and reference

- wrong grid digest or blind seed;
- 63/65 scan jobs, wrong final replicate, or wrong 64/64/8/8 fan-out counts;
- CPU, wrong partition, mixed profile, or resource drift;
- incomplete plan/status/checkpoint/results;
- mixed/dirty source or unexpected latest resolution;
- duplicate logical path, symlink, overwrite, and source mutation;
- repeated freeze produces identical bytes/descriptors;
- gzip threshold boundary, empty filename, `mtime=0`, raw/stored digests, sizes,
  header, and row count;
- truncation, wrong encoding, and corrupted descriptor rejected;
- atomic failure leaves no accepted reference.

### Comparator

- verified self-compare sanity;
- valid mapped fixture;
- same-root acceptance rejection;
- candidate v3 identity;
- changed/duplicate header, row reorder, seed, integer, metric, text, map entry;
- missing/extra protected artifact and root escape;
- exact JSON structure and nonnumeric strings;
- undeclared numeric strings remain exact;
- tolerance only at declared float fields;
- drift outside tolerance, NaN/infinity/sign behavior;
- exact volatile allowlist; and
- streamed raw/gzip equivalence.

### Regression and integration

- focused V4-0 suite;
- existing v3 study suite;
- existing toolkit suite;
- disposable ownership integration marker;
- real reference/candidate comparison behind explicit environment variables.

Use separate validator agent after implementation. Validator runs focused,
regression, disposable audit, and real E2E checks independently. Critic then
reviews invariants, comparison policy, and evidence before orchestrator accepts
or sends work back.

## Work package 8 — fresh v3 reference

Run only after implementation and focused tests exist at clean commit.

1. Verify clean checkout; record commit, branch, source/runtime receipts.
2. Choose fresh explicit isolated scratch staging root and stack ID. Do not use
   live v3 `results`.
3. Run current v3 `submit_stack.sh smoke`.
4. Require all four scientific fan-out stages on `gpu_test`; record controller
   CPU partition separately.
5. Preserve controller/fan-out dependency shape and all logs.
6. Require identical clean source/runtime closure across jobs.
7. Audit terminal ten-stage lineage and protected inventory.
8. Freeze once into tracked reference directory.
9. Verify reference without access to source staging root.
10. Commit reference payload separately.
11. Never rerun v3 for later V4-0/V4-1 acceptance.

Failed application/science run never becomes oracle and blocks V4-0. Failed
infrastructure submission remains preserved evidence; any fresh canonical rerun
requires recorded reason. Never create mixed-row oracle through backfill.

## Work package 9 — fresh v4 E2E and closeout

1. Start new isolated v4 root with sentinel.
2. Run v4 `submit_stack.sh smoke` from clean reference-bearing commit.
3. Verify 64 screening and eight final rows.
4. Verify every dispatch/controller/audit receipt.
5. Verify v3 source and live results unchanged.
6. Compare complete mapped inventory.
7. Require zero differences.
8. Write `comparison_report.json` with reference/candidate IDs, closure
   digests, Slurm evidence, counts, map digest, comparator version, and outcome.
9. Record exact evidence in roadmap and TODO.
10. Mark V4-0 **Observed / Verified gate** only after full acceptance.

Retain raw v3/v4 roots and Slurm logs. Report paths for later human custody; do
not remove data.

## Review and commit structure

One irreducible V4-0 PR against `dev`, split into reviewable commits:

1. `docs(experiments): approve V4-0 bootstrap plan`
2. `feat(experiments): add guarded V4-0 legacy dispatcher`
3. `test(experiments): add V4-0 ownership and parity harness`
4. `feat(experiments): add V4-0 smoke controller`
5. `test(experiments): freeze current v3 smoke reference`
6. `docs(experiments): record V4-0 E2E evidence`

Open or push only after later explicit instruction. Human merges. V4-1 starts
from updated `dev` on separate branch after V4-0 merges.

## Stop conditions

Stop and diagnose instead of weakening comparison when:

- config differs beyond approved identity/path substitutions;
- scientific task population, order, seed, metric, or command changes;
- mapped artifact disappears or changes meaning;
- comparison would require broad volatility/numeric normalization;
- environment would require dependency installation; or
- reference or candidate source is dirty/mixed.

Activate sealed snapshot only for structural bridge failure listed earlier.
Normal numerical mismatch is not snapshot trigger.

## Acceptance checklist

- [ ] Planning approval recorded and branch clean.
- [ ] V4 configs semantically equal after reviewed substitutions.
- [ ] Guarded root and source pin tests pass.
- [ ] Ten exact independently runnable routes pass focused tests.
- [ ] Disposable one-plan/one-train audit proves isolation and v4 identity.
- [ ] V3 source/live results remain unchanged.
- [ ] Fresh clean v3 `smoke.yaml`/`gpu_test` lineage completes.
- [ ] Reference audit passes and tracked inventory verifies independently.
- [ ] Comparator negative suite fails for every protected mismatch.
- [ ] Fresh v4 ten-stage `gpu_test` lineage completes.
- [ ] Complete mapped comparison reports zero differences.
- [ ] Separate validator confirms tests and E2E evidence.
- [ ] Critic finds no unresolved correctness or maintainability blocker.
- [ ] Roadmap/TODO record exact evidence and V4-0 verified status.
- [ ] No V4-1 contract, v3 edit, result deletion, dependency addition, push, or
      merge occurred outside explicit authorization.
