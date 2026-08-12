# Polaris allocation-pool validation receipt

Date: 2026-08-11/12 EDT (cluster logs are UTC)

Frozen source: `codex/allocation-pool-executor` at `3f86568cfdd80faeb4ed4fd1530ff3364847e43e3`.
No executor behavior or production checkout was changed. The run used an additive detached
checkout at `/home/rhu/src/TPEN-polaris-pool-validation-3f86568` and the existing absolute
overlay Python `/home/rhu/.venvs/tpen-polaris-alcf-2025-09-25/bin/python`.

## Onboarding and constraints followed

The full mandatory chain was read before remote access, and this receipt records provenance:

- Clusters `331fe74d` notes `00-orientation` and `doe-project-account-2026-08-06`: follow
  Clusters -> provider -> host -> workflow order; Polaris is ALCF and its exact account is
  `HetRxnEnergy`; verify allocation and never infer storage paths from scheduler accounts.
- ALCF `e75460c1` notes `00-orientation` and `operator-project-account-2026-08-06`: Polaris
  uses PBS and `HetRxnEnergy`; resolve Eagle independently.
- Polaris `2379aa13` notes `00-read-first-polaris`, `usage-conventions`,
  `operator-project-account-2026-08-06`, and `provider-and-hpss-orientation-2026-08-07`:
  PBS requires explicit account, walltime, `filesystems=home:eagle`, and `-k doe`; compute
  is on PBS nodes only; login GPUs are unusable; use the stricter live constraint.
- TPEN root `58348558` notes `cluster-access-read-first` and
  `testing-and-env-standing-rules-2026-08-11`: full canonical notes are mandatory before
  SSH, storage, scheduler, tests, or jobs; tests run on a cluster; preserve run data and
  logs; do not introduce dependencies or untracked load-bearing infrastructure.
- Portability `8919c16b` note `portability-lane-handoff-accepted-2026-08-11`: preserve
  receipts and overlays; Polaris uses the absolute overlay Python and no per-worker sync;
  every receipt states checked and deliberately unclaimed properties.
- Polaris Stage-0 item `c91a994b` notes `stage0-mechanism-proof-2026-08-07` and
  `stage0-execution-receipt-2026-08-07`: `module use /soft/modulefiles` precedes conda
  loading; Polaris `uv` is invoked as `$PYBIN -m uv`; use `uv lock --check` then locked sync;
  omit invalid bare `--no-extra`; set GPU visibility before importing Torch; preserve old
  overlays and all logs.

Live preflight on `polaris-login-02`, 2026-08-12T02:06:42Z:

```text
sbank-list-allocations -r polaris -p HetRxnEnergy
allocation 15525 / suballocation 15496 / available 135,758.5 node-hours / charged 0.4
/eagle/HetRxnEnergy exists: drwxrws--- root HetRxnEnergy
```

The submitted jobs used exactly `-A HetRxnEnergy -q debug -k doe` and
`-l select=1:system=polaris,walltime=00:20:00,filesystems=home:eagle,place=scatter`.
The initial submission without explicit `-q debug` was rejected by PBS before execution
(`Job violates queue and/or server resource limits`); no validation claim is based on it.

## Stage 1: one node, four rows, four GPUs

PBS job `7425428.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` finished with exit status 0,
queue `debug`, `exec_host=x3205c0s37b1n0/0*64`, requested one Polaris node and 20 minutes.
The frozen executor test module ran first: `11 passed in 8.39s`.

The allocation pool ran four planner-shaped rows with four workers and
`CUDA_VISIBLE_DEVICES` values `3,2,1,0` (Polaris reverse-order binding). All four records
completed successfully. Each worker observed exactly one visible device and an A100-SXM4-40GB:

```text
stage1-row-0 -> CUDA_VISIBLE_DEVICES=3, device_count=1, cc=8.0
stage1-row-1 -> CUDA_VISIBLE_DEVICES=2, device_count=1, cc=8.0
stage1-row-2 -> CUDA_VISIBLE_DEVICES=1, device_count=1, cc=8.0
stage1-row-3 -> CUDA_VISIBLE_DEVICES=0, device_count=1, cc=8.0
```

There is one `attempt1` directory and one immutable claim directory per row. The executor
records contain the PBS allocation ID, worker index, visibility variable/value, command,
claim path, attempt path, and launcher status path.

Stage-1 evidence root:
`/eagle/HetRxnEnergy/rhu/runs/polaris-pool-validation-20260812/stage1/`

I self-reviewed the receipts before Stage 2. They demonstrate the stated row count, one
attempt per row, direct overlay Python, four distinct bindings, successful Torch import on
compute, and preserved command/claim/status provenance. They do not claim multi-node scaling,
DDP behavior, throughput, or cross-facility equivalence.

## Stage 2: eight rows, refill, claims, resume, skip, deadline

PBS job `7425466.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` finished with exit status 0,
queue `debug`, `exec_host=x3106c0s13b1n0/0*64`, same one-node/20-minute resource shape.

Evidence root:
`/eagle/HetRxnEnergy/rhu/runs/polaris-pool-validation-20260812/stage2-rerun/`

- Dynamic refill: first pass `stage2-pass-a` produced 8 records for 8 uneven rows. Trace
  files show 2 events for each successful row and 4 events for the deliberately failing
  `stage2-row-1`, demonstrating its later retry rather than duplicate concurrent execution.
- Atomic claims: first pass has exactly 8 claim receipts, one per row. No duplicate claim
  directory exists within a pass.
- Failure/resume: `stage2-row-1` intentionally returned code 3 in pass A; fresh pass
  `stage2-pass-b` returned exactly that row. It has `attempt1` and `attempt2`; all other
  rows have one attempt.
- Completed-row skipping: pass B returned 1 record, and its claim set contains only the
  failed row; the seven completed rows were skipped.
- Deadline guard: `stage2-deadline` admitted exactly four rows, whose two-second commands
  completed; rows 4-7 have no claim or attempt. This verifies that the guard stops new
  claims without killing already-running commands.
- Combined Stage-2 rerun evidence has 13 claim receipts and 13 attempt status files: 8
  first-pass rows, 1 retry, and 4 deadline-guard rows.

The first Stage-2 PBS attempt (`7425449`) exposed only a validation-harness mistake: the
deadline invocation supplied `deadline_guard_min` twice and exited before the deadline phase.
Its partial artifacts and PBS logs remain at
`/eagle/HetRxnEnergy/rhu/runs/polaris-pool-validation-20260812/stage2/`; no result from that
attempt is presented as validation evidence. The corrected rerun above is the Stage-2 result.

## Deliberately not claimed

This is Polaris-only evidence. It does not claim Cannon/S4 evidence, multi-node scaling,
DDP correctness, queue throughput, scheduler submission by the executor (the executor is
allocation-local), cross-facility bit identity, or removal of any prior overlay/run/PBS log.
No run data, overlay, checkout, or PBS log was deleted.

