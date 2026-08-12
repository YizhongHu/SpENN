# S4 Cannon receipt for the Polaris portability stack

Status: **complete for the Cannon analogue**. A bounded fresh-environment
retry passed after the earlier poisoned environment failed. This is an FASRC
Cannon same-shape analogue, not Polaris hardware validation. A separate
Polaris-only round is executing on `codex/polaris-pool-validation`.

## Clean verification target

- Source: detached clone of S3 commit `3f86568cfdd80faeb4ed4fd1530ff3364847e43e`.
- Clean verification checkout: `/n/netscratch/kozinsky_lab/Lab/rhu/tpen-s4-cannon-20260811`
  (recorded evidence only; the repository does not depend on this path).
- Final source SHA observed in the Slurm job: `3f86568cfdd80faeb4ed4fd1530ff3364847e43e`.
- CPU environment: fresh separate `UV_PROJECT_ENVIRONMENT`, provisioned inside
  Slurm with `uv sync --extra cpu`; no `--nosync`, `--no-extra`, or GPU extra
  was used.
- Python/pytest execution was guarded by non-empty `SLURM_JOB_ID`.

## Slurm method and oracle

The failed first final job was Slurm `38469895`, submitted with account
`kozinsky_lab` after a read-only live association/allocation check, partition
`test`, `--time=12:00:00`, `--cpus-per-task=4`, and `--mem=32G`. The job ran on
`holy8a24101.rc.fas.harvard.edu`; `sacct` reported partition `test`, state
`FAILED`, exit `2:0`, elapsed `00:00:30`. CPU tests ran in Slurm, never on the
login node. It is retained as failed evidence.

The bounded fresh-environment retry was Slurm job `38471117`, with the same
explicit account, partition, walltime, CPU, and memory request. It ran on
`holy8a24101.rc.fas.harvard.edu`; `sacct` reported `test|COMPLETED|0:0|00:06:20`.
Its fresh environment was
`/n/netscratch/kozinsky_lab/Lab/rhu/tpen-s4-cannon-cpu-env-retry-20260812`
(recorded evidence only; never a repository dependency).

Declared commands:

```text
PYTHONPATH=. uv run --extra cpu pytest -q experiments/toolkit
PYTHONPATH=. uv run --extra cpu pytest -q
```

Results from the retained stdout log:

- Toolkit slice: **93 passed in 8.72s**.
- Repository suite on retry: **942 passed, 3 skipped in 124.90s**.
- The earlier repository-suite result was **58 collection errors**, caused by
  an incomplete torch installation after an interrupted sync; it is retained
  as failed evidence, not a code-result claim.

Retained logs include the final job stdout/stderr and earlier failed attempts.
The final log location is recorded in the Task Orchestrator note and is not a
runtime dependency of this repository. The checkout itself remained clean.

## Corrections and exclusions

An initial `/n/home10` checkout violated the Cannon rule that active test/smoke
work belongs on Netscratch; it was not used as final evidence. A first job also
placed logs inside its checkout, making it dirty; that run was rejected as
final evidence. Those logs and failed jobs were retained. The corrected run
used Netscratch, explicit resources, an explicit live account, and Slurm.

The poisoned environment was not reused, repaired, or deleted. The one
authorized retry used a brand-new environment and one uninterrupted sync
inside Slurm; it passed. No data, environment, logs, or failed evidence was
deleted.

## Compliance limits

This receipt does not validate Polaris PBS queues, ALCF account
`HetRxnEnergy`, Eagle/HPSS storage, CUDA/A100 behavior, or a Polaris allocation
pool. It does not extrapolate Cannon CPU results to Polaris hardware. The
Polaris-only validation round is separate and is executing on
`codex/polaris-pool-validation`.

Canonical onboarding constraints applied: root `58348558` notes
`cluster-access-read-first`, `testing-and-env-standing-rules-2026-08-11`, and
`netscratch-never-referenced-from-repo-2026-08-11`; Clusters item `331fe74d`
notes `00-orientation`, `doe-project-account-2026-08-06`; portability parent
`8919c16b` note `portability-lane-handoff-accepted-2026-08-11`; Cannon item
`b5a3170d` note `00-read-first-cannon`; and canonical Cannon policy item
`3541fe33` notes `00-read-first-index`, `process-scope-2026-08-06`,
`operator-correction-2026-08-06`, `operator-resource-policy-2026-08-06`,
`login-node-boundary-2026-08-06`, and `official-fasrc-conventions-2026-08-06`.
