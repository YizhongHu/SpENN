# S4 Cannon receipt for the Polaris portability stack

Status: **partial / blocked by Cannon CPU-environment provisioning**. This is
an FASRC Cannon same-shape analogue, not Polaris hardware validation. The
separate Polaris-only round remains outstanding and requires human approval.

## Clean verification target

- Source: detached clone of S3 commit `3f86568cfdd80faeb4ed4fd1530ff3364847e43e`.
- Clean verification checkout: `/n/netscratch/kozinsky_lab/Lab/rhu/tpen-s4-cannon-20260811`
  (recorded evidence only; the repository does not depend on this path).
- Final source SHA observed in the Slurm job: `3f86568cfdd80faeb4ed4fd1530ff3364847e43e`.
- CPU environment: separate `UV_PROJECT_ENVIRONMENT`, provisioned with
  `uv sync --extra cpu`; no `--nosync`, `--no-extra`, or GPU extra was used.
- Python/pytest execution was guarded by non-empty `SLURM_JOB_ID`.

## Slurm method and oracle

The final retained job was Slurm `38469895`, submitted with account
`kozinsky_lab` after a read-only live association/allocation check, partition
`test`, `--time=12:00:00`, `--cpus-per-task=4`, and `--mem=32G`. The job ran on
`holy8a24101.rc.fas.harvard.edu`; `sacct` reported partition `test`, state
`FAILED`, exit `2:0`, elapsed `00:00:30`. CPU tests ran in Slurm, never on the
login node.

Declared commands:

```text
PYTHONPATH=. uv run --extra cpu pytest -q experiments/toolkit
PYTHONPATH=. uv run --extra cpu pytest -q
```

Results from the retained stdout log:

- Toolkit slice: **93 passed in 8.72s**.
- Repository suite: **58 collection errors**, interrupted before execution;
  TPEN reported an incomplete torch installation (`torch` had no
  `__version__`). This is an environment failure, not a claimed code result.

Retained logs include the final job stdout/stderr and earlier failed attempts.
The final log location is recorded in the Task Orchestrator note and is not a
runtime dependency of this repository. The checkout itself remained clean.

## Corrections and exclusions

An initial `/n/home10` checkout violated the Cannon rule that active test/smoke
work belongs on Netscratch; it was not used as final evidence. A first job also
placed logs inside its checkout, making it dirty; that run was rejected as
final evidence. Those logs and failed jobs were retained. The corrected run
used Netscratch, explicit resources, an explicit live account, and Slurm.

The CPU environment was then repaired only with the declared CPU extra and a
torch reinstall. Installation remained incomplete/hung, so the full suite was
not rerun and no green repository-suite claim is made. No data, environment,
logs, or failed evidence was deleted.

## Compliance limits

This receipt does not validate Polaris PBS queues, ALCF account
`HetRxnEnergy`, Eagle/HPSS storage, CUDA/A100 behavior, or a Polaris allocation
pool. It does not extrapolate Cannon CPU results to Polaris hardware. The
Polaris-only validation round must separately use its canonical PBS/account,
environment, storage, and scheduler notes, with explicit human approval.

Canonical onboarding constraints applied: root `58348558` notes
`cluster-access-read-first`, `testing-and-env-standing-rules-2026-08-11`, and
`netscratch-never-referenced-from-repo-2026-08-11`; Clusters item `331fe74d`
notes `00-orientation`, `doe-project-account-2026-08-06`; portability parent
`8919c16b` note `portability-lane-handoff-accepted-2026-08-11`; Cannon item
`b5a3170d` note `00-read-first-cannon`; and canonical Cannon policy item
`3541fe33` notes `00-read-first-index`, `process-scope-2026-08-06`,
`operator-correction-2026-08-06`, `operator-resource-policy-2026-08-06`,
`login-node-boundary-2026-08-06`, and `official-fasrc-conventions-2026-08-06`.
