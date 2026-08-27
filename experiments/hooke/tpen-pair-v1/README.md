> **STALE as of 2026-08-26.** This material is retained for provenance of already-archived results. These studies run on the *legacy* experiment-toolkit surface and are **not a template for new work**; new work uses the dispatch seam in `experiments/toolkit/dispatch.py`. These studies were deliberately **not migrated to the new seam**—this is an operator ruling of 2026-08-26, not an oversight or a pending task.

# TPEN pair-v1 allocation-local smoke

`launch.py` is the one tracked launcher for this smoke. It runs only after a
facility scheduler has created an allocation and the operator has provisioned
an environment. It never calls `sbatch`, `qsub`, `srun`, `uv sync`, or any
other provisioning step. Facility selection is expressed by the envelope's
environment and command-line values; the Python launcher has no facility
branches.

The results root must be outside the checkout. Use an operator-owned run path,
not a repository output directory. Account strings, queues, partitions,
walltimes, overlay locations, and run roots are operator-supplied and are not
committed defaults. In particular, use account `MAT281` on Frontier and
`HetRxnEnergy` on Polaris/Aurora only when that account is assigned to your
allocation.

## Common launcher shape

Run this from the repository checkout inside the allocation. `--python` is the
already-provisioned facility or overlay interpreter. This smoke has one task
and one allocation worker; pass exactly one `--visibility-values` entry so the
device binding is deterministic.

```bash
python experiments/hooke/tpen-pair-v1/launch.py \
  --python /path/to/provisioned/bin/python \
  --results-root /path/to/runs \
  --run-id pair-v1-smoke-YYYYMMDD \
  --device cuda \
  --visibility-variable CUDA_VISIBLE_DEVICES \
  --visibility-values 0 \
  --allocation-id <scheduler-allocation-id> \
  --deadline-env-var SLURM_JOB_END_TIME \
  --pass-id smoke-pass-1
```

Use `--device xpu` and `--visibility-variable ZE_AFFINITY_MASK` for Aurora.
Use `--dry-run` to print the exact `run.py` argv and overrides without
starting a task. A later retry uses a new `--pass-id`.

## FASRC Cannon — Slurm + CUDA

Provision once in the operator's chosen environment using the Cannon CUDA
extra, then enter a GPU allocation. Keep account, partition, and run paths
outside the repository and fill them in at submission time:

```bash
sbatch -A <cannon-account> -p <gpu-partition> --gres=gpu:1 \
  --cpus-per-task=4 --time=00:30:00 --wrap='
cd <checkout>
export UV_PROJECT_ENVIRONMENT=<operator-cuda-overlay>
"$UV_PROJECT_ENVIRONMENT/bin/python" experiments/hooke/tpen-pair-v1/launch.py \
  --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  --results-root <operator-run-root> --run-id <run-id> --device cuda \
  --visibility-variable CUDA_VISIBLE_DEVICES --visibility-values 0 \
  --allocation-id "$SLURM_JOB_ID" \
  --deadline-env-var SLURM_JOB_END_TIME
'
```

The envelope owns provisioning and the allocation request; the launcher owns
only the local task. Select the CUDA extra appropriate to the provisioned
stack rather than copying this example's `cu126` blindly.

## ALCF Polaris — PBS + direct overlay Python + CUDA

Provision the system-site-packages overlay on a login node as documented in
the root README, then submit an allocation with an operator-supplied account,
queue, and Eagle run root. For this single-task smoke, choose one allocated
GPU explicitly:

```bash
qsub -A <polaris-account> -q <queue> -l \
  select=1:system=polaris,ngpus=4,filesystems=home:eagle,walltime=00:30:00 \
  -v TPEN_CHECKOUT=<checkout>,TPEN_PYTHON=<overlay>/bin/python,TPEN_RUNS=<eagle-run-root> \
  <<'PBS'
#!/bin/bash
cd "$TPEN_CHECKOUT"
"$TPEN_PYTHON" experiments/hooke/tpen-pair-v1/launch.py \
  --python "$TPEN_PYTHON" --results-root "$TPEN_RUNS" \
  --run-id <run-id> --device cuda \
  --visibility-variable CUDA_VISIBLE_DEVICES \
  --visibility-values 3 \
  --allocation-id "$PBS_JOBID" \
  --deadline-env-var PBS_WALLTIME
PBS
```

Do not select a TPEN CUDA extra on Polaris: the facility overlay supplies
PyTorch/CUDA. Bind `CUDA_VISIBLE_DEVICES` before Python imports Torch, as the
launcher does per worker.

## ALCF Aurora — PBS + facility overlay + Intel XPU

Provision the facility Aurora overlay on a login node. The overlay's Python
and any required Intel runtime modules must be loaded by the envelope. Supply
the project account and storage path at submission time:

```bash
qsub -A <aurora-account> -q <queue> -l \
  select=1:system=aurora,ngpus=4,filesystems=home:eagle,walltime=00:30:00 \
  -v TPEN_CHECKOUT=<checkout>,TPEN_PYTHON=<facility-overlay>/bin/python,TPEN_RUNS=<eagle-run-root> \
  <<'PBS'
#!/bin/bash
module load <aurora-facility-modules>
cd "$TPEN_CHECKOUT"
"$TPEN_PYTHON" experiments/hooke/tpen-pair-v1/launch.py \
  --python "$TPEN_PYTHON" --results-root "$TPEN_RUNS" \
  --run-id <run-id> --device xpu \
  --visibility-variable ZE_AFFINITY_MASK \
  --visibility-values 0 \
  --allocation-id "$PBS_JOBID" \
  --deadline-env-var PBS_WALLTIME
PBS
```

The `ZE_AFFINITY_MASK` value is an allocation-local binding. Confirm the
facility's supported mask syntax before running a real job; the launcher
passes the one value to its one worker unchanged.

## OLCF Frontier — Slurm + ROCm overlay

Provision the ROCm overlay with the system Python and the `rocm71` extra as
described in the root README, then request the operator's MAT281 allocation.
ROCm Torch exposes the CUDA Torch API, so keep `--device cuda` and use the
CUDA visibility variable:

```bash
sbatch -A <frontier-account> -p <frontier-partition> \
  --gpus-per-node=1 --time=00:30:00 --wrap='
cd <checkout>
<rocm-overlay>/bin/python experiments/hooke/tpen-pair-v1/launch.py \
  --python <rocm-overlay>/bin/python --results-root <operator-run-root> \
  --run-id <run-id> --device cuda \
  --visibility-variable CUDA_VISIBLE_DEVICES --visibility-values 0 \
  --allocation-id "$SLURM_JOB_ID" \
  --deadline-env-var SLURM_JOB_END_TIME
'
```

Do not change the runtime device to `rocm`; the TPEN runtime uses Torch's
CUDA-compatible API for ROCm.
