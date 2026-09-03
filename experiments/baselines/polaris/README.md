# Polaris launch helpers

## `multihost_launch.py`

Makes DeepQMC multi-host DDP work on Polaris. Run it as the command under `mpiexec`,
once per rank:

```sh
mpiexec -n 2 --ppn 1 --cpu-bind none \
  python experiments/baselines/polaris/multihost_launch.py \
  "$VENV/bin/deepqmc" hydra.run.dir="$D" ansatz=deeperwin task.seed=0 \
  task.electron_batch_size=4096 hamil/mol=He task.steps=200000
```

Proven by Polaris job `7587171`: two nodes, one rank each, four GPUs per rank, giving
`Running on 8 NVIDIA A100-SXM4-40GBs with 2 processes`, both ranks completing.

### Why a translation layer is needed at all

DeepQMC detects peers **only** from Slurm. `parallel.py maybe_init_multi_host()` reads
`SLURM_NTASKS`/`SLURM_PROCID` and calls `jax.distributed.initialize()` only when the
count exceeds one; its docstring says detection is *"only implemented for SLURM"*.
Polaris is PBS + PALS. Without translation every rank initialises alone — and it does
so **silently**: each process runs as an independent job, all succeed, and the elapsed
time looks like a wide measurement while being nothing of the sort.

### Why these five variables

jax 0.8.3's `SlurmCluster.is_env_present()` requires *all* of `SLURM_JOB_ID`,
`SLURM_STEP_NODELIST`, `SLURM_NTASKS`, `SLURM_PROCID`, `SLURM_LOCALID`. Miss one and no
cluster is detected — and because DeepQMC calls `initialize()` without a
`coordinator_address`, detection is the only thing that supplies it. jax derives:

| value | from |
|---|---|
| coordinator host | first entry of `SLURM_STEP_NODELIST` |
| coordinator port | `SLURM_JOB_ID % 4096 + 61440` — so the id must be an integer |

**The load-bearing detail:** jax parses that node list in *pure Python* and never
shells out to `scontrol`. That is the only reason a machine with no Slurm installed can
satisfy a Slurm-only code path.

Every rank must derive the same coordinator, so the node list is read from the same
`PBS_NODEFILE` and sorted identically everywhere.

### Trap: PALS pins each rank to one core

`mpiexec` under PALS delivers `cores=0` — a single core out of 64 — by default.
Measured on job `7588647`, DeepQMC He deeperwin, 4 GPUs, marginal s/step:

| launch | rate | affinity |
|---|---|---|
| direct | 0.1125 (bracketed repeat 0.1500) | unrestricted |
| `mpiexec`, PALS default | **0.2975** | `cores=0` of 64 |
| `mpiexec --cpu-bind none` | 0.1500 | `cores=0-63` |

A 2.0–2.6× slowdown, and **nothing fails** — the job completes, just far slower.
DeepQMC is host-bound for sampling and dispatch, so one core starves it.

Pass `--cpu-bind none` (or a real binding) **on the mpiexec command**. This wrapper runs
*inside* mpiexec and cannot set it for you.

It prints an `MHLAUNCH` line to stderr carrying rank, host, CPU affinity and visible
devices — and when it finds itself on a **single core** it says so explicitly, naming
the flag that fixes it. The trap then appears in the log at the moment it happens,
rather than being inferred later from a disappointing wall time.

The env mapping is a pure function, `slurm_env_from_pbs`, so it is unit-tested
directly rather than only through subprocess round-trips.

### Measurement caution

Repeated identical arms on Polaris have differed by 9–33% within a single allocation.
Any A/B comparison at this scale needs a repeated arm to bracket it, or drift will be
read as effect.

### Topology: GPU indices run REVERSE to NUMA ordering

ALCF's own affinity example does this, and the comment is the whole point:

```bash
# need to assign GPUs in reverse order due to topology
gpu=$(( num_gpus - 1 - PMI_LOCAL_RANK % num_gpus ))
```

Local rank 0 → GPU **3**, rank 1 → GPU 2, rank 2 → GPU 1, rank 3 → GPU 0. A naive
`rank → GPU rank` mapping places every rank on the *wrong* NUMA domain and maximises
cross-socket traffic. ALCF's recommended launch for one rank per GPU is
`--depth=8 --cpu-bind depth`, i.e. one rank per NUMA domain with 8 threads each.

**This wrapper is rank-per-NODE**, giving a single rank all four GPUs, so the reversed
mapping does not currently apply — the rank owns every device and every NUMA domain.
It becomes essential the moment anyone moves to rank-per-GPU, which the DeepQMC source
does support (each process would hold one local device). Anyone making that change must
apply the reversal, or they will silently get the worst possible placement while every
device count and banner still looks correct.

Note also that ALCF's `--cpu-bind depth` guidance is written for **one rank per GPU**.
It does not directly describe the right binding for one rank driving four GPUs, which is
what this wrapper does. `--cpu-bind none` is known to remove the 2.6× single-core
penalty; whether a locality-aware binding beats it for the rank-per-node case is
unmeasured, and should be measured rather than assumed.

Sources: ALCF *Using GPUs on Polaris* and the `argonne-lcf/GettingStarted`
`Examples/Polaris/affinity_gpu` example.
