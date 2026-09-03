#!/bin/bash
# Present a PBS/PALS allocation to DeepQMC as if it were Slurm, so multi-host DDP fires.
#
# Run as the command under `mpiexec`, one invocation per rank:
#
#   mpiexec -n 2 --ppn 1 --cpu-bind none \
#       multihost_launch.sh deepqmc hydra.run.dir=... ansatz=... task.seed=...
#
# WHY THIS IS NEEDED
# DeepQMC only detects peers from Slurm. `parallel.py maybe_init_multi_host()` reads
# SLURM_NTASKS and SLURM_PROCID and calls jax.distributed.initialize() ONLY when the
# count exceeds 1; its docstring states detection is "only implemented for SLURM".
# Polaris runs PBS with the PALS launcher, so without this translation every rank
# initialises alone.
#
# WHY THESE FIVE VARIABLES
# jax 0.8.3's SlurmCluster.is_env_present() requires ALL of SLURM_JOB_ID,
# SLURM_STEP_NODELIST, SLURM_NTASKS, SLURM_PROCID and SLURM_LOCALID. Missing any one
# means no cluster is detected, and since DeepQMC calls initialize() WITHOUT a
# coordinator_address, detection is the only way the address gets filled in. From
# those it derives:
#     coordinator = first host in SLURM_STEP_NODELIST
#     port        = SLURM_JOB_ID % 4096 + 61440        (so JOB_ID must be an integer)
# It parses the nodelist in PURE PYTHON and never shells out to `scontrol`. That is
# the single fact that makes this possible on a machine with no Slurm installed.
#
# EVERY RANK MUST COMPUTE THE SAME COORDINATOR, so the node list is read from the
# same PBS_NODEFILE, sorted identically, on every rank.
#
# ---------------------------------------------------------------------------
# TRAP: PALS `mpiexec` PINS EACH RANK TO ONE CORE BY DEFAULT.
# Measured on Polaris job 7588647, 64-core node, DeepQMC He deeperwin, 4 GPUs:
#     direct                     0.1125 s/step   (bracketed repeat: 0.1500)
#     mpiexec, PALS default      0.2975 s/step   taskset reported cores=0 of 64
#     mpiexec --cpu-bind none    0.1500 s/step   taskset reported cores=0-63
# A 2.0-2.6x slowdown, and NOTHING FAILS -- the job completes, just far slower.
# DeepQMC is host-bound for sampling and dispatch, so one core starves it.
# PASS --cpu-bind none (or a real binding) ON THE MPIEXEC COMMAND. This wrapper runs
# INSIDE mpiexec and therefore cannot set it for you.
# ---------------------------------------------------------------------------
#
# Devices per rank come from MH_VISIBLE (default all four on a Polaris node).
# jax.distributed.initialize() asserts CUDA_VISIBLE_DEVICES is not None and parses it
# as the local device id list, so it must be set before DeepQMC starts.
#
# Proven by Polaris job 7587171: two nodes, one rank each, four GPUs per rank ->
# "Running on 8 NVIDIA A100-SXM4-40GBs with 2 processes", both ranks completing.
set -uo pipefail

: "${PBS_NODEFILE:?PBS_NODEFILE unset -- this must run inside a PBS allocation}"
: "${PBS_JOBID:?PBS_JOBID unset -- this must run inside a PBS allocation}"

# PBS job ids look like 7587171.polaris-pbs-01...; jax needs an integer for the port.
export SLURM_JOB_ID="${PBS_JOBID%%.*}"
export SLURM_STEP_NODELIST="$(sort -u "$PBS_NODEFILE" | paste -sd, -)"
export SLURM_NTASKS="${PMI_SIZE:-${PALS_NRANKS:-1}}"
export SLURM_PROCID="${PMI_RANK:-${PALS_RANKID:-0}}"
export SLURM_LOCALID="${PALS_LOCAL_RANKID:-0}"
export CUDA_VISIBLE_DEVICES="${MH_VISIBLE:-0,1,2,3}"

# Report affinity so the trap above is visible in the job log rather than inferred
# from a disappointing wall time.
echo "MHLAUNCH rank=${SLURM_PROCID}/${SLURM_NTASKS} host=$(hostname)" \
     "cores=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //')" \
     "nproc=$(nproc) vis=${CUDA_VISIBLE_DEVICES} jobid=${SLURM_JOB_ID}" >&2

exec "$@"
