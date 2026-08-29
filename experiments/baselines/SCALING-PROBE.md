# Multi-GPU scaling probe

`scaling_probe.py` measures one model process over one requested GPU arm. It
does not launch one independent process per GPU: that would measure task
parallelism, not data-parallel latency.

Run every arm with the same total batch size (normally 4096). The command after
`--` is an argument vector owned by the selected backend. The probe records a
microsecond UTC timestamp as each backend log line arrives, validates the
backend-reported device count, reads the backend-reported per-device batch,
and writes one JSON result. A failed device, batch, energy, or timing evidence
check remains a JSON result with `status: failed`; it is never converted into a
speed number.

The device regular expression must expose `devices`, the batch expression
must expose `batch`, the energy expression must expose `energy` and `error`,
and the step expression must expose `step`. This makes the probe backend- and
ansatz-agnostic without importing an external code on a login node.

For FermiNet and Psiformer, use `run_ferminet_scaling_arm.py` as the command
inside the probe. It observes the exact per-device batch argument passed into
FermiNet's MCMC construction call and emits a blocking-error training-tail
energy after the run. It instruments the public call boundary at runtime; it
does not modify FermiNet. Its progress matcher must be
`SCALING_PROBE Step (?P<step>[0-9]+)`: FermiNet's own INFO progress and the
wrapper marker are both valid logs, but combining them fabricates sub-step
intervals. If a completed raw result used a broader matcher, use
`scaling_probe.py reanalyse` to write a new result from the immutable wrapper
log; retain the original result unchanged.

For DeepQMC, use `run_deepqmc_scaling_arm.py`.  It invokes DeepQMC's public
Hydra CLI unchanged, while runtime-instrumenting its sampler-state construction
and HDF5 logger.  Thus `Running on N ...` remains DeepQMC's own process-level
device evidence, `SCALING_PROBE device_batch_size` comes from the actual total
electron batch passed to the sharding call, and `SCALING_PROBE Step` is emitted
after each HDF5 optimizer-step write.  Pass an explicit non-empty Hydra
override list after `--`; set `task.electron_batch_size=4096` for every arm and
set `CUDA_VISIBLE_DEVICES` to a list (`0`, `0,1`, or `0,1,2,3`) before Python
imports DeepQMC.  DeepQMC requires that visibility variable for multi-host
initialization and validates total-batch divisibility itself.  Use a workload-
appropriate tail cut: its known slow ramp extends beyond step 9000, so a
1200-step FermiNet probe length is not suitable for DeepQMC.  Set
`hydra.run.dir=<arm-run-dir>` because DeepQMC's application assigns
`task.workdir` from Hydra's runtime directory.

After a 1-GPU arm and *before* advancing each higher-GPU rung, run
`scaling_probe.py gate` over that baseline and candidate JSON. It writes the
same structured comparison as `summarize` but exits non-zero unless each arm
has valid process evidence and their energies agree within combined statistical
error. A scheduler script should use that exit status to stop the ladder at a
wrong arm. After the completed ladder, `scaling_probe.py summarize` writes the
full table and reports warm-up-cut-100 speed efficiency `t1 / (N * tN)` only
for correctness-passing arms. It also retains independent 100 through 400
warm-up-cut fits plus consecutive-interval min, max, and median. Do not average
rates across systems or GPU counts.

On Polaris, schedule arms sequentially on one exclusive node, and include at
least one reverse-order pair after the forward 1 -> 2 -> 4 ladder. Store the
probe's wrapper logs and JSON files alongside the scheduler output; use those
log timestamps rather than Eagle mtimes.
