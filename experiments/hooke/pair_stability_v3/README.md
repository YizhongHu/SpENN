# Hooke pair-stability V3 study

This is the current runnable pair-stability study and the behavioral baseline
for future restructuring work (which continues in a v4 study, compared back
against this one). It descends from `experiments/hooke/pair_stability_v2`
(retired as a comparison target after end-to-end parity was confirmed) with
the same stage layout, scan axes, validation/final-evaluation task suites,
plots, reports, and metric names, plus profiling instrumentation
(`train/perf` phase timing, `eval/perf` component timing, `runtime` peak
memory) and compact cost tables (`cost_by_run.csv`, `cost_by_axis.csv`,
`cost_by_task.csv`).

The default grid is:

```text
major_grid: basis x update_normalization x feature_normalization
minor_grid: lr x channels x activation
scan_seed_rows: paired train-model/train-sampler/validation-sampler seeds
champions: energy selector only
final_replicates: 9
```

The checked-in scan has:

```text
2 bases x 3 update choices x 4 feature choices x 2 learning rates x 2 channel counts x 4 activations x 3 seed rows = 1152 scan jobs
24 major points x 1 energy representative = 24 selected champions
24 selected champions x 9 final seeds = 216 final jobs by default
```

`configs/pilot.yaml` is the full-workflow budget pilot:

```text
major_grid: max_steps x sampler_n_steps
minor_grid: basis x mechanism x lr x channels
mechanism: baseline, feature_gaussian_norm, update_gaussian_norm
activation: fixed to Tanh
channels: fixed to 8
3 max_steps x 2 sampler_n_steps x 2 bases x 3 mechanisms x 2 learning rates x 1 channel x 3 seed rows = 216 scan jobs
6 major points x 1 energy representative = 6 selected champions
6 selected champions x 9 final seeds = 54 final jobs by default
```

Channel count is fixed at 8 in the pilot. The v2 full-result lineage under
`SpENN/experiments/hooke/pair_stability_v2/results` did not scan channel 16;
all collect, select, and final rows used channel 8, so there is no result-backed
reason to keep channel count as a pilot axis.

`configs/pilot_smoke.yaml` is the matching smoke grid:

```text
2 max_steps x 2 sampler_n_steps x 2 bases x 3 mechanisms x 1 learning rate x 1 channel x 1 seed row = 24 scan jobs
4 major points x 1 energy representative = 4 selected champions
4 selected champions x 1 final seed = 4 final jobs
```

## Stages

The stage layout remains:

```text
00_grid -> 01_train -> 02_validation -> 03_collect -> 04_select
        -> 05_final_grid -> 06_final_train -> 07_final_eval
        -> 08_final_collect -> 09_final_report
```

`final_plan.py` writes nine final seeds per selected champion by default. Pass
`--replicates` only to override that configured count.

## Config Semantics

The training and validation configs expose slot-like run parameters:

```yaml
run_parameters:
  basis_slot: B00
  update_normalization_slot: U00
  feature_normalization_slot: F00
  activation_slot: SiLU
  lr: 1.0e-3
  channels: 8
  seed: 0
```

`choices.basis` owns the concrete input basis specs.
`choices.update_normalization`, `choices.feature_normalization`, and
`choices.activation` own the explicit layer controls:

```text
update_normalization
update_envelope
feature_normalization
feature_envelope
irrep_activation
```

When both a normalization and an envelope are configured at the same site, the
normalization runs first and the real-state envelope runs second. Embedding
controls are owned by `nn.Embedding`; update and feature controls are owned by
`nn.SpENNLayer`.

The output Hooke Gaussian envelope and electron-electron cusp are common across
all variants. The Hooke singlet's fixed opposite-spin cusp range is
`model_params.cusp_opposite_range_parameter: 0.25`; it is deliberately outside
the scan axes but may be overridden explicitly for a named ablation.

## Blinding

`plan.py` blinds the major axes by default. It shuffles each major axis
independently into slots and writes routine manifests with slot values:

```text
b-B00_m-A03_lr-1e-3_ch-4_seed-0
```

The semantic mapping is written only to:

```text
results/00_grid/<attempt_id>/unblind.json
```

Use `--blind-seed <int>` for reproducibility. Use `--no-blind` only for
debugging when semantic labels are intentionally desired in routine artifacts.

## Attempt Defaults

Stage scripts generate attempt ids by default. The id is a timestamp in
`America/New_York`:

```text
YYYYMMDDTHHMMSS-0400
```

`plan.py` writes `00_grid/latest.json`. `train.py` and `validate.py` default to
that latest grid attempt. Later stages default to the latest previous-stage
artifacts and trace provenance back to the source grid:

```text
collect.py            latest validation attempt per scan run -> source 00_grid
select_champions.py   latest 03_collect
final_plan.py         latest 04_select
final_train.py        latest 05_final_grid
final_eval.py         latest 05_final_grid + latest ready 06_final_train per final run
final_collect.py      latest 07_final_eval per final run
final_report.py       latest 08_final_collect
```

Fan-out stages also write per-run latest pointers:
`01_train/{run_id}/latest.json`, `02_validation/{run_id}/latest.json`,
`06_final_train/{final_run_id}/latest.json`, and
`07_final_eval/{final_run_id}/latest.json`.

Attempt ids are names only. Full and smoke runs are both normal attempts; the
planned grid file (`grid.yaml` or `smoke.yaml`) is recorded in the grid manifest
and determines the run scale.

Pass explicit `--attempt-id`, `--grid-attempt-id`, or previous-stage attempt
flags only when reproducing an older lineage or debugging.

## Device Selector

Stage launchers default to CPU for safety. Use `--device cpu`, `--device cuda`,
or `--device cpu,cuda` to choose the execution target. The selector switches the
uv environment, uv extra, `runtime.device` override, and Submitit resources
together:

| selector | uv environment | uv extra | runtime override | Submitit hardware default |
|----------|----------------|----------|------------------|---------------------------|
| `--device cpu` | `.venv` | `cpu` | `runtime.device=cpu` | `slurm_partition=sapphire,kozinsky,seas_compute`, `cpus_per_task=4`, `mem_per_cpu=8G`, no GPUs |
| `--device cuda` | `.venv-gpu` | `cu126` | `runtime.device=cuda` | `slurm_partition=seas_gpu,kozinsky_gpu`, `cpus_per_task=4`, `mem_per_cpu=8G`, `gpus_per_node=1` |
| `--device cpu,cuda` | both of the above | both | per claimed row | submits separate CPU and CUDA candidate arrays; the first candidate that starts claims each row |

Submitit launchers re-exec through `.venv-submitit` before creating arrays, so
the Submitit supervisor does not share the CPU worker's `.venv` while workers
run `uv sync`.
For manual Submitit launches, prefer prefixing the command with
`UV_PROJECT_ENVIRONMENT=.venv-submitit` so uv starts in the launcher environment
immediately.
CPU workers export `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` from the Slurm CPU
allocation so PyTorch and BLAS use the requested CPU allocation.

Submitit stages request memory with Slurm `--mem-per-cpu`, not `--mem`, to avoid
conflicting cluster-level memory environment variables. The default is 8G per
requested CPU; override it with `--slurm-mem-per-cpu-gb` only when a partition
requires a different per-CPU request.

Mixed `cpu,cuda` mode uses separate Submitit submissions because GPU resources
cannot be requested on CPU partitions. Use `--slurm-cpu-partition` and
`--slurm-cuda-partition` to pin the two candidates separately, such as `test`
and `gpu_test` for smoke sanity checks. Use `--slurm-cpu-timeout-min` and
`--slurm-cuda-timeout-min` when CPU and CUDA candidates need different walltime
limits.

## Runbook

Full and smoke runs use the same stage stack and should keep launcher flags,
partitions, resources, and dependencies as close to the real run as possible.
The normal difference is the grid file. Use different partitions or chunk sizes
only for an explicitly requested test-partition sanity check. `--smoke` is
retired; use `configs/smoke.yaml` for the smaller full-workflow smoke run.

Set the study path once:

```bash
STUDY=experiments/hooke/pair_stability_v3
```

### Pilot Scan Stages

Use `configs/pilot.yaml` for the full-workflow pilot where report axes are
`max_steps` and `sampler_n_steps`. Do not pass `--blind`; these numeric axes
are meant to remain visible.

```bash
uv run python $STUDY/plan.py \
  --grid $STUDY/configs/pilot.yaml \
  --no-blind
```

After planning the pilot grid, run the same scan, final, collect, and report
commands from the non-pilot full or smoke sections, skipping their `plan.py`
command. Later stages default to the latest planned grid lineage.

Use `configs/pilot_smoke.yaml` for the smaller pilot smoke lineage:

```bash
uv run python $STUDY/plan.py \
  --grid $STUDY/configs/pilot_smoke.yaml \
  --no-blind
```

After planning `pilot_smoke.yaml`, run the same commands as the pilot/full
lineage. Do not switch to the non-pilot smoke/test-partition launch stack unless
that is the explicit test objective.

### Non-Pilot Full Run

Plan the full grid from `configs/grid.yaml`. The attempt id is generated automatically in
`America/New_York` and recorded in `results/00_grid/latest.json`.

```bash
uv run python $STUDY/plan.py \
  --grid $STUDY/configs/grid.yaml \
  --blind \
  --blind-seed 811
```

Train and validate the latest full grid on production GPU partitions:

```bash
uv run --extra submitit python $STUDY/train.py \
  --backend submitit --device cuda \
  --chunk-size 18 \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-timeout-min 480
```

Validate completed train attempts from the latest grid. If the train launcher
job id is known, `--wait-job` submits a dependent launcher and exits; otherwise,
run the same command after train checkpoints are ready.

```bash
uv run --extra submitit python $STUDY/validate.py \
  --backend submitit --device cuda \
  --chunk-size 32 \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-timeout-min 120 \
  --wait-job <train_launcher_job_id>
```

Collect the latest validation lineage and select energy representatives.

```bash
uv run python $STUDY/collect.py

uv run python $STUDY/select_champions.py
```

### Full Final Stages

The checked-in grid sets `final_replicates: 9`, so the default final plan
continues selected champions through nine independent final seeds.

Plan final replicates from the latest champion selection:

```bash
uv run python $STUDY/final_plan.py
```

This writes `results/05_final_grid/<attempt_id>/final_jobs.csv` and
`05_final_grid/latest.json`. Each final job records the selected major-axis
aliases, frozen minor choices, source champion row, replicate index, and
independent final seeds:

```text
final_train_model_seed    = 100 + replicate_index
final_train_sampler_seed  = 1000 + replicate_index
final_eval_sampler_seed   = 10000 + replicate_index
```

Launch final training from the latest final grid:

```bash
uv run --extra submitit python $STUDY/final_train.py \
  --backend submitit --device cuda \
  --chunk-size 18 \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-timeout-min 480
```

The final-train launcher excludes rows whose selected attempt already completed
with a checkpoint, regardless of whether CPU or CUDA ran them. Incomplete rows
with complete checkpoint directories resume from the highest complete checkpoint
with `load.mode=train_resume`; fresh rows start from step zero. If final
training has already been submitted, do not rerun it just to continue the
lineage. Use the final-train launcher job id with `final_eval.py --wait-job` so
evaluation starts after Slurm marks the launcher complete.

Local final-train claimers stop taking new rows near the end of their enclosing
allocation. When `SLURM_JOB_END_TIME` is present, local claim mode uses it as the
deadline and stops claiming rows 60 minutes before that deadline by default. Use
`--local-deadline <unix-or-iso-time>` when running outside a Slurm allocation, or
adjust the buffer with `--local-deadline-guard-min`; set it to `0` only when a
local worker is allowed to be killed mid-row.

Launch final evaluation from the latest final grid and the latest ready
final-train checkpoint for each final run:

```bash
uv run --extra submitit python $STUDY/final_eval.py \
  --backend submitit --device cuda \
  --chunk-size 32 \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-timeout-min 120 \
  --wait-job <final_train_launcher_job_id>
```

Keep final-eval `--chunk-size` at the default `1` unless you intentionally want
multiple report-grade final-eval rows serialized inside one Slurm allocation.
`final_eval.py` records the exact final-train attempt and checkpoint directory
that it evaluates.

Collect compact final tables and render the report:

```bash
uv run python $STUDY/final_collect.py

uv run python $STUDY/final_report.py
```

`final_collect.py` reads raw final train/eval artifacts once and writes compact
CSV summaries under `08_final_collect/{attempt_id}/`. `final_report.py` reads
only those compact tables and writes `09_final_report/{attempt_id}/report.md`,
`tables/*.csv`, and `figures/*.png`.

### Non-Pilot Smoke Run

Plan the smoke grid. It has the same axes as the full grid, reduced to 64 scan
jobs and one paired seed row. It does not reduce validation or final-eval
parameters.

```bash
uv run python $STUDY/plan.py \
  --grid $STUDY/configs/smoke.yaml \
  --blind \
  --blind-seed 811
```

Train and validate the latest smoke grid on test partitions. Larger chunk sizes
reduce submitted Slurm array size.

```bash
uv run --extra submitit python $STUDY/train.py \
  --backend submitit --device cuda \
  --chunk-size 32 \
  --slurm-partition gpu_test \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-timeout-min 60

uv run --extra submitit python $STUDY/validate.py \
  --backend submitit --device cuda \
  --chunk-size 32 \
  --slurm-partition gpu_test \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-timeout-min 120 \
  --wait-job <train_launcher_job_id>

uv run python $STUDY/collect.py

uv run python $STUDY/select_champions.py
```

### Non-Pilot Smoke Final Stages

The smoke grid sets `final_replicates: 1`, so final planning continues each
smoke champion through one final seed.

```bash
uv run python $STUDY/final_plan.py

uv run --extra submitit python $STUDY/final_train.py \
  --backend submitit --device cuda \
  --chunk-size 8 \
  --slurm-cuda-partition gpu_test \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-cuda-timeout-min 60

uv run --extra submitit python $STUDY/final_eval.py \
  --backend submitit --device cuda \
  --chunk-size 8 \
  --slurm-partition gpu_test \
  --slurm-mem-per-cpu-gb 8 \
  --slurm-timeout-min 120 \
  --wait-job <final_train_launcher_job_id>

uv run python $STUDY/final_collect.py

uv run python $STUDY/final_report.py
```

`validate.py` and `final_eval.py` support `--wait-job <job_id>` when the
upstream Submitit launcher job id is known. They submit a lightweight Slurm
launcher with `--dependency=afterany:<job_id>` and exit immediately; the
dependent launcher reruns the same stage command without `--wait-job` and then
performs the normal readiness checks. Otherwise, rerun validation/final eval
after upstream checkpoints are ready; these stages always skip rows that are not
ready. The lightweight launcher defaults to the `test` partition; override it
with `--wait-launcher-partition` if needed. The real validation/final-eval array
uses the Slurm partition flags from the command that the dependent launcher
reruns.

### Run-And-Forget Stack

`submit_stack.sh` self-submits one CPU controller job that runs the complete
stage stack in order. The controller submits each GPU stage, waits for every
recorded Slurm job to finish, verifies row counts, and then advances to the
next stage.

```bash
$STUDY/submit_stack.sh full
$STUDY/submit_stack.sh smoke
$STUDY/submit_stack.sh pilot
$STUDY/submit_stack.sh pilot-smoke
```

`full`, `smoke`, `pilot`, and `pilot-smoke` use the matching checked-in grid
and the launcher resources shown above. Pilot smoke intentionally uses the
production pilot launch profile; non-pilot smoke uses `gpu_test`.

The command prints the controller job id and stack directory, then exits. All
controller output, child job ids, Slurm accounting records, copied submission
script, manifest, and final status are retained under:

```text
results/stack/<stack_id>/
```

The controller requests four CPUs and `8G` per CPU using `--mem-per-cpu`. It
holds that CPU allocation while polling, favoring operational simplicity over
allocation efficiency. Override controller placement or walltime when needed:

```bash
STACK_CONTROLLER_PARTITION=kozinsky \
STACK_CONTROLLER_TIME=7-00:00:00 \
$STUDY/submit_stack.sh full
```
