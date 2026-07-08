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
all variants.

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
| `--device cpu` | `.venv` | `cpu` | `runtime.device=cpu` | `slurm_partition=sapphire,kozinsky,seas_compute`, `cpus_per_task=16`, `mem_gb=128`, no GPUs |
| `--device cuda` | `.venv-gpu` | `cu126` | `runtime.device=cuda` | `slurm_partition=seas_gpu,kozinsky_gpu`, `cpus_per_task=8`, `mem_gb=80`, `gpus_per_node=1` |
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

Mixed `cpu,cuda` mode uses separate Submitit submissions because GPU resources
cannot be requested on CPU partitions. Use `--slurm-cpu-partition` and
`--slurm-cuda-partition` to pin the two candidates separately, such as `test`
and `gpu_test` for smoke sanity checks. Use `--slurm-cpu-timeout-min` and
`--slurm-cuda-timeout-min` when CPU and CUDA candidates need different walltime
limits.

## Runbook

Full and smoke runs use the same stage stack. The differences are the grid file,
Slurm partitions, and chunk sizes. `--smoke` is retired; use `configs/smoke.yaml`
for the smaller full-workflow smoke run.

Set the study path once:

```bash
STUDY=experiments/hooke/pair_stability_v3
```

### Full Scan Stages

Plan the full grid. The attempt id is generated automatically in
`America/New_York` and recorded in `results/00_grid/latest.json`.

```bash
uv run python $STUDY/plan.py \
  --grid $STUDY/configs/grid.yaml \
  --blind \
  --blind-seed 811
```

Train and validate the latest full grid:

```bash
uv run --extra submitit python $STUDY/train.py \
  --backend submitit --device cuda \
  --chunk-size 6 \
  --slurm-timeout-min 480
```

Validate completed train attempts from the latest grid. If the train launcher
job id is known, `--wait-job` submits a dependent launcher and exits; otherwise,
run the same command after train checkpoints are ready.

```bash
uv run --extra submitit python $STUDY/validate.py \
  --backend submitit --device cuda \
  --chunk-size 32 \
  --slurm-timeout-min 480 \
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
  --backend submitit --device cpu,cuda \
  --chunk-size 1 \
  --slurm-cpu-timeout-min 60 \
  --slurm-cuda-timeout-min 30
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
  --slurm-timeout-min 480 \
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

### Smoke Scan Stages

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
  --chunk-size 16 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 60

uv run --extra submitit python $STUDY/validate.py \
  --backend submitit --device cuda \
  --chunk-size 32 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 120 \
  --wait-job <train_launcher_job_id>

uv run python $STUDY/collect.py

uv run python $STUDY/select_champions.py
```

### Smoke Final Stages

The smoke grid sets `final_replicates: 1`, so final planning continues each
smoke champion through one final seed.

```bash
uv run python $STUDY/final_plan.py

uv run --extra submitit python $STUDY/final_train.py \
  --backend submitit --device cpu,cuda \
  --chunk-size 8 \
  --slurm-cpu-partition test \
  --slurm-cuda-partition gpu_test \
  --slurm-cpu-timeout-min 60 \
  --slurm-cuda-timeout-min 60

uv run --extra submitit python $STUDY/final_eval.py \
  --backend submitit --device cuda \
  --chunk-size 8 \
  --slurm-partition gpu_test \
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

### Smoke Review Gate

The smoke lineage is the release-readiness gate for this study. It should reach
`09_final_report` before any `0.2.1` version bump. Do not use `--smoke`; the
smoke attempt is normal grid lineage generated from `configs/smoke.yaml`.
Keep review smoke attempts in the default study results directory,
`experiments/hooke/pair_stability_v3/results`, so the artifacts remain
available for comparison and audit. Do not pass a temporary `--results-root`
for the Slurm smoke gate.

Expected smoke artifacts:

```text
results/00_grid/<attempt_id>/manifest.json          64 scan jobs
results/03_collect/<attempt_id>/summary.csv         64 validation rows when complete
results/04_select/<attempt_id>/champions.csv        8 energy champions
results/05_final_grid/<attempt_id>/final_jobs.csv   8 final jobs
results/08_final_collect/<attempt_id>/*.csv         compact report tables
results/09_final_report/<attempt_id>/report.md      review report
results/09_final_report/<attempt_id>/figures/*.png  review figures
```

Smoke Slurm sanity should use `test` and `gpu_test` partitions and the same
stage order as the full run. Record the launcher job id from train and
final-train submissions, then pass it to downstream `--wait-job` commands so
validation and final-eval enter the queue as dependencies rather than requiring
manual polling.

Minimum pre-bump checks:

```bash
uv run --extra cpu pytest -q experiments/hooke/pair_stability_v3
```

Optional local mini-lineage for debugging uses the same smoke grid with
`--backend local`, but the release gate is the Slurm smoke lineage in the
regular results directory. After the Slurm report exists, inspect report
tables/figures and cost summaries before promoting any version bump.
