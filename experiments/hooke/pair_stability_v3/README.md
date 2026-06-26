# Hooke pair-stability V3 study

This package is a resource-reduced copy of
`experiments/hooke/pair_stability_v2`. It keeps the same stage layout, scan
axes, validation/final-evaluation task suites, plots, reports, and metric names,
but uses a smaller default grid and shorter run budgets for restructuring work.

The default grid is:

```text
major_grid: basis x mechanism
minor_grid: lr x channels
scan_seeds: training/validation replicate seeds
champions: energy selector only
final_replicates: 2
```

The checked-in scan has:

```text
2 bases x 2 mechanisms x 2 learning rates x 1 channel count x 2 seeds = 16 scan jobs
4 major points x 1 energy representative = 4 selected champions
4 selected champions x 2 final seeds = 8 final jobs by default
```

## Stages

The stage layout remains:

```text
00_grid -> 01_train -> 02_validation -> 03_collect -> 04_select
        -> 05_final_grid -> 06_final_train -> 07_final_eval
        -> 08_final_collect -> 09_final_report
```

`final_plan.py` writes two final seeds per selected champion by default. Pass
`--replicates` only to override that configured count.

## Config Semantics

The training and validation configs expose slot-like run parameters:

```yaml
run_parameters:
  basis_slot: B00
  mechanism_slot: A00
  lr: 1.0e-3
  channels: 4
  seed: 0
```

`choices.basis` owns the concrete input basis specs. `choices.mechanism` owns
the explicit layer controls:

```text
embedding_activation
feature_activation
feature_envelope
irrep_activation
update_activation
update_envelope
```

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

Attempt ids are names only. Smoke/full identity is recorded in each attempt's
`attempt_metadata.json` and in latest-pointer payloads. When a full attempt is
known, `latest.json` points to the latest full attempt; smoke diagnostics update
`latest-smoke.json` and never displace an existing full default. Full stages
therefore default to the latest non-smoke upstream run, while smoke stages
default to the latest smoke upstream run.

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

## Non-Smoke Runbook

Use the commands in this section for the real study run. They intentionally
omit `--smoke`; each stage consumes the latest non-smoke upstream attempt by
default and will not pick up a newer smoke diagnostic run.

Set the study path once:

```bash
STUDY=experiments/hooke/pair_stability_v3
```

### Scan Stages

Plan the grid. The attempt id is generated automatically in
`America/New_York` and recorded in `results/00_grid/latest.json`.

```bash
uv run python $STUDY/plan.py \
  --blind \
  --blind-seed 811
```

Train the latest grid:

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

Collect the newest non-smoke validation lineage and select energy
representatives. These commands do not need a grid attempt id; collection
traces validation ancestry to the source grid manifest.

```bash
uv run python $STUDY/collect.py

uv run python $STUDY/select_champions.py
```

### Final Stages

The checked-in grid sets `final_replicates: 2`, so the default final plan
continues selected champions through two independent final seeds. The commands
below consume the latest non-smoke previous stage and write their own latest
pointers.

Plan final replicates from the latest champion selection:

```bash
uv run python $STUDY/final_plan.py
```

This writes `results/05_final_grid/<attempt_id>/final_jobs.csv` and
`05_final_grid/latest.json`. Each final job records the selected major-axis
aliases, frozen minor choices, source champion row, replicate index, and
independent final seeds:

```text
final_train_sampler_seed = 101 + replicate_index
final_train_model_seed   = 1001 + replicate_index
final_eval_seed          = 10001 + replicate_index
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

## Smoke Runs

Smoke runs are separate from full runs. Passing `--smoke` keeps the same source
grid but writes smoke-marked attempts, limits launchers to two jobs, sends CPU
smoke jobs to `test` and CUDA smoke jobs to `gpu_test` by default, and applies
only the stage-specific workload reductions in `configs/smoke.yaml`. The smoke
profile mirrors the small scaling used by `experiments/hooke/pair_stability`:
two train steps, small sampler settings, checkpoint/status every step, and
compact validation sample counts.

Example GPU smoke train from the latest grid:

```bash
uv run --extra submitit python $STUDY/train.py \
  --smoke \
  --backend submitit --device cuda \
  --chunk-size 1
```

Smoke final stages use the same lineage defaults but cap the final grid to the
first one or two champions, use one final seed, record smoke metadata, and use
the test partitions:

```bash
uv run python $STUDY/final_plan.py --smoke

uv run --extra submitit python $STUDY/final_train.py \
  --smoke \
  --backend submitit --device cpu,cuda \
  --chunk-size 1

uv run --extra submitit python $STUDY/final_eval.py \
  --smoke \
  --backend submitit --device cuda \
  --wait-job <final_train_launcher_job_id>

uv run python $STUDY/final_collect.py --smoke

uv run python $STUDY/final_report.py --smoke
```

`validate.py` and `final_eval.py` support `--wait-job <job_id>` when the
upstream Submitit launcher job id is known. They submit a lightweight Slurm
launcher with `--dependency=afterany:<job_id>` and exit immediately; the
dependent launcher reruns the same stage command without `--wait-job` and then
performs the normal readiness checks. Otherwise, rerun validation/final eval
after upstream checkpoints are ready; these stages always skip rows that are not
ready. The lightweight launcher defaults to the `test` partition; override it
with `--wait-launcher-partition` if needed. The real validation/final-eval array
still follows `--device` and `--smoke` partition defaults when the
dependent launcher runs.
