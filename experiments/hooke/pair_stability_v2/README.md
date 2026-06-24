# Hooke pair-stability V2 screening study (PR8.11)

This package is the breaking-refactor version of
`experiments/hooke/pair_stability`. PR8.11 uses it for a screening scan over
modular model-side scale controls, not a final-reporting experiment.

The default grid is:

```text
major_grid: basis x mechanism
minor_grid: lr x channels
scan_seeds: training/validation replicate seeds
champions: energy selector only
final_replicates: 0
```

The checked-in scan has:

```text
3 bases x 10 mechanisms x 3 learning rates x 1 channel count x 3 seeds = 270 scan jobs
30 major points x 1 energy representative = 30 selected screening champions
0 final jobs by default
```

## Stages

The stage layout remains:

```text
00_grid -> 01_train -> 02_validation -> 03_collect -> 04_select
        -> 05_final_grid -> 06_final_train -> 07_final_eval
        -> 08_final_collect -> 09_final_report
```

For PR8.11 the normal stopping point is `04_select`. `final_plan.py` will write
a `05_final_grid` attempt with zero final jobs unless `--replicates` is passed
explicitly.

## Config Semantics

The training and validation configs expose slot-like run parameters:

```yaml
run_parameters:
  basis_slot: B00
  mechanism_slot: A00
  lr: 1.0e-3
  channels: 8
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
b-B00_m-A03_lr-1e-3_ch-8_seed-0
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

Pass explicit `--attempt-id`, `--grid-attempt-id`, or previous-stage attempt
flags only when reproducing an older lineage or debugging.

## Screening Run

Set the study path once:

```bash
STUDY=experiments/hooke/pair_stability_v2
```

Plan the grid. The attempt id is generated automatically and recorded in
`results/00_grid/latest.json`.

```bash
uv run python $STUDY/plan.py \
  --blind \
  --blind-seed 811
```

Train the latest grid:

```bash
uv run --extra submitit python $STUDY/train.py \
  --backend submitit --cuda \
  --chunk-size 6 \
  --slurm-timeout-min 480
```

Validate completed train attempts from the latest grid:

```bash
uv run --extra submitit python $STUDY/validate.py \
  --backend submitit --cuda \
  --chunk-size 32 \
  --slurm-timeout-min 480 \
  --wait-job <train_launcher_job_id>
```

Collect the newest validation lineage and select energy representatives. These
commands do not need a grid attempt id; collection traces validation ancestry to
the source grid manifest.

```bash
uv run python $STUDY/collect.py

uv run python $STUDY/select_champions.py
```

## Optional Final Stages

V2 is a screening study, so the checked-in grid sets `final_replicates: 0`.
Run `final_plan.py` with an explicit non-zero `--replicates` when you want to
continue selected champions through report-grade final training/evaluation.
The default commands below consume the latest previous stage and write their own
latest pointers.

Plan final replicates from the latest champion selection:

```bash
# Example report-grade diagnostic plan: two final replicates per champion.
uv run python $STUDY/final_plan.py \
  --replicates 2
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
  --backend submitit --cuda \
  --chunk-size 6 \
  --slurm-timeout-min 480
```

Launch final evaluation from the latest final grid and the latest ready
final-train checkpoint for each final run:

```bash
uv run --extra submitit python $STUDY/final_eval.py \
  --backend submitit --cuda \
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

Smoke final stages use the same lineage defaults but cap the final grid to the
first one or two champions, use one final replicate, mark attempts with
`-smoke`, and use the test partitions:

```bash
uv run python $STUDY/final_plan.py --smoke

uv run --extra submitit python $STUDY/final_train.py \
  --smoke \
  --backend submitit --cuda \
  --chunk-size 1

uv run --extra submitit python $STUDY/final_eval.py \
  --smoke \
  --backend submitit --cuda \
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
still follows `--cpu`/`--cuda` and `--smoke` partition defaults when the
dependent launcher runs.

## Smoke Runs

Smoke runs are separate from full runs. Passing `--smoke` keeps the same source
grid but writes smoke-marked attempts, limits launchers to two jobs, sends CPU
smoke jobs to `test` and CUDA smoke jobs to `gpu_test` by default, and applies
only the stage-specific
workload reductions in `configs/smoke.yaml`. The smoke profile mirrors the
small scaling used by `experiments/hooke/pair_stability`: two train steps,
small sampler settings, checkpoint/status every step, and compact validation
sample counts.

Example GPU smoke train from the latest grid:

```bash
uv run --extra submitit python $STUDY/train.py \
  --smoke \
  --backend submitit --cuda \
  --chunk-size 1
```
