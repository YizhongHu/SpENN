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
collect.py            newest validation attempt -> source 00_grid
select_champions.py   latest 03_collect
final_plan.py         latest 04_select
```

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
  --only-ready \
  --chunk-size 32 \
  --slurm-timeout-min 480
```

Collect the newest validation lineage and select energy representatives. These
commands do not need a grid attempt id; collection traces validation ancestry to
the source grid manifest.

```bash
uv run python $STUDY/collect.py

uv run python $STUDY/select_champions.py
```

`validate.py` and `final_eval.py` support `--wait-job <job_id>` when the
upstream Submitit launcher job id is known. They submit a lightweight Slurm
launcher with `--dependency=afterany:<job_id>` and exit immediately; the
dependent launcher reruns the same stage command without `--wait-job` and then
performs the normal readiness checks. Otherwise, rerun validation/final eval
with `--only-ready` after upstream checkpoints are ready. The lightweight
launcher defaults to the `test` partition; override it with
`--wait-launcher-partition` if needed.

## Smoke Runs

Smoke runs are separate from full runs. Passing `--smoke` keeps the same source
grid but writes smoke-marked attempts, limits launchers to two jobs, uses the
`gpu_test`/`test` partitions by default, and applies only the stage-specific
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
