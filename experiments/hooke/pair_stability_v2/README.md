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

## Screening Run

Use explicit attempt ids for traceability:

```bash
STUDY=experiments/hooke/pair_stability_v2
GRID_ATTEMPT=pr8_11_scan
COLLECT_ATTEMPT=pr8_11_collect
SELECT_ATTEMPT=pr8_11_select
```

Plan:

```bash
uv run python $STUDY/plan.py \
  --attempt-id $GRID_ATTEMPT \
  --blind \
  --blind-seed 811
```

Train:

```bash
uv run --extra submitit python $STUDY/train.py \
  --backend submitit --cuda \
  --grid-attempt-id $GRID_ATTEMPT \
  --chunk-size 32 --slurm-array-parallelism 2 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 120
```

Validate completed train attempts:

```bash
uv run --extra submitit python $STUDY/validate.py \
  --backend submitit --cuda \
  --grid-attempt-id $GRID_ATTEMPT \
  --only-ready \
  --chunk-size 32 --slurm-array-parallelism 2 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 60
```

Collect and select energy representatives:

```bash
uv run python $STUDY/collect.py \
  --grid-attempt-id $GRID_ATTEMPT \
  --attempt-id $COLLECT_ATTEMPT

uv run python $STUDY/select_champions.py \
  --collection-attempt-id $COLLECT_ATTEMPT \
  --attempt-id $SELECT_ATTEMPT
```

`validate.py` supports `--wait-job <job_id>` when the upstream Submitit
launcher job id is known. Otherwise, rerun validation with `--only-ready` after
train jobs have written completed checkpoints.
