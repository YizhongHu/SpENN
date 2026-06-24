# Hooke pair-stability V2 study (PR8.10)

This package is the breaking-refactor version of
`experiments/hooke/pair_stability`. It keeps the same staged artifact lineage,
but the scan config is split into:

```text
major_grid: architecture x normalization
minor_grid: lr x channels
scan_seeds: seed replicates for selecting a minor representative
champions: named selector specs for final representatives
final_replicates: independent final train/eval repeats per champion
```

The default `configs/grid.yaml` is intentionally smoke-sized:

```text
2 architectures x 2 normalizations x 2 learning rates x 2 channel counts x 2 scan seeds = 32 scan jobs
4 major points x 2 champion kinds x 2 final replicates = 16 final jobs
```

`pair_stability_small/` remains the untracked PR8.9 flat-grid reference
implementation. Its smoke artifacts should stay under
`experiments/hooke/pair_stability_small/results/`. V2 artifacts should stay
under `experiments/hooke/pair_stability_v2/results/`.

## Stages

The stage contract is unchanged:

```text
00_grid -> 01_train -> 02_validation -> 03_collect -> 04_select
        -> 05_final_grid -> 06_final_train -> 07_final_eval
        -> 08_final_collect -> 09_final_report
```

V2 manifests make the split explicit. Each scan job records `major_id`,
`minor_id`, `config_id`, `major_choices`, `minor_choices`, and `scan_seed`.
Final jobs keep `major_id` and `minor_id`, record `source_scan_seeds`, and use
independent final seeds instead of scan seeds.

## Config Semantics

`champions` is not just a list of labels. Each entry names a final
representative and declares how it is selected:

```yaml
major_grid:
  architecture: [...]
  normalization: [...]
minor_grid:
  lr: [...]
  channels: [...]
scan_seed_axis: seed
scan_seeds: [...]

axis_id_labels:
  architecture: arch
  normalization: norm
  lr: lr
  channels: ch
  seed: seed

axis_overrides:
  architecture: run_parameters.architecture
  normalization: run_parameters.normalization
  lr: run_parameters.lr
  channels: run_parameters.channels

choice_validation:
  architecture:
    choices_path: choices.architecture
    tags_path: choices.architecture.{value}.tags
    exclude_suffixes: [_no_envelope]
  normalization:
    choices_path: choices.normalization
```

The planner infers major/minor axis order from the mapping order unless
`major_axes` or `minor_axes` is provided. `axis_id_labels` controls durable
`run_id`/`major_id`/`minor_id` spelling; `axis_overrides` controls which
OmegaConf override path receives each non-seed axis value. `choice_validation`
is optional and config-owned; it is the only place where axis values are tied
to train-config choice libraries or exclusion rules.

```yaml
champions:
  - name: energy
    selector: metric_ladder
    tasks: [stratified_geometry, tail, cusp, hooke_orbital]
    metric_template: eval/{task}/local_energy_mean
    mode: min
    fallback_metric: train/runtime/wall_time_sec
    fallback_mode: min

  - name: stability
    selector: metric
    metric: eval/feature_trace_stability/feature_rms_q95
    mode: min
    exclude: energy
```

The planner snapshots these specs into `00_grid/manifest.json`, and
`select_champions.py` consumes that manifest. Legacy string entries such as
`energy` still map to the default specs, but new studies should prefer explicit
specs so metric names and exclusion rules live in config.

Seed usage is also config-owned:

```yaml
seed_overrides:
  scan_train:
    run_parameters.seed: scan_seed
    runtime.seed: scan_seed
    sampler.seed: scan_seed
  validation:
    run_parameters.seed: scan_seed
    runtime.seed: scan_seed
    evaluation.seed: scan_seed
  final_train:
    run_parameters.seed: final_train_model_seed
    runtime.seed: final_train_model_seed
    sampler.seed: final_train_sampler_seed
  final_eval:
    run_parameters.seed: final_eval_seed
    runtime.seed: final_eval_seed
    evaluation.seed: final_eval_seed

final_seed_sequences:
  final_train_sampler_seed: {start: 101, step: 1}
  final_train_model_seed: {start: 1001, step: 1}
  final_eval_seed: {start: 10001, step: 1}
```

`scan_seeds` are used only for scan train/validation rows. Final jobs discard
scan seeds, generate named final seeds from `final_seed_sequences`, and carry
resolved per-stage seed overrides in each `05_final_grid/jobs/*.json` record.

## GPU Submitit Smoke

Use explicit attempt ids for traceability:

```bash
STUDY=experiments/hooke/pair_stability_v2
GRID_ATTEMPT=pr8_10_v2_scan_smoke
COLLECT_ATTEMPT=pr8_10_v2_collect_smoke
SELECT_ATTEMPT=pr8_10_v2_select_smoke
FINAL_GRID_ATTEMPT=pr8_10_v2_final_grid_smoke
FINAL_TRAIN_ATTEMPT=pr8_10_v2_final_train_smoke
FINAL_EVAL_ATTEMPT=pr8_10_v2_final_eval_smoke
FINAL_COLLECT_ATTEMPT=pr8_10_v2_final_collect_smoke
```

Then run:

```bash
uv run python $STUDY/plan.py --attempt-id $GRID_ATTEMPT

uv run --extra submitit python $STUDY/train.py \
  --backend submitit --cuda \
  --grid-attempt-id $GRID_ATTEMPT \
  --chunk-size 32 --slurm-array-parallelism 2 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 60

uv run --extra submitit python $STUDY/validate.py \
  --backend submitit --cuda \
  --grid-attempt-id $GRID_ATTEMPT \
  --only-ready \
  --chunk-size 32 --slurm-array-parallelism 2 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 30

uv run python $STUDY/collect.py \
  --grid-attempt-id $GRID_ATTEMPT \
  --attempt-id $COLLECT_ATTEMPT

uv run python $STUDY/select_champions.py \
  --collection-attempt-id $COLLECT_ATTEMPT \
  --attempt-id $SELECT_ATTEMPT

uv run python $STUDY/final_plan.py \
  --selection-attempt-id $SELECT_ATTEMPT \
  --attempt-id $FINAL_GRID_ATTEMPT

uv run --extra submitit python $STUDY/final_train.py \
  --backend submitit --cuda \
  --final-grid-attempt-id $FINAL_GRID_ATTEMPT \
  --attempt-id $FINAL_TRAIN_ATTEMPT \
  --chunk-size 16 --slurm-array-parallelism 2 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 60

uv run --extra submitit python $STUDY/final_eval.py \
  --backend submitit --cuda \
  --final-grid-attempt-id $FINAL_GRID_ATTEMPT \
  --final-train-attempt-id $FINAL_TRAIN_ATTEMPT \
  --attempt-id $FINAL_EVAL_ATTEMPT \
  --only-ready \
  --chunk-size 16 --slurm-array-parallelism 2 \
  --slurm-partition gpu_test \
  --slurm-timeout-min 60

uv run python $STUDY/final_collect.py \
  --final-eval-attempt-id $FINAL_EVAL_ATTEMPT \
  --attempt-id $FINAL_COLLECT_ATTEMPT

uv run python $STUDY/final_report.py \
  --final-collect-attempt-id $FINAL_COLLECT_ATTEMPT
```

`validate.py` and `final_eval.py` also support `--wait-job <job_id>` if the
upstream Submitit launcher job id is known. Otherwise, rerun those stages with
`--only-ready` after the upstream train jobs have written completed
checkpoints.
