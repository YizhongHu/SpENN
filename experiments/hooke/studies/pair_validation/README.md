# Hooke Pair Validation Study

Only `orchestrate.py` launches SpENN. Collectors, selectors, and planners read
and write files only.

The study has one train base config and one eval base config:

- `experiments/hooke/studies/pair_validation/configs/pair_train.yaml`
- `experiments/hooke/studies/pair_validation/configs/pair_eval.yaml`

All scientific variation comes from `manifest.yaml` phase overrides or generated
job rows.

`study.name` is stable across reruns of the same protocol family; `study.version`
is the versioned study instance. Every launched run records both fields plus
`study.phase`. True run, report, and Slurm directories live only in
`manifest.yaml`.

The legacy test/reference files under `experiments/hooke/configs/benchmark/`
are noncanonical copies; this study uses the configs above.

## Phase Flow

1. local CPU `smoke_train`
2. optional GPU `smoke_train`
3. SLURM `smoke_train`
4. `validation_train`
5. `collect.py`
6. `select.py`
7. `plan_final.py --phase final_train`
8. `final_train` smoke/launch through train orchestrators
9. `collect_final.py` for final-train checkpoints
10. `plan_final.py --phase smoke_eval`
11. `smoke_eval`
12. `plan_final.py --phase final_eval`
13. `final_eval`
14. `collect_final.py`

## Smoke Train

See all orchestration options:

```bash
uv run python experiments/hooke/studies/pair_validation/orchestrate.py --help
```

```bash
uv run python experiments/hooke/studies/pair_validation/orchestrate.py \
  --kind train \
  --backend slurm \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase smoke_train \
  --profile gpu \
  --dry-run
```

`smoke_train` is the target train phase plus its manifest `smoke.overlay`.
It uses the same base config and override-generation path as the real train
phase.

SLURM smoke launches use the manifest test profiles: `--profile cpu` submits to
the `test` partition and `--profile gpu` submits to `gpu_test`. In `--dry-run`
mode the orchestrator prints the exact `sbatch --array` command plus the first
array-task command.

For real launches, the run Python comes from the manifest profile
`uv_environment` (`.venv` or `.venv-gpu`) unless `--python` is supplied.

Study launches use flat run directories under the stage `outputs/` directory.
The first run-id folder is always `smoke/` or `full/`, for example
`01_train/outputs/smoke/<config_id>/seed=<seed>/...` and
`01_train/outputs/full/<config_id>/seed=<seed>/...`.
Collectors read only `full/` runs by default; pass `--include-smoke` when a
smoke run table is wanted for debugging.

## Validation Scan

```bash
uv run python experiments/hooke/studies/pair_validation/orchestrate.py \
  --kind train \
  --backend slurm \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase validation_train \
  --profile gpu
```

Validation happens inside Train at `train_end` and logs `validation/*`,
`validation/sampler/*`, and `validation/perf/*`. Selection must not use exact
reference errors.

## Collect

```bash
uv run python experiments/hooke/studies/pair_validation/collect.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase validation_train \
  --output-dir experiments/hooke/studies/pair_validation/reports/02_collect
```

The default run root is
`experiments/hooke/studies/pair_validation/reports/01_train/outputs`, and the
default output directory is
`experiments/hooke/studies/pair_validation/reports/02_collect`.

## Select

```bash
uv run python experiments/hooke/studies/pair_validation/select.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --runs experiments/hooke/studies/pair_validation/reports/02_collect/runs.csv \
  --output-dir experiments/hooke/studies/pair_validation/reports/03_select
```

`select.py` writes `selection.csv`, `selection.jsonl`,
`selection_report.md`, and `selected_config.yaml`. It does not plan final jobs.

## Final Planning

Plan final training:

```bash
uv run python experiments/hooke/studies/pair_validation/plan_final.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/03_select/selected_config.yaml \
  --phase final_train \
  --output-dir experiments/hooke/studies/pair_validation/reports/04_final_train/plans
```

Smoke the selected final-train protocol:

```bash
uv run python experiments/hooke/studies/pair_validation/orchestrate.py \
  --kind train \
  --backend slurm \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase smoke_train \
  --target-phase final_train \
  --selected-config experiments/hooke/studies/pair_validation/reports/03_select/selected_config.yaml \
  --profile gpu \
  --dry-run

uv run python experiments/hooke/studies/pair_validation/orchestrate.py \
  --kind train \
  --backend slurm \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase smoke_train \
  --target-phase final_train \
  --selected-config experiments/hooke/studies/pair_validation/reports/03_select/selected_config.yaml \
  --profile gpu
```

Launch final training:

```bash
uv run python experiments/hooke/studies/pair_validation/orchestrate.py \
  --kind train \
  --backend slurm \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase final_train \
  --selected-config experiments/hooke/studies/pair_validation/reports/03_select/selected_config.yaml \
  --profile gpu
```

After final training produces checkpoints, collect the final-train table:

```bash
uv run python experiments/hooke/studies/pair_validation/collect_final.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/03_select/selected_config.yaml \
  --final-train-root experiments/hooke/studies/pair_validation/reports/04_final_train/outputs
```

Plan the eval smoke row and run it:

```bash
uv run python experiments/hooke/studies/pair_validation/plan_final.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/03_select/selected_config.yaml \
  --final-train-runs experiments/hooke/studies/pair_validation/reports/04_final_train/final_train_runs.csv \
  --phase smoke_eval \
  --output-dir experiments/hooke/studies/pair_validation/reports/05_final_eval/plans

uv run python experiments/hooke/studies/pair_validation/orchestrate.py \
  --kind eval \
  --backend slurm \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase smoke_eval \
  --jobs experiments/hooke/studies/pair_validation/reports/05_final_eval/plans/smoke_eval_jobs.jsonl \
  --profile gpu
```

After eval smoke passes, plan and launch final eval rows:

```bash
uv run python experiments/hooke/studies/pair_validation/plan_final.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/03_select/selected_config.yaml \
  --final-train-runs experiments/hooke/studies/pair_validation/reports/04_final_train/final_train_runs.csv \
  --phase final_eval \
  --output-dir experiments/hooke/studies/pair_validation/reports/05_final_eval/plans
```

`final_eval_jobs.jsonl` contains row-specific `load.path` values and uses
`load.mode=model_only`. It is intentionally not Cartesian: completed final-train
checkpoints are sorted by train seed and paired one-to-one with
`final_evaluation.eval_seeds`, so train seed `100` uses eval seed `100000`, train
seed `101` uses eval seed `100001`, and so on.

## Final Benchmark

```bash
uv run python experiments/hooke/studies/pair_validation/orchestrate.py \
  --kind eval \
  --backend slurm \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --phase final_eval \
  --jobs experiments/hooke/studies/pair_validation/reports/05_final_eval/plans/final_eval_jobs.jsonl \
  --profile gpu

uv run python experiments/hooke/studies/pair_validation/collect_final.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --final-train-root experiments/hooke/studies/pair_validation/reports/04_final_train/outputs \
  --final-eval-root experiments/hooke/studies/pair_validation/reports/05_final_eval/outputs \
  --final-eval-jobs experiments/hooke/studies/pair_validation/reports/05_final_eval/plans/final_eval_jobs.jsonl
```

`collect_final.py` writes final train/eval tables plus
`final_benchmark_summary.csv`, `final_benchmark_summary.json`, and
`final_benchmark_report.md`.

Generate physics-sanity tables, plots, and the expanded final report from the
collected final-eval files:

```bash
uv run python experiments/hooke/studies/pair_validation/plot_final.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml
```

`plot_final.py` is file-only: it reads `final_eval_runs.csv`, summary files,
and indexed diagnostic artifacts. It does not load checkpoints, models, or exact
wavefunctions.

## Sync Reports

Use `sync_reports.py` to mirror the manifest report directory into another
location while keeping the snapshot compact. The destination is replaced on each
run. Slurm log directories and training checkpoints are skipped. Eval runs keep
only `checkpoints/latest.json` plus the checkpoint step directory referenced by
that file.

Preview the copy plan:

```bash
uv run python experiments/hooke/studies/pair_validation/sync_reports.py \
  ${MStore}/spenn-studies/hooke/pair_validation_v2/reports_snapshot \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --dry-run \
  --verbose
```

Write the snapshot:

```bash
uv run python experiments/hooke/studies/pair_validation/sync_reports.py \
  ${MStore}/spenn-studies/hooke/pair_validation_v2/reports_snapshot \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml
```

Pass `--source` to mirror a report directory other than the one in
`manifest.yaml`.

## Outputs To Keep

- `manifest.yaml`
- `01_train/outputs/<run_id>/`, `01_train/slurm_logs/`
- `02_collect/runs.csv`, `02_collect/runs.jsonl`
- `03_select/selection.csv`, `03_select/selection.jsonl`
- `03_select/selection_report.md`, `03_select/selected_config.yaml`
- `04_final_train/plans/final_train_manifest.yaml`
- `04_final_train/plans/final_train_jobs.jsonl`
- `04_final_train/final_train_runs.csv`, `04_final_train/final_train_runs.jsonl`
- `04_final_train/outputs/<run_id>/`, `04_final_train/slurm_logs/`
- `05_final_eval/plans/smoke_eval_manifest.yaml`
- `05_final_eval/plans/smoke_eval_jobs.jsonl`
- `05_final_eval/plans/final_eval_manifest.yaml`
- `05_final_eval/plans/final_eval_jobs.jsonl`
- `05_final_eval/final_eval_runs.csv`, `05_final_eval/final_eval_runs.jsonl`
- `05_final_eval/final_benchmark_summary.csv`
- `05_final_eval/final_benchmark_summary.json`
- `05_final_eval/final_benchmark_report.md`
- `05_final_eval/tables/`
- `05_final_eval/plots/`

W&B is visualization only. Local run directories and reports are authoritative.
