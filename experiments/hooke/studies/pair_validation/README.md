# Hooke Pair Validation Study

This is the runbook for `hooke_pair_validation_v1`.

Run commands from the repository root. Keep local run directories, generated
reports, checkpoints, and `slurm_logs/`.

## Quick Start

First, run the local CPU smoke on this node:

```bash
uv run pytest -q \
  tests/integration/hooke/test_pair_validation_study.py::test_local_smoke_pipeline_runs_collects_selects_and_plans
```

Then run launcher preflight tests:

```bash
uv run pytest -q \
  tests/integration/hooke/test_pair_validation_study.py::test_submitit_launcher_has_small_cpu_and_gpu_preflight_overrides \
  tests/integration/hooke/test_pair_validation_study.py::test_final_submitit_launcher_has_small_cpu_and_gpu_preflight_overrides
```

On the cluster, submit one CPU and one GPU dry-run job before launching the real
scan:

```bash
DEVICE=cpu HYDRA_LAUNCHER=submitit_slurm ARRAY_PARALLELISM=1 \
  RUN_ROOT=outputs/hooke_pair_validation_v1_cpu_smoke \
  bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
  dry_run=true job_index=0

DEVICE=cuda HYDRA_LAUNCHER=submitit_slurm ARRAY_PARALLELISM=1 \
  RUN_ROOT=outputs/hooke_pair_validation_v1_gpu_smoke \
  bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
  dry_run=true job_index=0
```

If those jobs print a clear `python -u run.py ...` command and exit
successfully, launch the validation scan:

```bash
DEVICE=cuda bash experiments/hooke/studies/pair_validation/launch_array.sh
```

Use `DEVICE=cpu` only when you intentionally want the CPU Slurm profile.

## Local Checks

Run the whole pair-validation test file:

```bash
uv run pytest -q tests/integration/hooke/test_pair_validation_study.py
```

Dry-run validation Submitit locally without Slurm submission:

```bash
DEVICE=cpu HYDRA_LAUNCHER=submitit_local \
  bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
  dry_run=true job_index=0

DEVICE=cuda HYDRA_LAUNCHER=submitit_local \
  bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
  dry_run=true job_index=0
```

These commands run `uv sync --extra ... --extra submitit`, activate `.venv` or
`.venv-gpu`, expand one Hydra job, print the direct Python command, and stop
before training because `dry_run=true`.

## Cluster Smoke

Use these before any large submission. They are intentionally one-job Slurm
submissions:

```bash
DEVICE=cpu HYDRA_LAUNCHER=submitit_slurm ARRAY_PARALLELISM=1 \
  RUN_ROOT=outputs/hooke_pair_validation_v1_cpu_smoke \
  bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
  dry_run=true job_index=0

DEVICE=cuda HYDRA_LAUNCHER=submitit_slurm ARRAY_PARALLELISM=1 \
  RUN_ROOT=outputs/hooke_pair_validation_v1_gpu_smoke \
  bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
  dry_run=true job_index=0
```

Check `slurm_logs/hooke_pair_validation_v1/` after each smoke. The job should
show the exact `python -u run.py --config ...` command and should not run
training.

## Validation Scan

Submit the real validation scan on GPU:

```bash
DEVICE=cuda bash experiments/hooke/studies/pair_validation/launch_array.sh
```

For a CPU validation scan:

```bash
DEVICE=cpu bash experiments/hooke/studies/pair_validation/launch_array.sh
```

The launcher reads [manifest.yaml](manifest.yaml), expands the declared grid,
and submits one Hydra Submitit job per grid point.

## Collect And Select

After validation jobs finish, collect local run artifacts:

```bash
uv run python experiments/hooke/studies/pair_validation/collect.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/reports
```

Select the winning non-seed config:

```bash
uv run python experiments/hooke/studies/pair_validation/select.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --runs experiments/hooke/studies/pair_validation/reports/runs.csv \
  --output-dir experiments/hooke/studies/pair_validation/reports
```

Review:

```text
experiments/hooke/studies/pair_validation/reports/selection_report.md
experiments/hooke/studies/pair_validation/reports/selected_config.yaml
```

## Final Benchmark

Generate final training and evaluation configs:

```bash
uv run python experiments/hooke/studies/pair_validation/evaluate_selected.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/selected_config.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/reports \
  --dry-run
```

Smoke-test the final launcher locally after `final_eval_inputs.csv` exists:

```bash
INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  DEVICE=cpu HYDRA_LAUNCHER=submitit_local STAGE=final_train \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh -- \
  dry_run=true job_index=0

INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  DEVICE=cuda HYDRA_LAUNCHER=submitit_local STAGE=final_eval \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh -- \
  dry_run=true job_index=0
```

Smoke-test one final Slurm job per phase:

```bash
INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  DEVICE=cuda HYDRA_LAUNCHER=submitit_slurm ARRAY_PARALLELISM=1 \
  STAGE=final_train \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh -- \
  dry_run=true job_index=0

INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  DEVICE=cuda HYDRA_LAUNCHER=submitit_slurm ARRAY_PARALLELISM=1 \
  STAGE=final_eval \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh -- \
  dry_run=true job_index=0
```

Submit final training:

```bash
INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  DEVICE=cuda STAGE=final_train \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh
```

Submit final evaluation only after final training checkpoints exist:

```bash
INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  DEVICE=cuda STAGE=final_eval \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh
```

Collect final benchmark summaries:

```bash
uv run python experiments/hooke/studies/pair_validation/evaluate_selected.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/selected_config.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/reports \
  --collect
```

## Outputs To Keep

Validation reports:

```text
runs.csv
runs.jsonl
selection.csv
selected_config.yaml
selection_report.md
```

Final planning outputs:

```text
final_train_configs/
final_eval_configs/
final_eval_commands.sh
final_eval_manifest.yaml
final_eval_inputs.csv
```

Final summary outputs:

```text
final_eval_runs.csv
final_benchmark_summary.csv
final_benchmark_summary.json
final_benchmark_report.md
```

For every real run, keep:

```text
run_start.json
resolved_config.yaml
metadata.json
status.json
metrics.csv
metrics.jsonl
events.jsonl
checkpoints/
slurm_logs/
```

## Reference

The manifest owns the study name, grid, validation seed axis, selection rule,
eligibility checks, geometry-warning policy, final-evaluation seeds, and default
Slurm resources. Changing [manifest.yaml](manifest.yaml) after running the
study implies a new study version.

Validation uses `experiments/hooke/configs/benchmark/pair_train.yaml`.
Validation metrics are logged under `validation/*`, `validation/sampler/*`, and
`validation/perf/*`. Validation does not use exact reference energy.

Selection uses median `validation/energy` across validation training seeds.
Failed or missing seeds count as `+inf`. Exact-reference metrics such as
`validation/energy_abs_error` are forbidden for selection. W&B is visualization only.
Local run directories and generated reports are authoritative.

Final evaluation uses held-out training seeds `100` through `109` and held-out
evaluation seeds `100000` through `100009`. Generated eval configs load
checkpoints explicitly:

```yaml
load:
  path: /path/to/checkpoints/latest.json
  mode: model_only
  strict: true
  allow_protocol_mismatch: false
```

The final eval phase checks checkpoint paths before running unless
`require_checkpoint=false` is passed for a dry operational test.
