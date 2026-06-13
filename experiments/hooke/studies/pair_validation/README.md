# Hooke Pair Validation Study

## Overview

This directory owns the PR8.2 post-processing workflow for
`hooke_pair_validation_v1`: collect validation-scan run outputs, select one
model/protocol deterministically, and generate final benchmark training and
evaluation commands.

Validation is used for model/protocol selection. Validation does not use exact reference energy.
Final evaluation is run after selection is frozen, and final evaluation may
compare against the exact Hooke reference energy.

Local run directories and generated CSV/JSON/Markdown reports are
authoritative. W&B is visualization only. Do not use W&B clicks as the source
of the selection decision.

## Prerequisites

Run commands from the repository root with the `uv` environment. The validation
launcher uses `uv sync --extra ... --extra submitit`, activates `.venv` or
`.venv-gpu`, and then runs `python` directly instead of `uv run` inside Slurm
jobs. Keep local run directories, `metrics.csv`, `metrics.jsonl`,
`metadata.json`, `run_start.json`, `status.json`, checkpoint directories,
generated reports, and SLURM logs for reproducibility.

The post-processing scripts are read-only over raw run directories. They do not
delete, move, rewrite, or repair raw run artifacts.

## Validation Scan Protocol

[manifest.yaml](manifest.yaml) declares the study name, grid axes, validation
seed axis, selection rule, eligibility checks, geometry-warning policy, and
final-evaluation seed/sampler policy. Changing the manifest after running the
study implies a new study version.

Validation scan runs use
`experiments/hooke/configs/benchmark/pair_train.yaml`. Each grid point trains a
normal `spenn.runner.Train` run, writes local artifacts, saves directory
checkpoints under `checkpoints/`, and runs train-end validation with an
independent validation sampler.

## How To Launch Training Scan

The launcher reads [manifest.yaml](manifest.yaml), builds one Hydra multirun
job per manifest grid point, and submits those jobs with Hydra Submitit:

```bash
mkdir -p slurm_logs
bash experiments/hooke/studies/pair_validation/launch_array.sh
```

CPU example:

```bash
DEVICE=cpu bash experiments/hooke/studies/pair_validation/launch_array.sh
```

Dry-run the generated Hydra jobs without running training:

```bash
bash experiments/hooke/studies/pair_validation/launch_array.sh -- dry_run=true
```

Smoke-test launcher wiring without Slurm submission:

```bash
DEVICE=cpu HYDRA_LAUNCHER=submitit_local \
  bash experiments/hooke/studies/pair_validation/launch_array.sh -- \
  dry_run=true job_index=0,1
```

For one-off cluster resource changes, prefer environment overrides rather than
editing the script:

```bash
DEVICE=cpu PARTITION=seas_compute GRES="" \
  bash experiments/hooke/studies/pair_validation/launch_array.sh
```

Each Submitit task runs a direct venv Python command shaped like:

```bash
python -u run.py \
  --config experiments/hooke/configs/benchmark/pair_train.yaml \
  run.root=outputs/hooke_pair_validation_v1 \
  study.name=hooke_pair_validation_v1 \
  study.config_id=<non-seed-config-id> \
  runtime.device=cuda \
  runtime.seed=<manifest-grid-seed> \
  optimizer_params.lr=<manifest-grid-lr> \
  model_params.channels=<manifest-grid-channels> \
  model_params.layers=<manifest-grid-layers> \
  model_params.gate_activation=<manifest-grid-gate>
```

## How Validation Runs At Train End

The training config includes a `spenn.callback.Validation` callback on
`train_end`. It logs selection metrics under `validation/*`,
`validation/sampler/*`, and `validation/perf/*`. It never logs
`validation/energy_error`, `validation/energy_abs_error`, or
`validation/reference_energy`.

## How To Collect Results

Collect local run outputs into normalized tables:

```bash
uv run python experiments/hooke/studies/pair_validation/collect.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/reports
```

By default, the collector includes only runs whose resolved config has
`study.name: hooke_pair_validation_v1`. Use `--allow-other-studies` only for
debugging mixed fixtures. The collector writes `runs.csv` and `runs.jsonl`.

## How To Select The Winning Config

Apply the manifest selection rule:

```bash
uv run python experiments/hooke/studies/pair_validation/select.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --runs experiments/hooke/studies/pair_validation/reports/runs.csv \
  --output-dir experiments/hooke/studies/pair_validation/reports
```

The selector writes `selection.csv`, `selected_config.yaml`, and
`selection_report.md`. Failed or missing validation seeds count as `+inf`
validation energy. If no candidate has a finite eligible median validation
score, selection fails instead of inventing a winner.

## Tie-Breaker Rule

The primary metric is median `validation/energy` across training seeds. A lower
median only clearly beats another candidate when it wins by the manifest
selection margin:

```text
max(
  2 * sqrt(stderr_a^2 + stderr_b^2),
  0.25 * max(iqr_a, iqr_b),
  1.0e-4,
)
```

Within that margin, tie-breakers are applied in order:

```text
lower median validation/energy_variance
lower validation-energy IQR across seeds
lower median validation/energy_stderr
fewer sampler-geometry warnings
smaller model_params.channels
lower median runtime/wall_time_sec
```

Geometry diagnostics are warnings and tie-breakers, not a primary objective.

## How To Generate Final Evaluation Commands

Generate final training and evaluation configs plus command files:

```bash
uv run python experiments/hooke/studies/pair_validation/evaluate_selected.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/selected_config.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/reports \
  --dry-run
```

Dry-run is the default. Generated final evaluation configs use the PR8.1 load
interface:

```yaml
load:
  path: /path/to/checkpoints/latest.json
  mode: model_only
  strict: true
  allow_protocol_mismatch: false
```

Checkpoint loading remains runner-owned and explicit; it is not hidden inside
model instantiation.

## How To Run Final Evaluation

Inspect `final_eval_commands.sh`, then submit or execute the generated commands.
The command script contains final training commands for the manifest final
training seeds followed by final evaluation commands that load each generated
training checkpoint through `load.mode=model_only`. Activate the intended venv
before running the command script directly.

To execute locally:

```bash
bash experiments/hooke/studies/pair_validation/reports/final_eval_commands.sh
```

For cluster runs, use the separate final Submitit launcher in two phases. First
submit final training:

```bash
STAGE=final_train INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh
```

After final training checkpoints exist, submit held-out final evaluation:

```bash
STAGE=final_eval INPUTS=experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh
```

Smoke-test either final phase without Slurm submission:

```bash
DEVICE=cpu HYDRA_LAUNCHER=submitit_local STAGE=final_train \
  bash experiments/hooke/studies/pair_validation/launch_final_submitit.sh -- \
  dry_run=true job_index=0,1
```

Keep the SLURM logs alongside the local run directories. The final eval phase
checks that each configured checkpoint path exists before running unless
`require_checkpoint=false` is passed for a dry operational test.

## How To Summarize Final Evaluation Results

After final evaluation outputs exist, collect final benchmark summaries:

```bash
uv run python experiments/hooke/studies/pair_validation/evaluate_selected.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/reports/selected_config.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/reports \
  --collect
```

The final summary may include `eval/energy_error`, `eval/energy_abs_error`, and
`eval/reference_energy` because it is post-selection reporting.

## Output Files

Collector outputs:

```text
runs.csv
runs.jsonl
```

Selector outputs:

```text
selection.csv
selected_config.yaml
selection_report.md
```

Final-evaluation planning outputs:

```text
final_train_configs/
final_eval_configs/
final_eval_commands.sh
final_eval_manifest.yaml
final_eval_inputs.csv
```

Final-evaluation collection outputs:

```text
final_eval_runs.csv
final_benchmark_summary.csv
final_benchmark_summary.json
final_benchmark_report.md
```

Generated real-run artifacts should not be committed as canonical source files.
Tiny fixtures for tests are fine.

## W&B Role

W&B is visualization only. Local run directories, metrics files, metadata,
status files, checkpoints, and generated summary files are authoritative. Do
not select a model from W&B clicks, panels, or manual notes.

## Reproducibility Notes

Record the manifest version, git SHA, SLURM logs, `run_start.json`,
`resolved_config.yaml`, `metadata.json`, `status.json`, metric files, checkpoint
manifests, selection report, and final-evaluation manifest. Validation and
final evaluation use different seeds by default: validation training seeds are
`3, 9, 11`, while final training seeds are `100` through `109` and final
evaluation seeds are `100000` through `100009`.
