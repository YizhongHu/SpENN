# Hooke Pair Validation Methods

This document describes the `hooke_pair_validation_v1` experiment protocol. See
[README.md](README.md) for the command runbook.

## Study Role

The study selects one Hooke-pair model/protocol from a validation scan and then
freezes that choice for held-out final evaluation.

Validation is used only for selection. It does not use exact reference energy.
Final evaluation happens after selection is frozen, so it may report exact
reference-energy comparisons.

Local run directories and generated CSV, JSON, and Markdown reports are
authoritative. W&B is visualization only and is not used as the source of the
selection decision.

## Protocol Source

[manifest.yaml](manifest.yaml) is the protocol contract. It declares the study
name, training config, final-evaluation template, grid axes, seed axis,
selection rule, eligibility checks, geometry-warning policy, final seeds, final
sampler settings, and default Slurm resources.

Changing the manifest after running the study implies a new study version.

## Validation Scan

Validation scan runs use
`experiments/hooke/configs/benchmark/pair_train.yaml`. Each grid point runs a
normal `spenn.runner.Train` job, writes local run artifacts, saves directory
checkpoints under `checkpoints/`, and runs a train-end validation callback with
an independent validation sampler.

The manifest grid is:

```text
runtime.seed: [3, 9, 11]
optimizer_params.lr: [3.0e-4, 1.0e-3, 3.0e-3]
model_params.channels: [8, 32, 128]
model_params.layers: [1]
model_params.gate_activation: [silu, sigmoid]
```

The seed key is `runtime.seed`. All other grid keys define the non-seed config
group used for selection.

Validation metrics are logged under:

```text
validation/*
validation/sampler/*
validation/perf/*
```

The validation callback must not log exact-reference selection metrics such as
`validation/energy_error`, `validation/energy_abs_error`, or
`validation/reference_energy`.

## Launcher Behavior

Validation jobs are submitted with Hydra Submitit. The shell launcher reads the
manifest, expands one Hydra multirun job per manifest grid point, runs
`uv sync --extra ... --extra submitit`, activates `.venv` or `.venv-gpu`, and
then uses direct `python -u run.py ...` commands inside the job.

Real CPU submissions default to `sapphire,kozinsky,seas_compute`. Real GPU
submissions default to `kozinsky_gpu,seas_gpu`. Cluster smoke submissions use
the dedicated [smoke_manifest.yaml](smoke_manifest.yaml), the smaller test
partitions (`test` for CPU and `gpu_test` for GPU), and a 15-minute timeout.
The launcher escapes partition commas for Hydra; Slurm still receives the
normal comma-separated partition list.

Each Submitit task is shaped like:

```bash
python -u run.py \
  --config experiments/hooke/configs/benchmark/pair_train.yaml \
  run.root=outputs/hooke_pair_validation_v1 \
  study.name=hooke_pair_validation_v1 \
  study.config_id=<non-seed-config-id> \
  runtime.device=<cpu-or-cuda> \
  runtime.seed=<manifest-grid-seed> \
  optimizer_params.lr=<manifest-grid-lr> \
  model_params.channels=<manifest-grid-channels> \
  model_params.layers=<manifest-grid-layers> \
  model_params.gate_activation=<manifest-grid-gate>
```

## Collection

`collect.py` normalizes local run outputs into `runs.csv` and `runs.jsonl`.
It reads raw run artifacts and does not delete, move, rewrite, or repair run
directories.

By default, the collector includes only runs whose resolved config has:

```yaml
study:
  name: hooke_pair_validation_v1
```

## Selection Rule

The primary selection metric is median `validation/energy` across validation
training seeds. Failed runs, missing metrics, and missing validation outputs
count as `+inf`, so incomplete candidates cannot silently win.

Eligibility requires configured integrity, gradient, and equivariance checks to
pass, and requires a finite local-energy fraction of `1.0`.

A lower median energy clearly beats another candidate only when it wins by the
manifest margin:

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

## Final Evaluation

Final evaluation uses held-out training seeds `100` through `109` and held-out
evaluation seeds `100000` through `100009`.

`evaluate_selected.py` generates final training configs, final evaluation
configs, a final-evaluation manifest, command files, and an input table for the
final Submitit launcher.

Generated final evaluation configs load the selected final-training checkpoint
explicitly:

```yaml
load:
  path: /path/to/checkpoints/latest.json
  mode: model_only
  strict: true
  allow_protocol_mismatch: false
```

Checkpoint loading remains runner-owned and explicit; it is not hidden inside
model instantiation.

Final training and final evaluation are launched as separate stages. The real
final evaluation stage checks that configured checkpoint paths exist before
running unless `require_checkpoint=false` is passed for an operational dry run.

## Reproducibility

Keep the manifest version, git SHA, Slurm logs, `run_start.json`,
`resolved_config.yaml`, `metadata.json`, `status.json`, metric files, checkpoint
manifests, selection report, final-evaluation manifest, final run summaries, and
local run directories.

Generated real-run artifacts should not be committed as canonical source files.
Tiny fixtures for tests are fine.
