# Hooke Pair Study Methods

Validation is used only for selection. Final evaluation is reported after the
selection policy freezes.

Only `orchestrate.py` launches SpENN. Run
`uv run python experiments/hooke/studies/pair_validation/orchestrate.py --help`
or `-h` for usage and examples.

The manifest separates stable `study.name` from `study.version`. Orchestrated
runs record `study.name`, `study.version`, and `study.phase`; scripts read true
run/report/Slurm directories from the manifest instead of constructing them from
local naming assumptions.

Generated artifacts live under numbered report stages:
`01_train`, `02_collect`, `03_select`, `04_final_train`, and `05_final_eval`.
Run outputs are flat below each stage `outputs/` directory. The first run-id
component is `smoke` or `full`, so smoke runs remain distinguishable even when a
collector discovers runs by walking `resolved_config.yaml` files. Collectors
default to `full` runs; `--include-smoke` is reserved for smoke diagnostics.

Canonical train/eval configs for this study live under
`experiments/hooke/studies/pair_validation/configs/`. Copies under
`experiments/hooke/configs/benchmark/` are legacy test/reference configs, not
the study source of truth.

`collect.py`, `select.py`, `plan_final.py`, and `collect_final.py` process
files only.

## Manifest

The static manifest uses phase-local overrides.sweep entries. The validation
scan includes:

```yaml
runtime.seed: [3, 9, 11]
optimizer_params.lr: [3.0e-4, 1.0e-4, 1.0e-3, 3.0e-3]
model_params.channels: [8, 16, 32, 64]
model_params.layers: [1]
model_params.gate_activation: [silu, sigmoid, tanh]
```

Smoke runs are target jobs plus manifest smoke overlays. There are no separate
canonical smoke commands for this study layer. Slurm submissions are direct
`sbatch --array` launches; the orchestrator prints the array command and the
first concrete job command in `--dry-run` mode.

## Selection Rule

Selection groups runs by non-seed hyperparameters and treats `runtime.seed` as a
replicate. Failed, missing, or ineligible seeds count as `+inf` validation
energy. The primary metric is median `validation/energy/local_energy_mean`. Exact-reference
metrics such as `validation/energy/energy_abs_error` and `eval/energy/energy_abs_error` are
forbidden for selection.

When primary medians are inside the declared margin, tie-breakers apply in
manifest order: lower median `validation/energy/local_energy_variance`, lower validation energy
IQR, lower median stderr, fewer geometry warnings, smaller model, then shorter
wall time.

## Final Evaluation

`plan_final.py` creates `04_final_train/plans/final_train_jobs.jsonl`,
`05_final_eval/plans/smoke_eval_jobs.jsonl`, and
`05_final_eval/plans/final_eval_jobs.jsonl`. Final eval is row-based because
each row carries a specific checkpoint path. Final eval rows are paired
one-to-one, not Cartesian: completed final-train checkpoints are sorted by train
seed and zipped with `final_evaluation.eval_seeds`. Eval rows use:

```yaml
load:
  mode: model_only
  strict: true
  allow_protocol_mismatch: false
```

`collect_final.py` writes the final benchmark CSV/JSON/Markdown artifacts.

## Reproducibility

Keep the static manifest, generated job manifests, run directories, collection
tables, selected config, and final reports together. Slurm logs should be kept
with the same report directory whenever possible.
