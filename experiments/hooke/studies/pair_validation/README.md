# Hooke pair validation study (`hooke_pair_validation_v1`)

A reproducible validation scan for selecting the Hooke pair model/protocol
before the final benchmark. The protocol is declared in [manifest.yaml](manifest.yaml);
once the scan starts, changing the manifest means a new study version.

The workflow:

```text
1. The user launches one training run per grid point (loop or SLURM array).
2. Each run executes train-end validation: an independent validation sampler
   draws fresh samples from the trained model and validation/* metrics are
   logged into the run directory.
3. collect.py normalizes the run directories into runs.csv / runs.jsonl.
4. select.py applies the manifest-declared rule and writes selection.csv,
   selected_config.yaml, and selection_report.md.
5. evaluate_selected.py turns the frozen winner into final benchmark commands
   (retrain on fresh seeds, then Evaluate runs against the exact reference);
   dry-run by default.
6. evaluate_selected.py --collect summarizes the finished final-eval runs into
   final_benchmark_summary.csv/json and final_benchmark_report.md.
```

Local run outputs are authoritative end to end. W&B may visualize the runs,
but none of the study scripts read W&B; W&B is visualization only and never
the source of a selection or benchmark decision.

## Quick test run (start here)

Before launching the real scan, confirm the whole pipeline
(train → validation → collect → select) works:

```bash
# local workstation / WSL (auto-detects GPU):
bash experiments/hooke/studies/pair_validation/test_run.sh

# FASRC SLURM, GPU:
sbatch -p gpu_test --gres=gpu:1 -c 4 -t 00:30:00 \
  experiments/hooke/studies/pair_validation/test_run.sh
```

```bash
# FASRC SLURM, CPU:
DEVICE=cpu sbatch -p test -c 4 -t 00:30:00 \
  experiments/hooke/studies/pair_validation/test_run.sh
```

It trains a tiny 2×2 grid at smoke scale into a scratch directory, runs the
collector and selector against it, checks every expected artifact and column,
and prints `TEST RUN PASSED` when everything works as expected.

## Launching the scan (user's duty)

Each grid point in the manifest is one training run of
[experiments/hooke/configs/benchmark/pair_train.yaml](../../configs/benchmark/pair_train.yaml)
with dotlist overrides. `run.py` takes one config per invocation (there is no
Hydra multirun in this repo), so expand the grid with a SLURM array (preferred
on FASRC) or a shell loop.

### SLURM array (preferred)

[launch_array.sh](launch_array.sh) maps `SLURM_ARRAY_TASK_ID` (0–53) onto the
54 grid points by mixed-radix decoding of the manifest axes, then launches one
training run per task. From the repo root:

```bash
mkdir -p slurm_logs   # keep these logs for reproducibility
sbatch experiments/hooke/studies/pair_validation/launch_array.sh
```

Defaults target `kozinsky_gpu` with one GPU per task; override partition or
device as needed, e.g. CPU on sapphire:

```bash
DEVICE=cpu sbatch -p sapphire --gres="" \
  experiments/hooke/studies/pair_validation/launch_array.sh
```

If the grid in the manifest changes, `launch_array.sh`'s axis arrays and
`--array` range must change with it — that is a new study version.

### Shell loop (small machines)

```bash
export UV_PROJECT_ENVIRONMENT=.venv-gpu  # GPU runs; omit for CPU (.venv)

ROOT=outputs/hooke_pair_validation_v1
for seed in 3 9 11; do
  for lr in 3e-4 1e-3 3e-3; do
    for channels in 8 32 128; do
      for gate in silu sigmoid; do
        uv run --extra cu126 python run.py \
          --config experiments/hooke/configs/benchmark/pair_train.yaml \
          run.root=$ROOT \
          study.name=hooke_pair_validation_v1 \
          runtime.seed=$seed \
          optimizer_params.lr=$lr \
          model_params.channels=$channels \
          model_params.gate_activation=$gate
      done
    done
  done
done
```

On the FASRC cluster, submit the same command shape as a SLURM array over the
54 grid points instead of a serial loop.

### W&B policy

Enable the W&B logger by uncommenting it in the benchmark config. Project
policy:

- test runs go to project `SpENN-QMC-test`;
- the real scan goes to project `SpENN-QMC` with tags `hooke-pair`, the
  accelerator (`cpu` or `cuda`), and `study-v1` (the benchmark config's
  `wandb:` block already sets these).

## Collecting results

```bash
uv run python experiments/hooke/studies/pair_validation/collect.py \
  --run-root outputs/hooke_pair_validation_v1 \
  --output-dir experiments/hooke/studies/pair_validation/results
```

The collector scans for run directories (any directory holding
`metadata.json`), reads `resolved_config.yaml`, `status.json`,
`metadata.json`, and `metrics.jsonl`, and writes one normalized row per run to
`runs.csv` and `runs.jsonl`. Failed and incomplete runs appear explicitly with
`status=failed` / `status=incomplete`. The collector never picks a winner.

Each row records the deterministic `config_id`
(e.g. `lr=0.001_channels=32_layers=1_gate_activation=silu`), derived from the
non-seed grid fields of the resolved config (or taken from `study.config_id`
when the launcher recorded one), plus the git sha and run directory.

## Selecting the winner

```bash
uv run python experiments/hooke/studies/pair_validation/select.py \
  --runs experiments/hooke/studies/pair_validation/results/runs.csv \
  --output-dir experiments/hooke/studies/pair_validation/results
```

The selection rule (declared in the manifest, applied by `select.py`):

```text
group runs by the non-seed hyperparameters
failed / ineligible / missing seeds count as +inf validation energy
any failed seed fails the whole config (selection.require_all_seeds)
rank by median validation/energy, but only outside the selection margin
candidates within the margin are tied and ranked by the tie-breakers
```

Eligibility requires `checks/data_integrity/passed`,
`checks/gradient/passed`, `checks/equivariance/full_model/passed`, and
`validation/local_energy_finite_fraction = 1.0`.

### Tie-breaker rule

Validation energies can sit within Monte Carlo uncertainty of each other, so
the selector never picks between configs based on tiny differences. Config A
clearly beats config B only when

```text
median_energy_A + selection_margin < median_energy_B

selection_margin = max(
  2 * sqrt(median_stderr_A^2 + median_stderr_B^2),
  0.25 * max(energy_iqr_A, energy_iqr_B),
  1.0e-4,   # absolute_energy_floor
)
```

where `median_stderr` is the median `validation/energy_stderr` over seeds and
`energy_iqr` is the seed-to-seed interquartile range of `validation/energy`.
Candidates within the margin of the best median energy are tied; the tie is
decided by the manifest tie-breakers, in order (lower always wins):

```text
1. median validation/energy_variance   - lower variance, better VMC wavefunction
2. validation/energy IQR across seeds  - lower spread, more robust optimization
3. median validation/energy_stderr     - lower estimator uncertainty
4. geometry warning count              - suspicious sampler geometry never
                                         decides the benchmark when energy ties
5. smaller model_params.channels       - prefer the simpler ansatz
6. median runtime/wall_time_sec        - last resort, not a scientific criterion
```

`selection_report.md` records the computed margin, the tie set, and which
tie-breaker decided the winner.

Outputs in `results/`: `selection.csv` (full ranking with margin/tie-breaker
statistics), `selected_config.yaml` (frozen winner + reproduction overrides +
the decision record), and `selection_report.md` (ranking, margin and
tie-breaker decisions, winner geometry table, and flags).

## Why validation never uses the exact reference energy

Validation exists to *choose* a model/protocol from independent samples.
Comparing against the exact Hooke energy during selection would leak the
benchmark answer into the choice and turn the final evaluation into a
formality. Validation therefore logs only estimator quality
(`validation/energy`, `validation/energy_variance`, `validation/energy_stderr`,
finite fractions, sampler metadata); `eval/energy_error`,
`eval/energy_abs_error`, and `eval/reference_energy` belong exclusively to
final evaluation, after the protocol is frozen. The `Validation` callback
rejects diagnostics configured with `reference_energy`.

## How final evaluation differs from validation

```text
validation  - runs at train_end inside each training job, fresh validation
              sampler, validation/* metrics, used for selection
final eval  - separate Evaluate run on the frozen selected protocol, fresh
              evaluation sampler, eval/* metrics, compares against the exact
              Hooke reference energy
```

## Final benchmark of the selected config

`evaluate_selected.py` turns the frozen winner into the final held-out
benchmark declared in the manifest's `final_evaluation` block. It is dry-run
by default: it writes commands and provenance without executing anything.

```bash
uv run python experiments/hooke/studies/pair_validation/evaluate_selected.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --selected-config experiments/hooke/studies/pair_validation/results/selected_config.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/results \
  --dry-run
```

This writes to `results/`:

```text
final_eval_commands.sh             - train + eval run.py commands, one pair per seed
final_eval_manifest.yaml           - frozen provenance (winner, seeds, sampler, git
                                     sha, checkpoint-loading mode, selection report
                                     path, exact-reference policy)
final_eval_inputs.csv              - one row per planned (training seed, eval seed)
                                     pair, with checkpoint/model-spec sources
final_eval_config_seed*_eval*.yaml - one self-contained generated eval config per
                                     pair (explicit model spec, checkpoint path,
                                     seeds, sampler, study identity all baked in)
```

The benchmark is two stages per seed pair, generated as standard `run.py`
commands (the script implements no physics itself; the `Evaluate` runner owns
all diagnostics):

```text
1. retrain the selected config with a fresh final training seed
   (pair_train.yaml + the selected overrides)
2. evaluate the trained checkpoint (checkpoints/latest.pt) by running the
   generated final_eval_config_seed<t>_eval<e>.yaml (derived from
   pair_final_eval.yaml): the Evaluate runner restores the weights
   (spenn.callback.checkpoint.load_model_checkpoint), samples with the large
   final-evaluation sampler (8192 walkers), and logs eval/* metrics including
   eval/energy_error against the exact reference
```

Final seeds are fresh: training seeds `100-109` and evaluation seeds
`100000-100009` are disjoint from the validation grid seeds, and the script
refuses to generate commands that reuse validation seeds unless the manifest
sets `final_evaluation.allow_validation_seed_reuse: true`.

### Checkpoint loading contract

A checkpoint pairs with an *explicit* model spec; architecture is never
inferred from state-dict keys. Training checkpoints use a structured schema
(`schema_version: 1`, written by `spenn.callback.Checkpoint`) that stores the
weights plus `model_config`, `model_config_hash` (sha256 of the canonical
resolved model config), `resolved_config_hash`, `config_id`, and
git/runtime/version provenance. Loading
(`spenn.callback.checkpoint.load_model_checkpoint`) is strict by default —
missing or unexpected keys fail loudly — and verifies
`evaluation.expected_model_config_hash` against the checkpoint when set; the
only escape hatch is the explicit
`evaluation.allow_model_config_mismatch: true`, which canonical benchmark
configs must not set.

`final_evaluation.checkpoint_loading` in the manifest picks how
`evaluate_selected.py` pairs eval configs with checkpoints:

```text
structured_checkpoint (default)
  Fresh final training runs write schema-v1 checkpoints. Each generated eval
  config carries the model spec the training command resolves to (training
  run's resolved_config.yaml when the run already exists, otherwise the same
  OmegaConf resolution the training command performs) and pins
  expected_model_config_hash, so loading verifies the pairing end to end.

legacy_resolved_config_workaround
  Only for pre-schema checkpoints of already-completed training runs (the
  current study's outputs do not need rerunning). The model spec is copied
  from the training run's resolved_config.yaml; loading stays strict, but no
  hash is verified because legacy payloads carry none. The mode is recorded
  in final_eval_manifest.yaml (checkpoint_loading.mode,
  model_config_hash_verified: false) and in each generated eval config; it
  must not become the long-term benchmark path.
```

Run the benchmark either locally (`--execute` runs the commands sequentially
and records `final_eval_runs.csv`) or through SLURM by submitting each command
from `final_eval_commands.sh` (keep the train -> eval dependency per seed,
e.g. `sbatch --dependency=afterok:<train_job>`). The training stage is
GPU-friendly (`kozinsky_gpu`, `seas_gpu`); keep the SLURM logs.

Once final-eval runs exist, summarize them:

```bash
uv run python experiments/hooke/studies/pair_validation/evaluate_selected.py \
  --manifest experiments/hooke/studies/pair_validation/manifest.yaml \
  --run-root outputs \
  --output-dir experiments/hooke/studies/pair_validation/results \
  --collect
```

which writes `final_benchmark_summary.csv`, `final_benchmark_summary.json`,
and `final_benchmark_report.md` (per-run eval energies, errors vs the exact
reference, and median aggregates) from the local run directories only.

## Sampler geometry diagnostics

`validation/sampler/*` includes walker-geometry summaries computed from the
validation walkers (`radius_*`, `electron_distance_*`, `position_*`,
`center_of_mass_rms`). For Hooke the sampler must remain confined by the
wavefunction envelope: growing `radius_q99`/`radius_max` means walkers are
escaping into large-radius tails, and a collapsing
`electron_distance_q01`/`electron_distance_min` means near-coalescence
sampling — both make energy estimates untrustworthy. `select.py` flags
missing/nonfinite geometry and `electron_distance_q01` below the manifest's
`geometry_flags.electron_distance_q01_min`, but geometry never decides the
winner unless the manifest lists it under eligibility or tie-breakers.

## Why DataIntegrity is separate from Validation

```text
DataIntegrity  - runtime soundness checks during training (finite local
                 energies, valid signs, well-formed batches), logged under
                 checks/data_integrity/*; a guardrail, not an estimator
Validation     - independent estimator on fresh samples used for
                 model/protocol selection, logged under validation/*
```

A run can be numerically sound yet a poor model (passes DataIntegrity, bad
validation energy), or a good model with a corrupted step (fails
DataIntegrity, which makes it ineligible for selection regardless of its
validation energy).

## Output files

Everything the study writes lands in `results/`:

```text
runs.csv / runs.jsonl            - collect.py: one normalized row per run
selection.csv                    - select.py: full margin-aware ranking
selected_config.yaml             - select.py: frozen winner + overrides
selection_report.md              - select.py: ranking, margin, tie-breakers, flags
final_eval_commands.sh           - evaluate_selected.py: train + eval commands
final_eval_manifest.yaml         - evaluate_selected.py: benchmark provenance
final_eval_inputs.csv            - evaluate_selected.py: planned run pairs
final_eval_config_*.yaml         - evaluate_selected.py: generated per-pair eval
                                   configs (explicit model spec + checkpoint pin)
final_eval_runs.csv              - evaluate_selected.py --execute: run statuses
final_benchmark_summary.csv/json - evaluate_selected.py --collect: per-run table
final_benchmark_report.md        - evaluate_selected.py --collect: report
```

## Reproducibility notes

- The manifest is the protocol contract: grid, eligibility, selection margin,
  tie-breakers, and final seeds. Changing it after the scan starts means a new
  study version (bump `study.name`).
- Every decision is recomputable from local files: re-running collect/select
  on the same run directories reproduces the same winner bit-for-bit, and
  `selection_report.md` documents why.
- Keep `slurm_logs/` and the committed `results/` tables; together with the
  recorded `git/sha` per run they pin what produced each number.
- Local run directories, `metrics.csv`, `metrics.jsonl`, `metadata.json`, and
  the generated summary files are authoritative. W&B is visualization only —
  never use W&B clicks as the source of the selection decision.

## Tests

Study-script tests live next to the scripts (experiments code is independent
of `spenn/`):

```bash
uv run pytest experiments/hooke/studies/pair_validation/tests -q
```
