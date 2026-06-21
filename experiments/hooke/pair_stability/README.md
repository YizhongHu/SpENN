# Hooke pair-stability study (PR8.8)

A pair-stability scan over **model-side inductive bias** (input basis + Gaussian
envelope) and **feature normalization**, for the two-electron Hooke system. It
is the first real experiment package built on the post-PR8.5 evaluation stack
and adds no new evaluation-runner abstractions.

```
architecture choice  x  normalization choice  x  lr  x  channels  x  seed
```

- **architecture choice** = a curated input basis + its hyperparameters + a
  Gaussian output envelope + tags. It is one axis, not a `basis x envelope`
  Cartesian product: only meaningful bundles are enumerated.
- **normalization choice** = where feature normalization is inserted in the
  SpENN model (`N0`-`N4`).

## Model pipeline

```
ElectronBatch
  -> ElectronBasis            (model-side equivariant featurization)
  -> ElectronBasisFeatures    (typed; one_body per-particle features)
  -> Embedding                (in_features derived from the selected basis)
  -> SpENN feature layers
  -> readout
  -> Gaussian envelope
```

The basis re-represents the per-particle input; the raw `ElectronBatch` still
flows to the readout and envelope so they see true coordinates. The embedding
input width is wired from the selected basis via the `spenn.basis_feature_dim`
OmegaConf resolver (`in_features: ${spenn.basis_feature_dim:${model.basis}}`),
so no per-variant YAML is needed.

### Architecture choices

Main scan variants (all include the Gaussian envelope):

| choice                 | basis                                   |
|------------------------|-----------------------------------------|
| `raw_envelope`         | `RawCoordinateBasis` (coords + spin)    |
| `hermite_o2_envelope`  | `HookeHermiteBasis(max_order=2)`        |
| `hermite_o3_envelope`  | `HookeHermiteBasis(max_order=3)`        |
| `orbital_s1_envelope`  | `HookeOrbitalBasis(max_shell=1)`        |
| `orbital_s2_envelope`  | `HookeOrbitalBasis(max_shell=2)`        |

Optional diagnostic choices (not in the default grid): `hermite_o4_envelope`,
`orbital_s3_envelope`.

Every main variant uses `HookeGaussianEnvelope` so the output-side asymptotic
prior is shared across architectures. The analytic electron-electron cusp is
held constant. No-envelope variants (`raw_no_envelope`, `hermite_no_envelope`,
`orbital_no_envelope`) are intentionally excluded; raw coordinates without an
envelope are known not to converge.

`HookeHermiteBasis` uses Hermite/oscillator polynomials **without** the Gaussian
factor (the clean match for an output-envelope model). `HookeOrbitalBasis` uses
oscillator orbital shapes (`include_gaussian_factor=true`); under a common
output envelope this is mildly double-normalized by design, to compare input
representations on equal footing.

### Normalization choices

`IrrepRMSNorm` is a parameter-free RMS norm over the channel axis (equivariant).
The mode selects where it is inserted:

| mode  | id   | semantics                                        |
|-------|------|--------------------------------------------------|
| `none`              | N0 | no normalization module is inserted            |
| `post_embedding`    | N1 | `h = norm(embedding(features))`                |
| `post_feature_layer`| N2 | `h = norm(layer(h))` after each feature layer  |
| `update`            | N3 | `delta = norm(update(h)); h = h + delta`       |
| `pre_readout`       | N4 | `output = readout(norm(h))`                    |

One mode is scanned at a time.

## Configs

- `configs/pair_stability.yaml` — training base. The public override surface is
  scalar: `run_parameters.{architecture,normalization,lr,channels,seed}`.
  `model.basis`, `model.envelope`, and `model.feature_normalization` resolve
  from the structured choice libraries (`choices.architecture`,
  `choices.normalization`) using the selected keys.
- `configs/pair_validation.yaml` — `EvaluationTask`-based validation/evaluation
  base. Restores a trained checkpoint and runs physical local-energy probes
  (`cusp`, `tail`, `stratified_geometry`, `hooke_orbital`), full-model
  antisymmetry, trace equivariance, and feature/readout trace-stability tasks.
  It does not include exact/reference energy comparison. The
  architecture/normalization must match the trained run.
- `configs/grid.yaml` — the `architecture x normalization x lr x channels x seed`
  grid.

## Submission

This repo's run entrypoint (`run.py`) is a plain OmegaConf launcher, **not** a
`@hydra.main` app, so there is no Hydra Submitit command path to reuse and no
study-specific `sbatch` code is added. The workflow is split into strict stage
entrypoints:

- `plan.py` writes the `00_grid` attempt (manifest + `commands.sh`) and submits
  nothing.
- `train.py` reads an existing `00_grid` attempt and launches its train
  commands into `01_train` with `--backend local` or `--backend submitit`. It
  does not expand grids or rewrite the `00_grid` manifest.
- `validate.py` reads `00_grid`, consumes selected `01_train` attempts, writes
  `source_train_attempt.json`, and launches validation into `02_validation`.
- `launch.py` is shared by `train.py` and `validate.py`; it owns local/Submitit
  execution, uv sync/activation, CPU/CUDA profile defaults, and Slurm resources.

This runbook assumes the real scan runs on CUDA through Submitit. The CLI keeps
CPU as the default for safety, so production launch examples pass `--cuda`
explicitly.

`train.py` and `validate.py` share the same execution profile. `--cpu` and
`--cuda` switch all three execution layers together:

| profile  | uv environment | uv extra | runtime override | Submitit hardware default |
|----------|----------------|----------|------------------|---------------------------|
| `--cpu`  | `.venv`        | `cpu`    | `runtime.device=cpu`  | `slurm_partition=seas_compute,kozinsky_lab,sapphire`, no GPUs |
| `--cuda` | `.venv-gpu`    | `cu126`  | `runtime.device=cuda` | `slurm_partition=seas_gpu,kozinsky_gpu`, `gpus_per_node=1` |

Each launched job syncs and activates the selected environment, then runs the
planned command through that environment's `python`. Override the environment
path with `--uv-environment`; pass `--uv-extra` one or more times to select
another extra such as `cu128` or `cu130`.

Submitit launches are always Slurm arrays via `submitit.AutoExecutor.map_array`,
not one independent `sbatch` per planned run. The default full-run array cap is
16 simultaneous array tasks (`--slurm-array-parallelism 16`); smoke runs cap at
2. By default `--chunk-size 1`, so each planned run is one array task. Larger
chunk sizes group multiple planned runs into one array task, and the launcher
balances chunks evenly rather than leaving a small tail. For example, 540 runs
with `--chunk-size 128` call for 5 chunks; instead of `128 + 128 + 128 + 128 +
28`, each array task receives `540 / 5 = 108` runs.

The planner is the source of truth for the study timezone (`--timezone`, default
`America/New_York`): it stamps attempt ids and the manifest `created_at`, and
always injects it as a `run.timezone` override on the compiled commands. The
configs set `run.timezone: null`, so a planned run takes its zone only from
`plan.py` (a direct `run.py` run with no override falls back to spenn's `UTC`
default).

```bash
# Plan the grid (dry run): writes results/00_grid/<attempt_id>/
uv run python experiments/hooke/pair_stability/plan.py

# Plan only the "main"-tagged architectures
uv run python experiments/hooke/pair_stability/plan.py --tags main
```

### Train launch options

Train smoke before the real scan:

```bash
# CUDA Submitit smoke: two jobs, gpu_test partition, 15 minute limit
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cuda --smoke

# CPU Submitit smoke: two jobs, test partition, 15 minute limit
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cpu --smoke

# Local smoke, useful on an interactive node
uv run python experiments/hooke/pair_stability/train.py \
  --backend local --cuda --smoke
```

`--smoke` submits only the first two planned grid jobs, appends `-smoke` to the
train attempt id, and overlays short-run settings (`training.max_steps=2`,
128 walkers, short burn-in/chain lengths, and checkpoint/status every step).

Standard CUDA Submitit launch after smoke passes:

```bash
# Submit the latest 00_grid attempt on the GPU partition
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cuda

# Submit a specific 00_grid attempt
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cuda \
  --grid-attempt-id 20260619T195112-0400
```

Other supported execution modes:

```bash
# CUDA local run, for an interactive GPU node or tiny smoke run
uv run python experiments/hooke/pair_stability/train.py \
  --backend local --cuda

# CPU local run, the CLI default profile
uv run python experiments/hooke/pair_stability/train.py \
  --backend local --cpu

# CPU Submitit run on a CPU partition
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cpu
```

Environment and Slurm overrides:

```bash
# Use a different CUDA Torch build
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cuda \
  --uv-extra cu128

# Use a different GPU partition
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cuda \
  --slurm-partition seas_gpu

# Run at most four array tasks at a time
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cuda \
  --slurm-array-parallelism 4

# Group multiple planned runs into each array task
uv run --extra submitit python experiments/hooke/pair_stability/train.py \
  --backend submitit --cuda \
  --chunk-size 8
```

### Validation launch options

Smoke validation after the train smoke:

```bash
# CUDA Submitit validation smoke: first two jobs, gpu_test, 15 minute limit
uv run --extra submitit python experiments/hooke/pair_stability/validate.py \
  --backend submitit --cuda --smoke

# CPU Submitit validation smoke: first two jobs, test, 15 minute limit
uv run --extra submitit python experiments/hooke/pair_stability/validate.py \
  --backend submitit --cpu --smoke
```

`validate.py --smoke` looks for smoke-marked train attempts, writes smoke-marked
validation attempts, and overlays small evaluation grids. Real validation does
not auto-select smoke train attempts.

Standard CUDA Submitit validation after training finishes:

```bash
# Validate the latest non-smoke train attempts for the latest 00_grid attempt
uv run --extra submitit python experiments/hooke/pair_stability/validate.py \
  --backend submitit --cuda \
  --chunk-size 128

# Validate an exact train attempt and write an exact validation attempt id
uv run --extra submitit python experiments/hooke/pair_stability/validate.py \
  --backend submitit --cuda \
  --grid-attempt-id 20260619T195112-0400 \
  --train-attempt-id 20260619T195112-0400 \
  --attempt-id 20260620T090000-0400 \
  --chunk-size 128
```

Each planned grid point becomes scalar overrides, e.g.:

```
run_parameters.architecture=hermite_o3_envelope
run_parameters.normalization=N2
run_parameters.lr=0.001
run_parameters.channels=16
run_parameters.seed=0
run.root=experiments/hooke/pair_stability/results/01_train
run.layout=flat
run.run_id=<run_id>/<attempt_id>
run.timezone=America/New_York   # always injected; --timezone selects the zone
```

The execution profile adds `runtime.device=cpu` or `runtime.device=cuda` when
launching. With the flat run layout, `run.dir = run.root / run.run_id`, which
realizes the staged attempt directory.

## Staged results layout

```
results/
  00_grid/        defines planned jobs               (manifest + commands)
  01_train/       consumes 00_grid job specs         (training attempts)
  02_validation/  consumes selected 01_train attempts (evaluation attempts)
  03_collect/     consumes 02_validation attempts     (summary tables)
  04_select/      consumes 03_collect summaries        (champions)
```

Artifact inheritance chain (each stage records exactly which earlier artifact it
consumed; provenance uses explicit attempt ids, never `latest`):

```
00_grid attempt
   -> 01_train/{run_id}/{attempt_id}
        -> 02_validation/{run_id}/{attempt_id}/source_train_attempt.json
             -> 03_collect/{attempt_id}/source_validation_attempts.json
                  -> 04_select/{attempt_id}/source_collection_attempt.json
```

Every directory under a stage (or under a stage's run id) is an attempt, so
there is no intermediate `attempts/` path segment.

Rerunnable units are indexed by attempt ids of the form
`YYYYMMDDTHHMMSS-0400` in the planner's timezone (America/New_York by
default), which is also injected as the `run.timezone` override, so attempt ids
and run logs share one wall clock. Detailed layout:

```
results/
  00_grid/
    {attempt_id}/
      manifest.json          # planned/submitted jobs (the durable run list)
      commands.sh            # exact run.py commands
      grid.yaml              # snapshot of the grid
      pair_stability.yaml    # snapshot of the train config
      pair_validation.yaml   # snapshot of the validation config
      jobs/{run_id}.json     # per-job spec
    latest.json -> {attempt_id}
  01_train/{run_id}/{attempt_id}/   # source_grid_attempt.json, submission.json, config.yaml, checkpoints/, ...
  02_validation/{run_id}/{attempt_id}/
      source_train_attempt.json     # train attempt + checkpoint consumed
      cusp/ tail/ stratified_geometry/ hooke_orbital/   # local-energy probes
      full_model_antisymmetry/ trace_equivariance/      # transform/trace checks
      feature_trace_stability/ readout_trace_stability/ # internal stability checks
      diagnostics/index.json, status.json, metrics.*
  03_collect/{attempt_id}/          # summary.csv, failures.csv, collection_report.json, source_*.json
  04_select/{attempt_id}/           # champions.csv, selection_report.json, source_collection_attempt.json
```

### Manifest

`00_grid/.../manifest.json` is the durable record of planned jobs. Each job
records its `run_id`, `train_dir`, `validation_dir`, the exact scalar
`overrides`, the `command`, the resolved `choices`, and the architecture `tags`.
Submission fields are initialized but not updated there; `train.py` records
launch provenance under each `01_train/{run_id}/{attempt_id}/`, and
`validate.py` records validation launch provenance under
`02_validation/{run_id}/{attempt_id}/`.

## Collect and select

```bash
# Collect the latest validation attempt per run id into a 03_collect attempt
uv run python experiments/hooke/pair_stability/collect.py

# Select one champion per architecture by the study's local-energy hierarchy
uv run python experiments/hooke/pair_stability/select_champions.py
```

`collect.py` walks `02_validation`, reads each attempt's status, evaluation
metrics (`metrics.jsonl`), and `source_train_attempt.json`, and writes
`summary.csv` / `failures.csv` plus collection provenance. `select_champions.py`
reads a `03_collect` summary and writes per-architecture champions and a
selection report, recording the collection attempt it consumed. The default
selector compares local-energy means with their standard-error bars in this
order: `stratified_geometry`, `tail`, `cusp`, `hooke_orbital`. If the current
leader's error bar overlaps another row's error bar, those rows remain tied and
the next local-energy task breaks the tie. If the hierarchy is exhausted, the
shortest available wall time wins.

The study scripts (`plan.py`, `train.py`, `validate.py`, `collect.py`,
`select_champions.py`) share their stage-layout vocabulary,
attempt-id/timezone helpers, run-id grammar, JSON IO, and staged-directory path
helpers through `run_utils.py`; `train.py` and `validate.py` additionally share
execution mechanics through `launch.py`.

## Tests

- Reusable model/component math is tested under `tests/unit/model/`
  (`test_electron_basis.py`, `test_feature_normalization.py`,
  `test_pair_stability_config_choices.py`).
- Study orchestration and file layout are tested in `test_pair_stability.py`
  (grid/choice consistency, attempt-id timezone, the `attempt_ids` listing, the
  `run.timezone` override, planner manifest, strict train orchestration,
  staged layout, attempt provenance, and a one-grid-point smoke run through the
  normal run path).

## Relationship to `pair_validation`

The archival `experiments/hooke/pair_validation/` study is not canonical. This
package reuses its orchestration/collection patterns (grid expansion, scalar
overrides, run-id conventions, summary tables) but replaces its
diagnostics-based evaluation with the `EvaluationTask` stack, drops the
phase/required-era semantics, and uses explicit task `output_dir`s and staged
attempt directories throughout.
