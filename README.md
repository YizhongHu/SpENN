# Specht-module Equivariant Neural Network (SpENN)

The active SpENN core scaffold is documented in the PR brief and the package
docstrings under `spenn/data`, `spenn/reps`, `spenn/nn`, and `spenn/equivariance`.

## Quick Start

Use `uv` for local environment management. Keep CPU and GPU work in separate
virtual environments so Slurm jobs never replace each other's Torch install.
Both environments resolve from this one `pyproject.toml`.

### CPU Environment

CPU work uses the default `.venv`:

```bash
uv sync --extra cpu
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

### GPU Environment

CUDA work uses a separate `.venv-gpu`:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-gpu
uv sync --extra cu126
uv run --extra cu126 python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

Use `cu128` or `cu130` instead if that is the CUDA Torch build you want. Keep
the `UV_PROJECT_ENVIRONMENT` setting in GPU Slurm scripts so GPU jobs do not
mutate the CPU `.venv`.

Core tests are the active validation target:

```bash
uv run --extra cpu pytest -q
```

Configured runs go through the single `run.py` entrypoint, which launches one
`spenn.runner.Runner` from a YAML config. The Hooke pair smoke training config
is a working example:

```bash
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

Human-readable run timestamps are controlled by `run.timezone`, an IANA
timezone name. The code default is `UTC`; the Hooke smoke config uses
`America/New_York` so run IDs, `metadata.json`, `status.json`, and terminal
status boxes share the same cluster-log convention.

For a syntax-only check:

```bash
uv run --extra cpu python -m compileall spenn run.py typechecked.py
```

## Optional W&B Tracking

SpENN can optionally mirror scalar run metrics to Weights & Biases for
dashboarding and monitoring. W&B is an observability backend only; the local run
directory remains the authoritative experiment record.

Install optional W&B support:

```bash
uv sync --extra wandb
```

For interactive authentication:

```bash
wandb login
```

For non-interactive jobs:

```bash
export WANDB_API_KEY=<your-api-key>
```

Add W&B as another root-level logger:

```yaml
loggers:
  - _target_: spenn.logging.CSV
    path: ${run.dir}/metrics.csv
  - _target_: spenn.logging.JSONL
    path: ${run.dir}/metrics.jsonl
  - _target_: spenn.logging.WandB
    project: spenn-qmc
    entity: null
    mode: online
    group: hooke_pair
    tags:
      - hooke
      - vmc
```

For jobs without reliable internet, use W&B offline mode:

```bash
wandb offline
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
wandb sync --sync-all
```

By default, SpENN does not upload checkpoints, traces, raw batches, per-sample
arrays, or full run directories to W&B. W&B receives scalar metrics and compact
config/provenance metadata; CSV/JSONL logs and local artifacts remain canonical.


## Config section types

SpENN configs use two main kinds of sections:

```text
component specs
parameter blocks
```

### Component specs

Component specs describe Python objects that Hydra should instantiate.

They usually contain `_target_`.

Examples:

```yaml
sampler:
  _target_: spenn.sampling.MetropolisSampler
  n_walkers: 16
  n_electrons: 2
  spatial_dim: 3

optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-3
```

### Parameter blocks

Parameter blocks are config-only namespaces. They are not instantiated as Python objects.

They exist to collect readable, user-facing values that component specs can reference.

The parameter blocks make the user-facing knobs easy to find:

```yaml
training:
  batch_size: 16
  n_steps: 100
  learning_rate: 1.0e-3
```

The component specs use those knobs through interpolation:

```yaml
sampler:
  n_walkers: ${training.batch_size}

optimizer:
  lr: ${training.learning_rate}

trainer:
  n_steps: ${training.n_steps}
```

## Timing Metrics

Timing instrumentation is callback-owned and logs through the same CSV/JSONL
logger path as other metrics:

```yaml
callbacks:
  - _target_: spenn.callback.RunTiming

  - _target_: spenn.callback.TrainStepTiming
    every_n_steps: 1
    rolling_window: 20
    cuda_synchronize: false

  - _target_: spenn.callback.EvaluationTiming
    cuda_synchronize: false

  - _target_: spenn.callback.DiagnosticTiming
    cuda_synchronize: false
```
GPU synchronization is opt-in with `cuda_synchronize: true` for
benchmarking; it is disabled by default for normal training.

## SageMath

Regenerate checked-in Specht irrep cache files from SageMath with:

```bash
uv run python -m spenn.reps.fixture_generators.sage_specht \
  --sage-executable /n/sw/sage-10.3/sage \
  --max-order 3 \
  --out-json spenn/cache/irreps.json \
  --out-cache spenn/cache/irreps_m3.pt
```

## Eager Model Invariant

SpENN model construction owns trainable state. All trainable parameters must be
registered during ``__init__`` from explicit architecture metadata such as
channel counts and maximum order. A forward pass may allocate activations whose
tuple axes depend on the runtime particle count, including zero-sized axes, but
it must not create, resize, replace, move, or cast parameters or buffers.

The sampler is never involved in model construction. There is no materialization
batch in the eager design: setup moves the model to the configured device and
dtype, the optimizer is built from the complete parameter set, and forward only
evaluates the already-constructed model.

Current tensor layouts are:

```text
RealFeature order m:
  [batch, channels, i1, ..., im]

IrrepInteraction:
  [batch, channels, paths, indices..., alpha, beta_in]

IrrepFeature:
  [batch, channels, indices..., alpha, beta]
```


## Checks After Changes

After code changes, run the fast syntax and test checks:

```bash
uv run python -m compileall spenn run.py typechecked.py
uv run pytest -q
```

## Runtime Type Checking

Pytest installs Typeguard instrumentation for `spenn` by default.

Run tests with Typeguard instrumentation for `spenn`:

```bash
uv run pytest -q
```

## Equivariance Checking

Equivariance checks are runtime checks on `spenn.equivariance.EquivariantMap`.
When enabled, small systems are checked against every particle permutation;
larger systems are checked against adjacent transpositions and reversal. Configs
force checks with `probability: 1.0` on the `RuntimeEquivariance` callback.

Runtime validation is a typed, per-object contract kept **separate** from
equivariance. `RealFeature`, `RealInteraction`, `IrrepFeature`,
`IrrepInteraction`, and `ElectronBatch` each expose a `validate()` method (and,
where useful, `validity_metrics()`) that checks their own semantic fields; the
static contracts live in `spenn.data.validation` (`RuntimeValidatable`,
`RuntimeValidityMetrics`). This is deliberately distinct from
`spenn.data.equivariant_state.EquivariantState`, which declares only particle
permutation (`permute`) and comparison (`compare`). There is no generic
tree-validation, tree-permutation, or particle-count-inference helper:
validation, permutation, and comparison are declared by typed data objects,
never inferred by recursively probing arbitrary containers.

Exact testing strategy:

- Permutation convention and algebra:
  `spenn.data.permutation.Permutation`,
  `spenn.data.indices.permute_tuple_slots`, and
  `tests/unit/data/test_permutation.py`.
- State actions:
  `spenn.data.equivariant_state.EquivariantState`,
  `spenn.data.real.RealFeature`, `RealInteraction`, `RealUpdate`,
  `spenn.data.batch.WavefunctionOutput`, and tests in
  `tests/unit/data/test_equivariant_state.py`,
  `tests/unit/data/test_real_feature.py`,
  `tests/unit/data/test_real_interaction.py`, and
  `tests/unit/data/test_real_update.py`.
- Runtime equivariance checks:
  `spenn.equivariance.checks.FullModelEquivarianceChecker` and
  `TraceEquivarianceChecker` (driven by `spenn.callback.RuntimeEquivariance`),
  using `apply_particle_permutation` and typed `.compare(...)`. Pytest-only
  assertion helpers live under `tests/helpers/equivariance.py`, with coverage in
  `tests/unit/equivariance/test_equivariant_map.py`.
- Tensor shape checks:
  `RealFeature`, `RealInteraction`, and `RealUpdate` are dense order-indexed
  lists of tensors. Index 0 is reserved for zero-order data and must have zero
  channels; use `spenn.data.real.zero_block` to construct that sentinel. Irrep
  tensors are keyed directly by `spenn.data.partition.Partition`, whose
  `order` defines the tuple order. Validation coverage lives in
  `tests/unit/data/test_tensor_validation.py`.
- Layer-level checks:
  `spenn.nn.Update`, `spenn.nn.Activation`, `spenn.nn.PathAggregation`, and
  `spenn.nn.SpENNLayer`, with forced runtime
  checks in `tests/unit/nn/test_update_equivariance.py`,
  `tests/unit/nn/test_activation_equivariance.py`,
  `tests/unit/nn/test_path_aggregation_equivariance.py`, and
  `tests/unit/nn/test_spenn_layer_scaffold.py`.
- Virtual-support combinatorics:
  `spenn.reps.paths.PathMetadata`, `generate_virtual_paths`, and
  `validate_virtual_path`, with coverage in
  `tests/unit/reps/test_virtual_paths.py`.

For `n_particles <= 5`, the runtime schedule is exhaustive over all
permutations. Larger-particle tests use deterministic random inputs and random
permutations in addition to the runtime generator schedule.

The new core scaffold is direct, not a compatibility layer:

- `spenn.data`: common state names are exported at the package root for
  convenience, while helpers stay with their owner modules:
  `spenn.data.batch`, `spenn.data.real`, `spenn.data.irrep`,
  `spenn.data.partition`, `spenn.data.permutation`, and `spenn.data.indices`.
  Electron-batch geometry helpers live under `spenn.data.batch`.
- `spenn.reps`: virtual path metadata, irrep metadata, Sage-backed fixture
  generation, and cache-backed Fourier transforms.
- `spenn.nn`: `EquivariantMixing`, `GatedNormActivation`, `PathAggregation`,
  `ResidualUpdate`, `SpENNLayer`, `SpENNWaveFunction`, and readouts under
  `spenn.nn.readout`.
- `spenn.equivariance`: traceable `EquivariantMap`, passive trace recording, and
  runtime equivariance checkers (`spenn.equivariance.checks`).

## Documentation

Documentation sources live under `docs/` and use Sphinx with NumPy-style
docstrings via Numpydoc. The docs tooling is in the opt-in `docs` dependency
group, so normal installs do not include it.

Build the local HTML docs with:

```bash
uv run --extra cpu --group docs sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`, or serve them locally:

```bash
uv run --extra cpu python -m http.server --directory docs/_build/html 8000
```

## Versioning

The backwards compatibility of this repository is only with respect to the behavior
of Hydra config files. Before v1.0.0, every minor version can break backwards compatibility.
v0.2.0 does not have to be able to reproduce a v0.1.0 config. But patches have to be
compatible with each other.

## Conventions

### Metrics Naming Scheme

Metric naming and logger conventions are documented in
[`spenn/metrics_naming.md`](spenn/metrics_naming.md).
