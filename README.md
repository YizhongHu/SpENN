# Specht-module Equivariant Neural Network (SpENN)

The active SpENN core scaffold is documented in the PR brief and the package
docstrings under `spenn/data`, `spenn/reps`, `spenn/nn`, and `spenn/equivariance`.

## Quick Start

Use `uv` for local environment management. The default CPU environment is
`.venv`. GPU work uses a separate `.venv-gpu` so CUDA Torch does not replace the
CPU Torch install. Both environments still resolve from this one `pyproject.toml`.

For CPU work:

```bash
uv sync --extra cpu
```

For CUDA work:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-gpu
uv sync --extra cu126
```

Use `cu128` or `cu130` instead if that is the CUDA Torch build you want.

Core tests are the active validation target:

```bash
uv run pytest -q
```

Configured runs go through the single `run.py` entrypoint, which launches one
`spenn.runner.Runner` from a YAML config. The Hooke pair smoke training config
is a working example:

```bash
uv run python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

For a syntax-only check:

```bash
uv run python -m compileall spenn run.py typechecked.py
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
uv run python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
wandb sync --sync-all
```

By default, SpENN does not upload checkpoints, traces, raw batches, per-sample
arrays, or full run directories to W&B. W&B receives scalar metrics and compact
config/provenance metadata; CSV/JSONL logs and local artifacts remain canonical.

## Terminal and SLURM Status

Terminal output is a human-facing status view, not the authoritative metric
store. Configured runs can enable line-oriented Python logging for local
terminals and SLURM `.out` files:

```yaml
terminal:
  enabled: true
  level: info
  color: auto      # auto | always | never
  rich: auto       # reserved; plain logging remains the fallback
  progress: false  # keep false for SLURM-compatible logs
```

Use the `Status` callback for compact lifecycle and progress lines:

```yaml
callbacks:
  - _target_: spenn.callback.Status
    triggers:
      - run_start
      - run_end
      - exception
    output_path: ${run.dir}/status.json

  - _target_: spenn.callback.Status
    triggers:
      - step_end
    every_n_steps: 10
    include:
      - train/loss
      - train/energy
      - train/energy_stderr
      - train/sampler/acceptance_rate
      - train/grad_norm
      - train/local_energy_finite_fraction
```

Status lines are grep-friendly, for example:

```text
[run] started id=... dir=... device=cpu dtype=float64 git=... dirty=false host=...
[train] step=10 loss=0.421 energy=2.104 stderr=0.031 acc=0.61 grad=0.012 finite=1
[run] completed dir=...
```

`run.py`, trainers, models, samplers, diagnostics, and loggers do not print
training metrics directly. CSV/JSONL remain the canonical local metric records.
For SLURM jobs, prefer unbuffered output:

```bash
export PYTHONUNBUFFERED=1
uv run python -u run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

Regenerate checked-in Specht irrep cache files from SageMath with:

```bash
uv run python -m spenn.reps.fixture_generators.sage_specht \
  --sage-executable /n/sw/sage-10.3/sage \
  --max-order 3 \
  --out-json spenn/cache/irreps.json \
  --out-cache spenn/cache/irreps_m3.pt
```

## Eager Model Invariant

SpENN feature layouts are unchanged.
All trainable parameters are registered during __init__.
Forward may allocate activations but not trainable parameters.
Sampler never materializes models.

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

## Config ownership

**Callbacks and loggers are config-root and owned by the `RunContext`.** They
live at the top level, *not* inside the runner block. A runner config that
declares `callbacks` or `loggers` is rejected by `run_from_config`:

```yaml
runner:
  _target_: spenn.runner.Train
  model: ${model}
  sampler: ${sampler}
  hamiltonian_terms: ${hamiltonian_terms}
  optimizer: ${optimizer}
  trainer: ${trainer}

callbacks: [...]   # config-root, RunContext-owned
loggers: [...]     # config-root, RunContext-owned
```

- `model`, `sampler`, `hamiltonian_terms`, `optimizer`, and `trainer` are
  reusable top-level blocks referenced by the runner via `${...}`.
- `hamiltonian_terms` may be either a sequence or a mapping. Mapping keys are
  the public term names used for decompositions and metrics, so they must be
  non-empty strings; sequence entries are named from snake-case term class names
  with index suffixes added for repeats. Every term object must expose
  `local_energy(wavefunction, batch)` and return a `LocalEnergyResult` whose
  `total` has shape `[batch]`. With `return_terms=True`, configured mapping
  keys are preserved as the public decomposition names.
- `optimizer` names a partial factory (`_partial_: true`) that builds an
  optimizer from model parameters; `Train` applies it to `model.parameters()`.
- `spenn.runner.Train` runs the VMC training loop. `spenn.runner.Evaluate` is a
  sampled diagnostic evaluator (`model`, `sampler`, `hamiltonian_terms`,
  `diagnostics`, `return_terms`). Reference-energy comparison belongs to
  evaluation diagnostics such as `spenn.diagnostics.EnergyEvaluation`.
- Training and evaluation term metrics use `energy_term_<name>` for the finite
  mean and suffixes such as
  `_variance`, `_std`, `_stderr`, `_n_finite`, `_n_total`, `_finite_fraction`,
  and `_nonfinite_count` for companion statistics.
- Metric identity is `namespace + key`: training metrics use `train`, sampler
  stats use `train/sampler` or `eval/sampler`, runtime checks use `checks/...`,
  and evaluation diagnostics use `eval`. See `spenn/metrics_naming.md`.

`prepare_run_context` instantiates the config-root `callbacks`/`loggers` into the
`RunContext`. `Runner.emit(...)` dispatches lifecycle events through
`context.callbacks`, runners log through `context.log(...)`, and
`logger.finish()` runs against the context's loggers. `VMCTrainer` owns only
training-loop hyperparameters (`max_steps`, `log_every_n_steps`, `return_terms`,
`gradient_clip_norm`) and the loss/backward/step mechanics; it does not own
callbacks, loggers, reference energy, or diagnostics.

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
