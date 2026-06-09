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

The legacy Hooke experiment configs were removed from this restructuring
branch. Core tests remain the active validation target:

```bash
uv run pytest -q
```

For a syntax-only check:

```bash
uv run python -m compileall spenn train.py typechecked.py
```

Regenerate checked-in Specht irrep cache files from SageMath with:

```bash
uv run python -m spenn.reps.fixture_generators.sage_specht \
  --sage-executable /n/sw/sage-10.3/sage \
  --max-order 3 \
  --out-json spenn/cache/irreps.json \
  --out-cache spenn/cache/irreps_m3.pt
```

## Config ownership

Top-level config blocks are reusable object definitions. They are instantiated
only when referenced by the selected runner or by run-context setup:

- `callbacks` and `loggers` are **run-context owned** — the run context
  instantiates and drives them; runners dispatch into them via events/logging.
- `model`, `sampler`, `hamiltonian_terms`, `optimizer_factory`, and `trainer`
  are **runner-wired** — they are inert unless the selected runner references
  them explicitly (e.g. `spenn.runner.Train` takes `optimizer_factory:
  ${optimizer_factory}` and `trainer: ${trainer}`).
- `diagnostics` are **Evaluate-owned** and are not consumed by `Train`.

`optimizer_factory` is named to reflect that the configured object is a factory
(`_partial_: true`) used to build an optimizer from model parameters, not an
optimizer instance. `VMCTrainer` owns training-loop hyperparameters
(`max_steps`, `log_every_n_steps`, `return_terms`, `gradient_clip_norm`) and the
loss/backward/step mechanics; it does not own callbacks, loggers, reference
energy, or diagnostics.

## Checks After Changes

After code changes, run the fast syntax and test checks:

```bash
uv run python -m compileall spenn train.py typechecked.py
uv run pytest -q
```

## Runtime Type Checking

Pytest installs Typeguard instrumentation for `spenn` by default.

Run tests with Typeguard instrumentation for `spenn`:

```bash
uv run pytest -q
```

Equivariance checks are runtime checks on `spenn.data.EquivariantMap`. When
enabled, small systems are checked against every particle permutation; larger
systems are checked against adjacent transpositions and reversal. Tests force
checks with `check_probability=1.0`.

Tensor state validation is a typed, per-object contract. `RealFeature`,
`RealInteraction`, `RealUpdate`, `IrrepInteraction`, `IrrepFeature`, and
`IrrepUpdate` each expose a `validate()` method that validates their own
semantic fields. There is no generic tree-validation helper: validation,
particle permutation (`permute`), and comparison (`compare`) are declared by
typed data objects, never inferred by recursively probing arbitrary containers.
A compound value that needs validation should be a typed container (e.g.
`ConcatenatedState`) whose own `validate()` delegates to its children.

Exact testing strategy:

- Permutation convention and algebra:
  `spenn.data.permutation.Permutation`,
  `spenn.data.indices.permute_tuple_slots`, and
  `tests/equivariance/test_permutation.py`.
- State actions:
  `spenn.data.equivariant_state.EquivariantState`,
  `spenn.data.real.RealFeature`, `RealInteraction`, `RealUpdate`,
  `spenn.data.batch.WavefunctionOutput`, and tests in
  `tests/equivariance/test_equivariant_state.py`,
  `tests/equivariance/test_real_feature.py`,
  `tests/equivariance/test_real_interaction.py`, and
  `tests/equivariance/test_real_update.py`.
- Runtime equivariance checks:
  `spenn.equivariance.checks.FullModelEquivarianceChecker` and
  `TraceEquivarianceChecker` (driven by `spenn.callback.RuntimeEquivariance`),
  using `apply_particle_permutation` and typed `.compare(...)`. Pytest-only
  assertion helpers live under `tests/helpers/equivariance.py`, with coverage in
  `tests/equivariance/test_equivariant_map.py`.
- Tensor shape checks:
  `RealFeature`, `RealInteraction`, and `RealUpdate` are dense order-indexed
  lists of tensors. Index 0 is reserved for zero-order data and must have zero
  channels; use `spenn.data.real.zero_block` to construct that sentinel. Irrep
  tensors are keyed directly by `spenn.data.partition.Partition`, whose
  `order` defines the tuple order. Validation coverage lives in
  `tests/unit/data/test_tensor_validation.py`.
- Layer-level checks:
  `spenn.nn.Update`, `spenn.nn.Activation`, `spenn.nn.ActivationByType`,
  `spenn.nn.PathAggregation`, and `spenn.nn.SpENNLayer`, with forced runtime
  checks in `tests/equivariance/test_update.py`,
  `tests/equivariance/test_activation.py`,
  `tests/equivariance/test_path_aggregation.py`, and
  `tests/equivariance/test_spenn_layer_scaffold.py`.
- Virtual-support combinatorics:
  `spenn.reps.paths.PathMetadata`, `generate_virtual_paths`, and
  `validate_virtual_path`, with coverage in
  `tests/equivariance/test_virtual_paths.py`.

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
- `spenn.nn`: `EquivariantMixing`, `Activation`, `ActivationByType`,
  `PathAggregation`, `Update`, `ChannelMappedUpdate`, `SpENNLayer`,
  `SpENNWaveFunction`, and readouts under `spenn.nn.readout`.
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