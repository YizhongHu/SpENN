# Specht-module Equivariant Neural Network (SpENN)

Project plans are in `spenn_project_instructions.md`.

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

Run the generic Hydra training entrypoint with:

```bash
uv run python train.py
```

After `uv sync`, a quick CPU smoke run is:

```bash
uv run python train.py --config=config.yaml training.vmc_steps=1 sampler.n_walkers=4 sampler.warmup_steps=1 sampler.production_blocks=1 sampler.steps_per_block=1 diagnostics.radial.n_points=8 diagnostics.cusp.n_points=4 diagnostics.exchange.n_samples=8 artifacts.write_checkpoint=false
```

Use the same command with a CUDA environment and append `device=cuda` for a GPU
smoke run.

## Checks After Changes

After code changes, run the fast syntax and test checks:

```bash
uv run python -m compileall spenn train.py typechecked.py
uv run pytest -q
```

For changes that touch typed public APIs, data containers, or entrypoints, also
run the opt-in entrypoint runtime type check:

```bash
uv run python typechecked.py train.py -- --config=config.yaml training.vmc_steps=1 sampler.n_walkers=4 sampler.warmup_steps=1 sampler.production_blocks=1 sampler.steps_per_block=1 diagnostics.radial.n_points=8 diagnostics.cusp.n_points=4 diagnostics.exchange.n_samples=8 artifacts.write_checkpoint=false
```

## Runtime Type Checking

Pytest installs Typeguard instrumentation for `spenn` by default. Entrypoint
runtime type checking is opt-in so normal training stays fast. The policy is
tracked in `configs/typecheck.yaml`.

Run tests with Typeguard instrumentation for `spenn`:

```bash
uv run pytest -q
```

Run an entrypoint with the Typeguard import hook installed before `spenn` is
imported:

```bash
uv run python typechecked.py train.py -- --config=config.yaml training.vmc_steps=1 sampler.n_walkers=4 sampler.warmup_steps=1 sampler.production_blocks=1 sampler.steps_per_block=1 diagnostics.radial.n_points=8 diagnostics.cusp.n_points=4 diagnostics.exchange.n_samples=8 artifacts.write_checkpoint=false
```

Phase 1 currently supports the hard-coded `M = 2` prototype with a Pfaffian
readout, batched Metropolis sampling, and an autograd-based local-energy path.
The active Hydra configuration lives in `configs/config.yaml` as one
constructor-oriented file.

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
