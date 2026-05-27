# Specht-module Equivariant Neural Network (SpENN)

Project plans are in `spenn_project_instructions.md`.

## Quick Start

Use `uv` for local environment management. For CPU version:
```bash
uv sync --extra cpu
```
For CUDA version
```bash
uv sync --extra cu130
```
or whatever cuda version you have

Run the phase 1 Hydra entrypoints with:

```bash
uv run python scripts/train.py
uv run python scripts/debug_equivariance.py
uv run python scripts/debug_local_energy.py
```

## Checks After Changes

After code changes, run the fast syntax and test checks:

```bash
uv run python -m compileall spenn scripts
uv run pytest -q
```

For changes that touch typed public APIs, data containers, or script
entrypoints, also run the opt-in runtime type checks:

```bash
uv run pytest --typeguard-packages=spenn -q
uv run python scripts/typechecked.py scripts/debug_equivariance.py -- sampler.n_walkers=4
```

## Runtime Type Checking

Runtime type checking is opt-in so normal training and debugging stay fast. The
policy is tracked in `configs/typecheck.yaml`.

Run tests with Typeguard instrumentation for `spenn`:

```bash
uv run pytest --typeguard-packages=spenn -q
```

Run an entrypoint with the Typeguard import hook installed before `spenn` is
imported:

```bash
uv run python scripts/typechecked.py scripts/debug_equivariance.py -- sampler.n_walkers=4
uv run python scripts/typechecked.py scripts/train.py -- trainer.max_steps=1 sampler.n_walkers=4
```

Phase 1 currently supports the hard-coded `M = 2` prototype with a Pfaffian
readout, batched Metropolis sampling, and an autograd-based local-energy path.
The active Hydra configuration lives in `configs/config.yaml` as one
constructor-oriented file.
