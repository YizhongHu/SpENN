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

Phase 1 currently supports the hard-coded `M = 2` prototype with a Pfaffian
readout, batched Metropolis sampling, and an autograd-based local-energy path.
The active Hydra configuration lives in `configs/config.yaml` as one
constructor-oriented file.
