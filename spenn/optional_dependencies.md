# Optional Dependencies

PyTorch is optional in package metadata so downstream installs, config parsing,
documentation, and non-execution tooling can import SpENN without a full Torch
install.

Training, sampling, neural-network modules, physics evaluation, diagnostics, and
runtime equivariance checks require a complete PyTorch install. Keep CPU and GPU
work in separate uv environments so concurrent Slurm jobs do not replace each
other's Torch install.

## CPU Environment

CPU work uses the default `.venv` and the `cpu` Torch extra:

```bash
uv sync --extra cpu
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

## GPU Environment

CUDA work uses a separate environment selected by `UV_PROJECT_ENVIRONMENT`:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-gpu
uv sync --extra cu126
uv run --extra cu126 python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

Use `cu128` or `cu130` instead if that is the CUDA Torch build you want. Keep
the `UV_PROJECT_ENVIRONMENT` setting in GPU Slurm scripts.

The public `run.py` entrypoint preflights configured Hydra targets. If a config
requires PyTorch and the active environment only has the lightweight optional
dependency set, it fails before Hydra instantiation with an actionable message:

```text
configured SpENN run requires a complete `torch` installation. Install it with `uv sync --extra cpu` or run with `uv run --extra cpu ...`.
```

Code that needs PyTorch should import it through `spenn.dependencies`:

```python
from spenn.dependencies import require_torch, require_torch_nn

torch = require_torch(feature="my SpENN feature")
nn = require_torch_nn(feature="my SpENN feature")
```

Use `require_torch_functional(...)` for `torch.nn.functional`. This keeps the
optional dependency policy centralized and avoids scattered `try`/`except`
boilerplate.
