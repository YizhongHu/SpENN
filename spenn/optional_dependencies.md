# Optional Dependencies

PyTorch is optional in package metadata so downstream installs, config parsing,
documentation, and non-execution tooling can import SpENN without a full Torch
install.

The repository's default `uv run` development environment includes the local
`cpu` dependency group, which installs the CPU Torch build so smoke training
configs run without extra command-line flags:

```bash
uv run python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

Training, sampling, neural-network modules, physics evaluation, diagnostics, and
runtime equivariance checks require a complete PyTorch install. Use one of the
Torch extras before running those paths from a minimal install:

```bash
uv sync --extra cpu
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

Use `cu126`, `cu128`, or `cu130` instead of `cpu` when running against a CUDA
Torch build in a GPU environment. The local default `cpu` group conflicts with
CUDA Torch extras, so disable it explicitly for CUDA runs:

```bash
uv run --no-group cpu --extra cu128 python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

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
