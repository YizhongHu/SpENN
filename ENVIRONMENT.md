# Environment

Use `uv` with its normal defaults. The project keeps one guard in
`pyproject.toml` to prevent system Python from `/usr/bin` from being selected.

## Install

```bash
uv sync
```

## CPU

```bash
uv sync --extra cpu
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## GPU

Choose the CUDA wheel index that matches the node and driver.

```bash
uv sync --extra cu128
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If the GPU node needs the older CUDA 12.6 wheel set instead:

```bash
uv sync --extra cu126
```

The `cpu`, `cu126`, and `cu128` extras are mutually exclusive by design.
