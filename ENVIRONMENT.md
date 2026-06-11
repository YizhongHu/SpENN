# Environment

Use `uv` with its normal defaults. The project keeps one guard in
`pyproject.toml` to prevent system Python from `/usr/bin` from being selected.

To enable multiple jobs running in the same directory, quarantine the CPU and GPU
envs. CPU takes the default `.venv` while GPU takes `.venv-gpu`.

## CPU

```bash
uv sync --extra cpu
uv run --extra cpu python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## GPU

Choose the CUDA wheel index that matches the node and driver.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-gpu
uv sync --extra cu126
uv run --extra cu126 python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The `cpu`, `cu126`, `cu128`, and `cu130` extras are mutually exclusive by design.
