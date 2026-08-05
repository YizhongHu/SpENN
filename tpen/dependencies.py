"""Optional runtime dependency helpers."""

from __future__ import annotations

import importlib
from typing import Any


class OptionalDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is required but unavailable."""


def require_module(module: str, *, feature: str, extra: str | None = None) -> Any:
    """Import an optional module or raise a clear installation error."""

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise OptionalDependencyError(_optional_dependency_message(module, feature, extra=extra)) from exc


def require_torch(*, feature: str) -> Any:
    """Return a complete PyTorch module for features that execute tensor code."""

    torch = require_module("torch", feature=feature, extra="cpu")
    try:
        importlib.import_module("torch.nn")
    except ImportError as exc:
        raise OptionalDependencyError(_optional_dependency_message("torch", feature, extra="cpu")) from exc
    if getattr(torch, "__file__", None) is None and getattr(torch, "__version__", None) is None:
        raise OptionalDependencyError(_optional_dependency_message("torch", feature, extra="cpu"))
    return torch


def require_torch_nn(*, feature: str) -> Any:
    """Return ``torch.nn`` for neural-network features."""

    require_torch(feature=feature)
    return importlib.import_module("torch.nn")


def require_torch_functional(*, feature: str) -> Any:
    """Return ``torch.nn.functional`` for neural-network features."""

    require_torch(feature=feature)
    return importlib.import_module("torch.nn.functional")


def _optional_dependency_message(module: str, feature: str, *, extra: str | None) -> str:
    if extra is None:
        return f"{feature} requires optional dependency `{module}`."
    return (
        f"{feature} requires a complete `{module}` installation. "
        f"Install it with `uv sync --extra {extra}` or run with `uv run --extra {extra} ...`."
    )


__all__ = [
    "OptionalDependencyError",
    "require_module",
    "require_torch",
    "require_torch_functional",
    "require_torch_nn",
]
