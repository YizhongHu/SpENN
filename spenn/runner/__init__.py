"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

from .base import Runner, _runtime_dtype


def __getattr__(name: str) -> object:
    """Load torch-dependent runner targets only when they are requested."""

    if name == "Evaluate":
        from .evaluate import Evaluate

        return Evaluate
    if name == "Train":
        from .train import Train

        return Train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Evaluate", "Runner", "Train"]
