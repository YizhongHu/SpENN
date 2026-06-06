"""Training-loop namespace with lazy public exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "NullCallback",
    "TrainerConfig",
    "VMCTrainer",
    "gradient_norm",
    "parameter_norm",
    "run_config",
]


def __getattr__(name: str) -> Any:
    """Load training exports only when requested."""

    if name == "NullCallback":
        from spenn.training.callbacks import NullCallback

        return NullCallback
    if name in {"gradient_norm", "parameter_norm"}:
        from spenn.training import metrics

        return getattr(metrics, name)
    if name == "run_config":
        from spenn.training.run import run_config

        return run_config
    if name in {"TrainerConfig", "VMCTrainer"}:
        from spenn.training.trainer import TrainerConfig, VMCTrainer

        return {"TrainerConfig": TrainerConfig, "VMCTrainer": VMCTrainer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
