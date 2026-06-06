"""Metric helpers and local logger implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LogRecord:
    """One metric logging record."""

    step: int | None
    namespace: str
    metrics: dict[str, Any]
    event: str | None = None


class Logger:
    """Base logger interface for configured runs."""

    def log(self, record: LogRecord) -> None:
        """Log one metric record."""

        raise NotImplementedError

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """Optionally log an artifact path."""

        pass

    def finish(self) -> None:
        """Flush and close logger resources."""

        pass


class CSV(Logger):
    """Append scalar metrics to a simple CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output CSV path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def log(self, record: LogRecord) -> None:
        """Append one row per scalar metric."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            if not self._wrote_header:
                handle.write("step,namespace,key,value\n")
                self._wrote_header = True
            for key, value in record.metrics.items():
                step = "" if record.step is None else str(record.step)
                handle.write(f"{step},{record.namespace},{key},{_csv_value(value)}\n")


class JSONL(Logger):
    """Append metric records as JSON lines.

    Parameters
    ----------
    path : str or pathlib.Path
        Output JSONL path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, record: LogRecord) -> None:
        """Append one JSON object."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": record.step,
            "namespace": record.namespace,
            "event": record.event,
            "metrics": record.metrics,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
            handle.write("\n")


def parameter_norm(model):
    """Return the L2 norm of trainable initialized parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are inspected.

    Returns
    -------
    torch.Tensor
        Scalar parameter norm. Returns zero when no initialized trainable
        parameters are present.
    """

    import torch
    from torch.nn.parameter import UninitializedParameter

    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if isinstance(param, UninitializedParameter):
            continue
        value = param.detach().pow(2).sum()
        total = value if total is None else total + value
    return torch.sqrt(total) if total is not None else torch.tensor(0.0)


def gradient_norm(model):
    """Return the L2 norm of available gradients.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameter gradients are inspected.

    Returns
    -------
    torch.Tensor
        Scalar gradient norm. Returns zero when no gradients are present.
    """

    import torch
    from torch.nn.parameter import UninitializedParameter

    total = None
    for param in model.parameters():
        if param.grad is None:
            continue
        if isinstance(param, UninitializedParameter):
            continue
        value = param.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    return torch.sqrt(total) if total is not None else torch.tensor(0.0)


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


__all__ = [
    "CSV",
    "JSONL",
    "LogRecord",
    "Logger",
    "gradient_norm",
    "parameter_norm",
]
