"""Backend-neutral logging primitives."""

from __future__ import annotations

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



__all__ = ["LogRecord", "Logger"]
