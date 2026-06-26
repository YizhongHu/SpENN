"""JSON lines metric logger."""

from __future__ import annotations

import json
from pathlib import Path

from .base import LogRecord, Logger


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



__all__ = ["JSONL"]
