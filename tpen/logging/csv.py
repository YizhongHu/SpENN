"""CSV metric logger."""

from __future__ import annotations

from pathlib import Path

from .base import LogRecord, Logger


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
                if isinstance(value, bool):
                    value_text = "true" if value else "false"
                elif value is None:
                    value_text = ""
                else:
                    value_text = str(value)
                handle.write(f"{step},{record.namespace},{key},{value_text}\n")



__all__ = ["CSV"]
