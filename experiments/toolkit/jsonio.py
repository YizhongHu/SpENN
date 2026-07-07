"""Small JSON/JSONL helpers for durable experiment plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def to_jsonable(value: Any) -> Any:
    """Return a JSON-compatible copy of ``value``."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file of objects."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL rows must be objects")
            rows.append(row)
    return rows
