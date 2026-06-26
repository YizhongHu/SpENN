"""JSON and provenance-record IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def read_json(path: str | Path) -> Any:
    """Read JSON from ``path``."""

    return json.loads(Path(path).read_text())


def read_json_object(path: str | Path, warnings: list[str] | None = None) -> dict[str, Any]:
    """Read a JSON object, optionally recording missing/invalid input as warnings."""

    path = Path(path)
    if not path.is_file():
        if warnings is not None:
            warnings.append(f"missing JSON file: {path}")
            return {}
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if warnings is not None:
            warnings.append(f"{path}: invalid JSON: {exc}")
            return {}
        raise
    if not isinstance(payload, dict):
        message = f"{path}: expected JSON object"
        if warnings is not None:
            warnings.append(message)
            return {}
        raise ValueError(message)
    return payload


def read_json_object_list(
    path: str | Path,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Read a JSON list of objects, optionally recording problems as warnings."""

    path = Path(path)
    if not path.is_file():
        if warnings is not None:
            warnings.append(f"missing JSON file: {path}")
            return []
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if warnings is not None:
            warnings.append(f"{path}: invalid JSON: {exc}")
            return []
        raise
    if not isinstance(payload, list):
        message = f"{path}: expected JSON list"
        if warnings is not None:
            warnings.append(message)
            return []
        raise ValueError(message)
    return [item for item in payload if isinstance(item, dict)]


def path_from_record(record: dict[str, Any], key: str) -> Path | None:
    """Return an absolute path from a provenance record field."""

    raw = record.get(key)
    if raw in (None, ""):
        return None
    return Path(str(raw)).resolve()
