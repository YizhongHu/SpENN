"""Diagnostic artifact indexing for evaluation runs."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


INDEX_SCHEMA_VERSION = 1


def update_diagnostic_index(
    *,
    run_dir: str | Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> Path:
    """Merge diagnostic artifact entries into ``diagnostics/index.json``.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Evaluation run directory.
    artifacts : sequence of mappings
        Artifact entries keyed by diagnostic artifact name. Unknown existing
        entries are preserved; entries with the same ``name`` are replaced.

    Returns
    -------
    pathlib.Path
        Path to the updated index file.
    """

    root = Path(run_dir)
    index_path = root / "diagnostics" / "index.json"
    existing = _read_index(index_path)
    by_name = {
        str(entry.get("name")): dict(entry)
        for entry in existing.get("artifacts", [])
        if isinstance(entry, Mapping) and entry.get("name")
    }
    for artifact in artifacts:
        entry = _artifact_entry(root, artifact)
        by_name[str(entry["name"])] = entry
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "run_dir": str(root),
        "artifacts": [by_name[name] for name in sorted(by_name)],
    }
    _write_json_atomic(index_path, payload)
    return index_path


def artifact_entry(
    *,
    run_dir: str | Path,
    name: str,
    kind: str,
    path: str | Path | None,
    created_by: str,
    enabled: bool = True,
    expected: bool = True,
    warning: str = "",
) -> dict[str, Any]:
    """Return one index-ready artifact entry."""

    return _artifact_entry(
        Path(run_dir),
        {
            "name": name,
            "kind": kind,
            "path": path,
            "created_by": created_by,
            "enabled": enabled,
            "expected": expected,
            "warning": warning,
        },
    )


def _artifact_entry(run_dir: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    name = str(artifact.get("name") or "").strip()
    if not name:
        raise ValueError("diagnostic artifact entries require a non-empty name")
    kind = str(artifact.get("kind") or "").strip() or _infer_kind(artifact.get("path"))
    path_value = artifact.get("path")
    relative_path = _relative_artifact_path(run_dir, path_value)
    absolute_path = None if relative_path is None else run_dir / relative_path
    exists = bool(absolute_path is not None and absolute_path.exists())
    readable = bool(exists and absolute_path is not None and os.access(absolute_path, os.R_OK))
    rows: int | None = None
    read_warning = ""
    if readable and absolute_path is not None:
        try:
            rows = _count_rows(absolute_path, kind)
        except Exception as exc:  # pragma: no cover - defensive filesystem edge
            readable = False
            read_warning = f"{type(exc).__name__}: {exc}"
    warning = str(artifact.get("warning") or read_warning or "")
    return {
        "name": name,
        "kind": kind,
        "path": None if relative_path is None else relative_path.as_posix(),
        "enabled": bool(artifact.get("enabled", True)),
        "expected": bool(artifact.get("expected", True)),
        "exists": exists,
        "readable": readable,
        "rows": rows,
        "created_by": str(artifact.get("created_by") or ""),
        "warning": warning,
    }


def _read_index(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": INDEX_SCHEMA_VERSION, "artifacts": []}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"schema_version": INDEX_SCHEMA_VERSION, "artifacts": []}
    if not isinstance(payload, Mapping):
        return {"schema_version": INDEX_SCHEMA_VERSION, "artifacts": []}
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    return {"schema_version": payload.get("schema_version", INDEX_SCHEMA_VERSION), "artifacts": artifacts}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def _relative_artifact_path(run_dir: Path, path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.relative_to(run_dir)
        except ValueError:
            return path
    return path


def _infer_kind(path_value: object) -> str:
    if path_value in (None, ""):
        return "unknown"
    suffix = Path(str(path_value)).suffix.lower().lstrip(".")
    return suffix or "unknown"


def _count_rows(path: Path, kind: str) -> int | None:
    if kind == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    if kind == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return None


__all__ = [
    "INDEX_SCHEMA_VERSION",
    "artifact_entry",
    "update_diagnostic_index",
]
