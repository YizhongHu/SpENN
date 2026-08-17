"""Numbered stage layout for the He-v1 production study.

The study writes one directory per stage, one attempt directory per stage
invocation, and one row directory per manifest row. Nothing is ever rewritten
in place and nothing is ever deleted: a re-plan or a re-launch creates a new
attempt beside the old one, so a receipt that cites an attempt id keeps
pointing at the bytes it was written from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STAGE_PLAN = "00_plan"
STAGE_LAUNCH = "01_launch"
STAGE_TRAIN = "02_train"
STAGE_EVAL = "03_eval"
STAGE_COLLECT = "04_collect"
STAGE_REPORT = "05_report"

MANIFEST_FILENAME = "manifest.json"
ROWS_FILENAME = "rows.csv"
LATEST_FILENAME = "latest.json"


def stage_dir(results_root: str | Path, stage: str) -> Path:
    """Return the directory of one numbered stage."""

    return Path(results_root) / stage


def attempt_dir(results_root: str | Path, stage: str, attempt_id: str) -> Path:
    """Return one stage attempt directory."""

    _require_attempt_id(attempt_id)
    return stage_dir(results_root, stage) / attempt_id


def plan_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``00_plan`` attempt directory."""

    return attempt_dir(results_root, STAGE_PLAN, attempt_id)


def manifest_path(results_root: str | Path, attempt_id: str) -> Path:
    """Return the manifest path of one plan attempt."""

    return plan_attempt_dir(results_root, attempt_id) / MANIFEST_FILENAME


def launch_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``01_launch`` attempt directory."""

    return attempt_dir(results_root, STAGE_LAUNCH, attempt_id)


def row_dir(results_root: str | Path, stage: str, row_id: str, attempt_id: str) -> Path:
    """Return the durable directory of one row inside one attempt.

    The attempt id is the leaf rather than the parent so every artifact of one
    row -- across plan attempts -- stays under that row's own directory.
    """

    _require_row_id(row_id)
    _require_attempt_id(attempt_id)
    return stage_dir(results_root, stage) / row_id / attempt_id


def collect_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``04_collect`` attempt directory."""

    return attempt_dir(results_root, STAGE_COLLECT, attempt_id)


def report_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``05_report`` attempt directory."""

    return attempt_dir(results_root, STAGE_REPORT, attempt_id)


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    """Read one JSON document."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_latest(stage_path: str | Path, attempt_id: str) -> Path:
    """Record the latest attempt id under one stage directory.

    The pointer is advisory. Every stage still accepts an explicit attempt id,
    because a receipt that says "latest" says nothing a month later.
    """

    _require_attempt_id(attempt_id)
    return write_json(Path(stage_path) / LATEST_FILENAME, {"attempt_id": str(attempt_id)})


def read_latest(stage_path: str | Path) -> str | None:
    """Return the recorded latest attempt id, or ``None`` when unrecorded."""

    pointer = Path(stage_path) / LATEST_FILENAME
    if not pointer.is_file():
        return None
    payload = read_json(pointer)
    attempt_id = payload.get("attempt_id") if isinstance(payload, dict) else None
    return str(attempt_id) if attempt_id else None


def resolve_attempt_id(results_root: str | Path, stage: str, requested: str | None) -> str:
    """Return the requested attempt id, or the recorded latest one.

    Raises
    ------
    FileNotFoundError
        If no attempt id was requested and none was recorded. Guessing an
        attempt would silently collect a different study than the one asked
        for.
    """

    if requested:
        return str(requested)
    latest = read_latest(stage_dir(results_root, stage))
    if latest is None:
        raise FileNotFoundError(
            f"no attempt id given and no {LATEST_FILENAME} recorded under "
            f"{stage_dir(results_root, stage)}"
        )
    return latest


def _require_attempt_id(attempt_id: str) -> None:
    text = str(attempt_id).strip()
    if not text:
        raise ValueError("attempt id must be a non-empty string")
    if "/" in text or text in {".", ".."}:
        raise ValueError(f"attempt id is not a single path component: {attempt_id!r}")


def _require_row_id(row_id: str) -> None:
    text = str(row_id).strip()
    if not text:
        raise ValueError("row id must be a non-empty string")
    if "/" in text or text in {".", ".."}:
        raise ValueError(f"row id is not a single path component: {row_id!r}")


__all__ = [
    "LATEST_FILENAME",
    "MANIFEST_FILENAME",
    "ROWS_FILENAME",
    "STAGE_COLLECT",
    "STAGE_EVAL",
    "STAGE_LAUNCH",
    "STAGE_PLAN",
    "STAGE_REPORT",
    "STAGE_TRAIN",
    "attempt_dir",
    "collect_attempt_dir",
    "launch_attempt_dir",
    "manifest_path",
    "plan_attempt_dir",
    "read_json",
    "read_latest",
    "report_attempt_dir",
    "resolve_attempt_id",
    "row_dir",
    "stage_dir",
    "write_json",
    "write_latest",
]
