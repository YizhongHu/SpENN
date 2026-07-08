"""Numbered stage layout and latest-pointer helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json, write_json

STAGE_GRID = "00_grid"
STAGE_TRAIN = "01_train"
STAGE_VALIDATION = "02_validation"
STAGE_COLLECT = "03_collect"
STAGE_SELECT = "04_select"
STAGE_FINAL_GRID = "05_final_grid"
STAGE_FINAL_TRAIN = "06_final_train"
STAGE_FINAL_EVAL = "07_final_eval"
STAGE_FINAL_COLLECT = "08_final_collect"
STAGE_FINAL_REPORT = "09_final_report"
ATTEMPT_METADATA = "attempt_metadata.json"


def stage_dir(results_root: str | Path, stage: str) -> Path:
    """Return the directory for a numbered stage."""

    return Path(results_root) / stage


def grid_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``00_grid`` attempt directory."""

    return stage_dir(results_root, STAGE_GRID) / attempt_id


def train_run_dir(results_root: str | Path, run_id: str) -> Path:
    """Return the per-run-id directory under ``01_train``."""

    return stage_dir(results_root, STAGE_TRAIN) / run_id


def validation_run_dir(results_root: str | Path, run_id: str) -> Path:
    """Return the per-run-id directory under ``02_validation``."""

    return stage_dir(results_root, STAGE_VALIDATION) / run_id


def final_grid_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``05_final_grid`` attempt directory."""

    return stage_dir(results_root, STAGE_FINAL_GRID) / attempt_id


def final_train_run_dir(results_root: str | Path, final_run_id: str) -> Path:
    """Return the per-final-run-id directory under ``06_final_train``."""

    return stage_dir(results_root, STAGE_FINAL_TRAIN) / final_run_id


def final_eval_run_dir(results_root: str | Path, final_run_id: str) -> Path:
    """Return the per-final-run-id directory under ``07_final_eval``."""

    return stage_dir(results_root, STAGE_FINAL_EVAL) / final_run_id


def train_attempt_dir(results_root: str | Path, run_id: str, attempt_id: str) -> Path:
    """Return the train attempt directory for a run id."""

    return train_run_dir(results_root, run_id) / attempt_id


def validation_attempt_dir(results_root: str | Path, run_id: str, attempt_id: str) -> Path:
    """Return the validation attempt directory for a run id."""

    return validation_run_dir(results_root, run_id) / attempt_id


def final_train_attempt_dir(results_root: str | Path, final_run_id: str, attempt_id: str) -> Path:
    """Return the final-train attempt directory for a final run id."""

    return final_train_run_dir(results_root, final_run_id) / attempt_id


def final_eval_attempt_dir(results_root: str | Path, final_run_id: str, attempt_id: str) -> Path:
    """Return the final-eval attempt directory for a final run id."""

    return final_eval_run_dir(results_root, final_run_id) / attempt_id


def attempt_ids(parent: str | Path) -> list[str]:
    """Return sorted attempt-id directory names directly under ``parent``."""

    parent = Path(parent)
    if not parent.is_dir():
        return []
    return sorted(
        child.name
        for child in parent.iterdir()
        if child.is_dir() and child.name != "latest"
    )


def _latest_payload(parent: str | Path, *, filename: str = "latest.json") -> dict[str, Any] | None:
    """Return one latest-pointer payload when present."""

    latest = Path(parent) / filename
    if not latest.is_file():
        return None
    payload = read_json(latest)
    return payload if isinstance(payload, dict) else None


def read_latest_attempt_id(parent: str | Path, *, filename: str = "latest.json") -> str | None:
    """Return the latest-pointer attempt id under ``parent`` when present."""

    payload = _latest_payload(parent, filename=filename)
    attempt_id = None if payload is None else payload.get("attempt_id")
    return str(attempt_id) if attempt_id else None


def attempt_metadata(parent: str | Path, attempt_id: str) -> dict[str, Any]:
    """Return metadata recorded for one stage attempt."""

    metadata_path = Path(parent) / str(attempt_id) / ATTEMPT_METADATA
    if not metadata_path.is_file():
        return {}
    metadata = read_json(metadata_path)
    return metadata if isinstance(metadata, dict) else {}


def _valid_latest_payload(parent: Path, payload: dict[str, Any]) -> bool:
    """Return whether a latest-pointer payload points at an attempt directory."""

    attempt_id = str(payload.get("attempt_id") or "")
    return bool(attempt_id) and (parent / attempt_id).is_dir()


def latest_attempt_id(parent: str | Path) -> str | None:
    """Return the preferred latest attempt id under ``parent``."""

    parent = Path(parent)
    payload = _latest_payload(parent, filename="latest.json")
    if payload is not None and _valid_latest_payload(parent, payload):
        return str(payload["attempt_id"])
    candidates = attempt_ids(parent)
    return candidates[-1] if candidates else None


def _write_attempt_metadata(stage_path: Path, attempt_id: str) -> None:
    """Record attempt lineage metadata independent of its name."""

    write_json(
        stage_path / str(attempt_id) / ATTEMPT_METADATA,
        {"attempt_id": str(attempt_id)},
    )


def _write_latest_pointer(stage_path: Path, filename: str, attempt_id: str) -> None:
    """Write one portable latest pointer."""

    write_json(
        stage_path / filename,
        {"attempt_id": str(attempt_id), "path": str(attempt_id)},
    )


def _write_latest_symlink(stage_path: Path, link_name: str, attempt_id: str) -> None:
    """Best-effort latest symlink update."""

    link = stage_path / link_name
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(attempt_id, target_is_directory=True)
    except OSError:
        pass


def _write_primary_latest(stage_path: Path, attempt_id: str) -> None:
    """Write the primary latest pointer and symlink."""

    _write_latest_pointer(stage_path, "latest.json", attempt_id)
    _write_latest_symlink(stage_path, "latest", attempt_id)


def write_latest(stage_path: Path, attempt_id: str) -> None:
    """Record latest attempt ids under ``stage_path``."""

    stage_path = Path(stage_path)
    _write_attempt_metadata(stage_path, attempt_id)
    _write_primary_latest(stage_path, attempt_id)
