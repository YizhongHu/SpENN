"""Provenance ancestry tracing for pair-stability staged artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from run_utils import (
    STAGE_COLLECT,
    STAGE_FINAL_COLLECT,
    STAGE_FINAL_EVAL,
    STAGE_FINAL_GRID,
    STAGE_FINAL_REPORT,
    STAGE_FINAL_TRAIN,
    STAGE_GRID,
    STAGE_SELECT,
    stage_dir,
)


@dataclass(frozen=True)
class Ancestry:
    """Traced result roots for a final report attempt."""

    roots: frozenset[Path]
    warnings: tuple[str, ...]


def trace_final_report_ancestry(results_root: str | Path, report_attempt_id: str) -> Ancestry:
    """Trace the result directories consumed by ``09_final_report/{attempt}``."""

    results_root = Path(results_root).resolve()
    roots: set[Path] = set()
    warnings: list[str] = []
    report_dir = stage_dir(results_root, STAGE_FINAL_REPORT) / report_attempt_id
    _add_dir(roots, warnings, report_dir, "final report")

    report_json = _read_json_dict(report_dir / "final_report.json", warnings)
    collect_attempt_id = str(report_json.get("final_collect_attempt_id") or "")
    if not collect_attempt_id:
        warnings.append(f"{report_dir / 'final_report.json'}: missing final_collect_attempt_id")
        return Ancestry(frozenset(roots), tuple(warnings))

    collect_dir = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    _add_dir(roots, warnings, collect_dir, "final collect")

    run_index = _read_csv(collect_dir / "run_index.csv")
    final_run_ids = [str(row.get("final_run_id", "")) for row in run_index if row.get("final_run_id")]
    final_eval_attempts = _final_eval_attempts_from_collect_manifest(collect_dir / "manifest.yaml", final_run_ids)
    final_eval_dirs = _resolve_final_eval_dirs(results_root, final_eval_attempts)
    for eval_dir in final_eval_dirs:
        _trace_final_eval(eval_dir, roots, warnings)

    return Ancestry(frozenset(roots), tuple(warnings))


def _trace_final_eval(eval_dir: Path, roots: set[Path], warnings: list[str]) -> None:
    if not _add_dir(roots, warnings, eval_dir, "final eval"):
        return
    _trace_source_final_grid(eval_dir / "source_final_grid_attempt.json", roots, warnings)
    source_train = _read_json_dict(eval_dir / "source_final_train_attempt.json", warnings)
    train_dir = _path_from_record(source_train, "final_train_attempt_dir")
    if train_dir is not None:
        _trace_final_train(train_dir, roots, warnings)


def _trace_final_train(train_dir: Path, roots: set[Path], warnings: list[str]) -> None:
    if not _add_dir(roots, warnings, train_dir, "final train"):
        return
    _trace_source_final_grid(train_dir / "source_final_grid_attempt.json", roots, warnings)


def _trace_source_final_grid(path: Path, roots: set[Path], warnings: list[str]) -> None:
    source_grid = _read_json_dict(path, warnings)
    grid_dir = _path_from_record(source_grid, "final_grid_attempt_dir")
    if grid_dir is None:
        return
    if not _add_dir(roots, warnings, grid_dir, "final grid"):
        return
    source_selection = _read_json_dict(grid_dir / "source_selection_attempt.json", warnings)
    select_dir = _path_from_record(source_selection, "selection_attempt_dir")
    if select_dir is not None:
        _trace_selection(select_dir, roots, warnings)


def _trace_selection(select_dir: Path, roots: set[Path], warnings: list[str]) -> None:
    if not _add_dir(roots, warnings, select_dir, "selection"):
        return
    source_collection = _read_json_dict(select_dir / "source_collection_attempt.json", warnings)
    collection_attempt_id = str(source_collection.get("collection_attempt_id") or "")
    if not collection_attempt_id:
        return
    collect_stage = select_dir.parents[1] / STAGE_COLLECT
    collect_dir = collect_stage / collection_attempt_id
    _trace_collection(collect_dir, roots, warnings)


def _trace_collection(collect_dir: Path, roots: set[Path], warnings: list[str]) -> None:
    if not _add_dir(roots, warnings, collect_dir, "collection"):
        return
    sources = _read_json_list(collect_dir / "source_validation_attempts.json", warnings)
    for source in sources:
        validation_dir = _path_from_record(source, "validation_attempt_dir")
        if validation_dir is not None:
            _trace_validation_grid_only(validation_dir, roots, warnings)


def _trace_validation_grid_only(validation_dir: Path, roots: set[Path], warnings: list[str]) -> None:
    """Read validation provenance without adding train/validation copy roots."""

    validation_dir = validation_dir.resolve()
    if not validation_dir.is_dir():
        warnings.append(f"missing validation directory for provenance: {validation_dir}")
        return
    source_train = _read_json_dict(validation_dir / "source_train_attempt.json", warnings)
    grid_attempt_id = str(source_train.get("grid_attempt_id") or "")
    if grid_attempt_id:
        grid_dir = validation_dir.parents[2] / STAGE_GRID / grid_attempt_id
        _add_dir(roots, warnings, grid_dir, "grid")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_dict(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.is_file():
        warnings.append(f"missing JSON file: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: invalid JSON: {exc}")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"{path}: expected JSON object")
        return {}
    return payload


def _read_json_list(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        warnings.append(f"missing JSON file: {path}")
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: invalid JSON: {exc}")
        return []
    if not isinstance(payload, list):
        warnings.append(f"{path}: expected JSON list")
        return []
    return [item for item in payload if isinstance(item, dict)]


def _final_eval_attempts_from_collect_manifest(path: Path, final_run_ids: Sequence[str]) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"final collect manifest is required for lineage: {path}")
    manifest = OmegaConf.load(path)
    fixed_attempt_id = _config_text(manifest, "final_eval_attempt_id")
    if fixed_attempt_id:
        return {final_run_id: fixed_attempt_id for final_run_id in final_run_ids}
    raw_mapping = OmegaConf.select(manifest, "final_eval_attempts", default=None)
    if raw_mapping is None:
        raise ValueError(
            f"{path}: missing final_eval_attempt_id/final_eval_attempts; "
            "rerun final_collect.py with current code or pass --final-eval-attempt-id"
        )
    mapping = OmegaConf.to_container(raw_mapping, resolve=True)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: final_eval_attempts must be a mapping of final_run_id to attempt id")
    attempts = {str(run_id): str(attempt_id) for run_id, attempt_id in mapping.items() if attempt_id not in (None, "")}
    missing = [final_run_id for final_run_id in final_run_ids if final_run_id not in attempts]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"{path}: final_eval_attempts missing final_run_id entries: {preview}{suffix}")
    return {final_run_id: attempts[final_run_id] for final_run_id in final_run_ids}


def _resolve_final_eval_dirs(
    results_root: Path,
    final_eval_attempts: dict[str, str],
) -> list[Path]:
    dirs = []
    for final_run_id, attempt_id in final_eval_attempts.items():
        run_dir = stage_dir(results_root, STAGE_FINAL_EVAL) / final_run_id
        attempt_dir = run_dir / attempt_id
        if not attempt_dir.is_dir():
            raise FileNotFoundError(f"manifested final-eval attempt does not exist: {attempt_dir}")
        dirs.append(attempt_dir)
    return dirs


def _path_from_record(record: dict[str, Any], key: str) -> Path | None:
    raw = record.get(key)
    if raw in (None, ""):
        return None
    return Path(str(raw)).resolve()


def _add_dir(roots: set[Path], warnings: list[str], path: Path, label: str) -> bool:
    path = path.resolve()
    if path.is_dir():
        if path in roots:
            return False
        roots.add(path)
        return True
    warnings.append(f"missing {label} directory: {path}")
    return False


def _config_text(config: Any, dotted_key: str) -> str | None:
    value = OmegaConf.select(config, dotted_key, default=None)
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "None", "none", "null"}:
        return None
    return text
