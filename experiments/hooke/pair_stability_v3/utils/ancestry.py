"""Provenance ancestry tracing for staged study artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from .io import path_from_record, read_json, read_json_object, read_json_object_list
from .layout import (
    STAGE_COLLECT,
    STAGE_FINAL_COLLECT,
    STAGE_FINAL_EVAL,
    STAGE_FINAL_GRID,
    STAGE_FINAL_REPORT,
    STAGE_FINAL_TRAIN,
    STAGE_GRID,
    STAGE_SELECT,
    STAGE_TRAIN,
    STAGE_VALIDATION,
    grid_attempt_dir,
    stage_dir,
)


@dataclass(frozen=True)
class SourceGrid:
    """Resolved source ``00_grid`` attempt for a downstream artifact."""

    attempt_id: str
    attempt_dir: Path
    manifest_path: Path

    def to_record(self) -> dict[str, str]:
        """Return a JSON-safe provenance record."""

        return {
            "grid_attempt_id": self.attempt_id,
            "grid_attempt_dir": str(self.attempt_dir),
            "manifest_path": str(self.manifest_path),
        }

    def read_manifest(self) -> dict[str, Any]:
        """Read this grid attempt's routine manifest."""

        manifest = read_json(self.manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"grid manifest must be a JSON object: {self.manifest_path}")
        return manifest


@dataclass(frozen=True)
class SourceAncestry:
    """Result roots traced through stage provenance records."""

    roots: frozenset[Path]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Ancestry:
    """Traced result roots for a final report attempt."""

    roots: frozenset[Path]
    warnings: tuple[str, ...]


def source_grid_from_id(results_root: str | Path, grid_attempt_id: str) -> SourceGrid:
    """Return the ``00_grid`` source descriptor for ``grid_attempt_id``."""

    attempt_id = str(grid_attempt_id)
    attempt_dir = grid_attempt_dir(results_root, attempt_id).resolve()
    return SourceGrid(
        attempt_id=attempt_id,
        attempt_dir=attempt_dir,
        manifest_path=(attempt_dir / "manifest.json").resolve(),
    )


def source_grid_from_record(
    results_root: str | Path,
    record: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> SourceGrid | None:
    """Resolve a source-grid record into a ``SourceGrid`` descriptor."""

    attempt_id = str(record.get("grid_attempt_id") or "").strip()
    attempt_dir = path_from_record(record, "grid_attempt_dir")
    if not attempt_id and attempt_dir is not None:
        attempt_id = attempt_dir.name
    if not attempt_id:
        return None
    source = source_grid_from_id(results_root, attempt_id)
    if attempt_dir is not None:
        source = SourceGrid(
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            manifest_path=path_from_record(record, "manifest_path") or (attempt_dir / "manifest.json").resolve(),
        )
    if warnings is not None and not source.attempt_dir.is_dir():
        warnings.append(f"missing grid directory: {source.attempt_dir}")
    if warnings is not None and not source.manifest_path.is_file():
        warnings.append(f"missing grid manifest: {source.manifest_path}")
    return source


def source_grid_from_attempt(
    results_root: str | Path,
    attempt_dir: str | Path,
    *,
    warnings: list[str] | None = None,
) -> SourceGrid | None:
    """Trace an attempt's provenance back to its source ``00_grid`` attempt."""

    results_root = Path(results_root).resolve()
    return _source_grid_from_attempt(results_root, Path(attempt_dir).resolve(), warnings=warnings, seen=set())


def trace_source_ancestry(
    results_root: str | Path,
    attempt_dir: str | Path,
    *,
    include_scan_run_roots: bool = True,
) -> SourceAncestry:
    """Trace result roots reachable from a stage attempt's provenance records."""

    roots: set[Path] = set()
    warnings: list[str] = []
    _trace_attempt_roots(
        Path(results_root).resolve(),
        Path(attempt_dir).resolve(),
        roots,
        warnings,
        include_scan_run_roots=include_scan_run_roots,
        seen=set(),
    )
    return SourceAncestry(frozenset(roots), tuple(warnings))


def trace_final_report_ancestry(results_root: str | Path, report_attempt_id: str) -> Ancestry:
    """Trace the result directories consumed by ``09_final_report/{attempt}``."""

    results_root = Path(results_root).resolve()
    roots: set[Path] = set()
    warnings: list[str] = []
    report_dir = stage_dir(results_root, STAGE_FINAL_REPORT) / report_attempt_id
    _add_dir(roots, warnings, report_dir, "final report")

    report_json = read_json_object(report_dir / "final_report.json", warnings)
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
        ancestry = trace_source_ancestry(
            results_root,
            eval_dir,
            include_scan_run_roots=False,
        )
        roots.update(ancestry.roots)
        warnings.extend(ancestry.warnings)

    return Ancestry(frozenset(roots), tuple(warnings))


def _source_grid_from_attempt(
    results_root: Path,
    attempt_dir: Path,
    *,
    warnings: list[str] | None,
    seen: set[Path],
) -> SourceGrid | None:
    attempt_dir = attempt_dir.resolve()
    if attempt_dir in seen:
        return None
    seen.add(attempt_dir)

    if _stage_name(attempt_dir, results_root) == STAGE_GRID:
        return source_grid_from_id(results_root, attempt_dir.name)

    direct = _source_grid_from_direct_file(results_root, attempt_dir / "source_grid_attempt.json", warnings=warnings)
    if direct is not None:
        return direct

    train_source = _source_grid_from_direct_file(
        results_root,
        attempt_dir / "source_train_attempt.json",
        warnings=None,
    )
    if train_source is not None:
        return train_source

    source_train = _read_optional_object(attempt_dir / "source_train_attempt.json", warnings=warnings)
    train_dir = path_from_record(source_train, "train_attempt_dir")
    if train_dir is not None:
        source = _source_grid_from_attempt(results_root, train_dir, warnings=warnings, seen=seen)
        if source is not None:
            return source

    for filename, path_key, id_key, stage in (
        ("source_collection_attempt.json", "collection_attempt_dir", "collection_attempt_id", STAGE_COLLECT),
        ("source_selection_attempt.json", "selection_attempt_dir", "selection_attempt_id", STAGE_SELECT),
        ("source_final_grid_attempt.json", "final_grid_attempt_dir", "final_grid_attempt_id", STAGE_FINAL_GRID),
        ("source_final_train_attempt.json", "final_train_attempt_dir", "final_train_attempt_id", STAGE_FINAL_TRAIN),
    ):
        record = _read_optional_object(attempt_dir / filename, warnings=warnings)
        upstream = _upstream_attempt_dir(results_root, record, path_key=path_key, id_key=id_key, stage=stage)
        if upstream is None:
            continue
        source = _source_grid_from_attempt(results_root, upstream, warnings=warnings, seen=seen)
        if source is not None:
            return source

    for source in _read_optional_object_list(attempt_dir / "source_validation_attempts.json", warnings=warnings):
        validation_dir = path_from_record(source, "validation_attempt_dir")
        if validation_dir is None:
            continue
        source_grid = _source_grid_from_attempt(results_root, validation_dir, warnings=warnings, seen=seen)
        if source_grid is not None:
            return source_grid
    return None


def _source_grid_from_direct_file(
    results_root: Path,
    path: Path,
    *,
    warnings: list[str] | None,
) -> SourceGrid | None:
    if not path.is_file():
        return None
    return source_grid_from_record(results_root, read_json_object(path, warnings=warnings), warnings=warnings)


def _trace_attempt_roots(
    results_root: Path,
    attempt_dir: Path,
    roots: set[Path],
    warnings: list[str],
    *,
    include_scan_run_roots: bool,
    seen: set[Path],
) -> None:
    attempt_dir = attempt_dir.resolve()
    if attempt_dir in seen:
        return
    seen.add(attempt_dir)

    stage = _stage_name(attempt_dir, results_root)
    if stage == STAGE_GRID:
        _add_existing_root(roots, warnings, attempt_dir, "grid")
        return
    if stage not in {STAGE_TRAIN, STAGE_VALIDATION} or include_scan_run_roots:
        _add_existing_root(roots, warnings, attempt_dir, stage or "attempt")
    elif not attempt_dir.is_dir():
        warnings.append(f"missing {stage or 'attempt'} directory: {attempt_dir}")
        return

    direct_grid = _source_grid_from_direct_file(results_root, attempt_dir / "source_grid_attempt.json", warnings=warnings)
    if direct_grid is not None:
        _add_existing_root(roots, warnings, direct_grid.attempt_dir, "grid")

    source_train = _read_optional_object(attempt_dir / "source_train_attempt.json", warnings=warnings)
    direct_from_train = source_grid_from_record(results_root, source_train, warnings=warnings)
    if direct_from_train is not None:
        _add_existing_root(roots, warnings, direct_from_train.attempt_dir, "grid")
    train_dir = path_from_record(source_train, "train_attempt_dir")
    if train_dir is not None:
        _trace_attempt_roots(
            results_root,
            train_dir,
            roots,
            warnings,
            include_scan_run_roots=include_scan_run_roots,
            seen=seen,
        )

    for filename, path_key, id_key, upstream_stage in (
        ("source_collection_attempt.json", "collection_attempt_dir", "collection_attempt_id", STAGE_COLLECT),
        ("source_selection_attempt.json", "selection_attempt_dir", "selection_attempt_id", STAGE_SELECT),
        ("source_final_grid_attempt.json", "final_grid_attempt_dir", "final_grid_attempt_id", STAGE_FINAL_GRID),
        ("source_final_train_attempt.json", "final_train_attempt_dir", "final_train_attempt_id", STAGE_FINAL_TRAIN),
    ):
        record = _read_optional_object(attempt_dir / filename, warnings=warnings)
        upstream = _upstream_attempt_dir(results_root, record, path_key=path_key, id_key=id_key, stage=upstream_stage)
        if upstream is None:
            continue
        _trace_attempt_roots(
            results_root,
            upstream,
            roots,
            warnings,
            include_scan_run_roots=include_scan_run_roots,
            seen=seen,
        )

    for source in _read_optional_object_list(attempt_dir / "source_validation_attempts.json", warnings=warnings):
        validation_dir = path_from_record(source, "validation_attempt_dir")
        if validation_dir is None:
            continue
        _trace_attempt_roots(
            results_root,
            validation_dir,
            roots,
            warnings,
            include_scan_run_roots=include_scan_run_roots,
            seen=seen,
        )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_optional_object(path: Path, *, warnings: list[str] | None) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json_object(path, warnings=warnings)


def _read_optional_object_list(path: Path, *, warnings: list[str] | None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_json_object_list(path, warnings=warnings)


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


def _upstream_attempt_dir(
    results_root: Path,
    record: dict[str, Any],
    *,
    path_key: str,
    id_key: str,
    stage: str,
) -> Path | None:
    path = path_from_record(record, path_key)
    if path is not None:
        return path
    attempt_id = str(record.get(id_key) or "").strip()
    if not attempt_id:
        return None
    return stage_dir(results_root, stage) / attempt_id


def _stage_name(path: Path, results_root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(results_root.resolve())
    except ValueError:
        return None
    return relative.parts[0] if relative.parts else None


def _add_existing_root(roots: set[Path], warnings: list[str], path: Path, label: str) -> bool:
    path = path.resolve()
    if path.is_dir():
        roots.add(path)
        return True
    warnings.append(f"missing {label} directory: {path}")
    return False


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
