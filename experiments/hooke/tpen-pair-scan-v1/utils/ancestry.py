"""Source-grid provenance resolution for staged study artifacts.

Reduced port of the v3 module: this study keeps only the source-``00_grid``
resolution that ``collect.py``, ``select_champions.py``, and ``final_plan.py``
need to trace a downstream attempt back to the grid that planned it. The v3
report-ancestry tracing (``Ancestry``, ``trace_final_report_ancestry`` and the
root-collection helpers beneath them) existed only to serve the archival
``sync.py`` stage, which this study does not port.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import path_from_record, read_json, read_json_object, read_json_object_list
from .layout import (
    STAGE_COLLECT,
    STAGE_FINAL_GRID,
    STAGE_FINAL_TRAIN,
    STAGE_GRID,
    STAGE_SELECT,
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


def _read_optional_object(path: Path, *, warnings: list[str] | None) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json_object(path, warnings=warnings)


def _read_optional_object_list(path: Path, *, warnings: list[str] | None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_json_object_list(path, warnings=warnings)


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


