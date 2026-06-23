"""Create compact pair-stability snapshots from the latest final report.

The snapshot contains the current study directory plus the result ancestry
needed to reproduce the latest ``09_final_report``. Result ancestry is traced
through explicit stage provenance files. Checkpoint directories are omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from omegaconf import OmegaConf

from run_utils import (
    DEFAULT_STUDY_TIMEZONE,
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
    attempt_ids,
    read_json,
    resolve_timezone,
    stage_dir,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STUDY_DIR / "configs" / "pair_stability.yaml"
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
CHECKPOINT_DIRNAME = "checkpoints"
PYCACHE_DIRNAME = "__pycache__"
SNAPSHOT_SUFFIX_FORMAT = "%Y%m%dT%H%M%S%z"
RUN_METADATA_FILENAMES = {
    "config.yaml",
    "launcher_status.json",
    "metadata.json",
    "resolved_config.yaml",
    "run_stat.json",
    "run_stats.json",
    "run_start.json",
    "source_grid_attempt.json",
    "source_train_attempt.json",
    "status.json",
    "submission.json",
}
STAGE_NAMES = {
    STAGE_GRID,
    STAGE_TRAIN,
    STAGE_VALIDATION,
    STAGE_COLLECT,
    STAGE_SELECT,
    STAGE_FINAL_GRID,
    STAGE_FINAL_TRAIN,
    STAGE_FINAL_EVAL,
    STAGE_FINAL_COLLECT,
    STAGE_FINAL_REPORT,
}


@dataclass(frozen=True)
class SyncPlan:
    """Concrete snapshot copy plan."""

    snapshot_dir: Path
    files: tuple[Path, ...]
    ancestry_roots: tuple[Path, ...]
    skipped_checkpoint_dirs: int
    warnings: tuple[str, ...] = ()


@dataclass
class SyncSummary:
    """Counts and provenance from one snapshot sync."""

    study_name: str
    source_study_dir: Path
    results_root: Path
    snapshot_dir: Path
    final_report_attempt_id: str
    dry_run: bool
    scanned_files: int
    planned_files: int
    planned_bytes: int
    copied_files: int
    copied_bytes: int
    skipped_checkpoint_dirs: int
    ancestry_stage_counts: dict[str, int] = field(default_factory=dict)
    ancestry_roots: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def copied_mb(self) -> float:
        """Return copied bytes as MiB."""

        return self.copied_bytes / (1024 * 1024)

    @property
    def planned_mb(self) -> float:
        """Return planned transfer bytes as MiB."""

        return self.planned_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary."""

        return {
            "study_name": self.study_name,
            "source_study_dir": str(self.source_study_dir),
            "results_root": str(self.results_root),
            "snapshot_dir": str(self.snapshot_dir),
            "final_report_attempt_id": self.final_report_attempt_id,
            "dry_run": self.dry_run,
            "scanned_files": self.scanned_files,
            "planned_files": self.planned_files,
            "planned_mb": round(self.planned_mb, 3),
            "copied_files": self.copied_files,
            "copied_mb": round(self.copied_mb, 3),
            "skipped_checkpoint_dirs": self.skipped_checkpoint_dirs,
            "ancestry_stage_counts": dict(self.ancestry_stage_counts),
            "ancestry_roots": list(self.ancestry_roots),
            "warnings": list(self.warnings),
        }


def sync_snapshot(
    *,
    destination: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    results_root: str | Path | None = None,
    report_attempt_id: str | None = None,
    study_dir: str | Path = STUDY_DIR,
    dry_run: bool = False,
    verbose: bool = False,
    moment: datetime | None = None,
) -> SyncSummary:
    """Copy the current study plus latest final-report ancestry into a snapshot."""

    study_dir = Path(study_dir).resolve()
    config_path = Path(config_path).resolve()
    config = load_study_config(config_path)
    study_name = _study_name(config)
    timezone = resolve_timezone(_config_text(config, "run.timezone") or DEFAULT_STUDY_TIMEZONE)
    results_root = resolve_results_root(config_path=config_path, explicit=results_root).resolve()
    report_attempt_id = resolve_final_report_attempt_id(results_root, report_attempt_id)
    snapshot_dir = Path(destination).resolve() / snapshot_name(study_name, moment=moment, timezone_name=timezone.key)
    _validate_destination(study_dir, results_root, snapshot_dir)

    plan = build_sync_plan(
        study_dir=study_dir,
        results_root=results_root,
        snapshot_dir=snapshot_dir,
        report_attempt_id=report_attempt_id,
    )
    planned_files = len(plan.files)
    planned_bytes = sum(path.lstat().st_size for path in plan.files)
    if verbose:
        for path in plan.files:
            print(_display_relative(path, study_dir, results_root), flush=True)
    copied_files = 0
    copied_bytes = 0
    if not dry_run:
        snapshot_dir.mkdir(parents=True)
        for path in plan.files:
            _copy_file(path, snapshot_dir / _snapshot_relative_path(path, study_dir, results_root))
        copied_files = planned_files
        copied_bytes = planned_bytes
        write_json(snapshot_dir / "sync_manifest.json", _summary_payload(
            study_name=study_name,
            study_dir=study_dir,
            results_root=results_root,
            snapshot_dir=snapshot_dir,
            report_attempt_id=report_attempt_id,
            dry_run=dry_run,
            scanned_files=len(plan.files),
            planned_files=planned_files,
            planned_bytes=planned_bytes,
            copied_files=copied_files,
            copied_bytes=copied_bytes,
            skipped_checkpoint_dirs=plan.skipped_checkpoint_dirs,
            ancestry_stage_counts=_stage_counts(plan.ancestry_roots, results_root),
            ancestry_roots=plan.ancestry_roots,
            warnings=plan.warnings,
        ))
    return SyncSummary(
        study_name=study_name,
        source_study_dir=study_dir,
        results_root=results_root,
        snapshot_dir=snapshot_dir,
        final_report_attempt_id=report_attempt_id,
        dry_run=dry_run,
        scanned_files=len(plan.files),
        planned_files=planned_files,
        planned_bytes=planned_bytes,
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        skipped_checkpoint_dirs=plan.skipped_checkpoint_dirs,
        ancestry_stage_counts=_stage_counts(plan.ancestry_roots, results_root),
        ancestry_roots=[str(path) for path in plan.ancestry_roots],
        warnings=list(plan.warnings),
    )


def build_sync_plan(
    *,
    study_dir: str | Path,
    results_root: str | Path,
    snapshot_dir: str | Path,
    report_attempt_id: str,
) -> SyncPlan:
    """Return source files needed for one pair-stability snapshot."""

    study_dir = Path(study_dir).resolve()
    results_root = Path(results_root).resolve()
    snapshot_dir = Path(snapshot_dir).resolve()
    ancestry = trace_final_report_ancestry(results_root, report_attempt_id)
    files: set[Path] = set()
    skipped_checkpoint_dirs = 0

    for path in _iter_study_files(study_dir, results_root):
        files.add(path)
    for root in ancestry.roots:
        root_files, skipped = _files_under_root(root)
        files.update(root_files)
        skipped_checkpoint_dirs += skipped
    warnings = list(ancestry.warnings)
    files.update(_run_metadata_files_from_collection_roots(ancestry.roots, results_root, warnings))
    return SyncPlan(
        snapshot_dir=snapshot_dir,
        files=tuple(sorted(files)),
        ancestry_roots=tuple(sorted(ancestry.roots)),
        skipped_checkpoint_dirs=skipped_checkpoint_dirs,
        warnings=tuple(warnings),
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
    final_eval_dirs = _resolve_final_eval_dirs(
        results_root,
        final_eval_attempts,
    )
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


def load_study_config(config_path: str | Path) -> Any:
    """Load the study config that owns sync defaults."""

    return OmegaConf.load(config_path)


def resolve_results_root(*, config_path: str | Path, explicit: str | Path | None = None) -> Path:
    """Resolve results root from CLI or from ``run.root`` in the config."""

    if explicit is not None:
        return Path(explicit)
    config = load_study_config(config_path)
    raw_root = _config_text(config, "run.root")
    if raw_root:
        run_root = _resolve_config_path(raw_root, config_path)
        if run_root.name in STAGE_NAMES:
            return run_root.parent
    return DEFAULT_RESULTS_ROOT


def resolve_final_report_attempt_id(results_root: str | Path, requested: str | None = None) -> str:
    """Return requested or latest final-report attempt id."""

    if requested is not None:
        return requested
    report_stage = stage_dir(results_root, STAGE_FINAL_REPORT)
    latest = report_stage / "latest.json"
    if latest.is_file():
        attempt_id = read_json(latest).get("attempt_id")
        if attempt_id:
            return str(attempt_id)
    attempts = attempt_ids(report_stage)
    if not attempts:
        raise FileNotFoundError(f"no final-report attempts under {report_stage}")
    return attempts[-1]


def snapshot_name(
    study_name: str,
    *,
    moment: datetime | None = None,
    timezone_name: str = DEFAULT_STUDY_TIMEZONE,
) -> str:
    """Return ``<study>_snapshot_<YYYYMMDD>T<HHMMSS>-0400`` in the study timezone."""

    tz = resolve_timezone(timezone_name)
    now = (moment or datetime.now(tz)).astimezone(tz)
    return f"{_safe_component(study_name)}_snapshot_{now.strftime(SNAPSHOT_SUFFIX_FORMAT)}"


def _iter_study_files(study_dir: Path, results_root: Path) -> Iterable[Path]:
    study_dir = study_dir.resolve()
    results_root = results_root.resolve()
    for directory, dirnames, filenames in os.walk(study_dir, topdown=True, followlinks=False):
        current = Path(directory)
        dirnames[:] = [
            dirname
            for dirname in sorted(dirnames)
            if not _prune_source_dir(current / dirname, results_root)
        ]
        for filename in sorted(filenames):
            path = current / filename
            if path.is_file() or path.is_symlink():
                yield path.resolve()


def _files_under_root(root: Path) -> tuple[set[Path], int]:
    files: set[Path] = set()
    skipped_checkpoint_dirs = 0
    if not root.is_dir():
        return files, skipped_checkpoint_dirs
    for directory, dirnames, filenames in os.walk(root.resolve(), topdown=True, followlinks=False):
        current = Path(directory)
        kept_dirnames = []
        for dirname in sorted(dirnames):
            child = current / dirname
            if dirname == PYCACHE_DIRNAME:
                continue
            if dirname == CHECKPOINT_DIRNAME:
                skipped_checkpoint_dirs += 1
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames
        for filename in sorted(filenames):
            path = current / filename
            if _skip_pycache(path):
                continue
            if path.is_file() or path.is_symlink():
                files.add(path.resolve())
    return files, skipped_checkpoint_dirs


def _run_metadata_files_from_collection_roots(
    roots: Iterable[Path],
    results_root: Path,
    warnings: list[str],
) -> set[Path]:
    files: set[Path] = set()
    results_root = results_root.resolve()
    for root in roots:
        root = root.resolve()
        if not _is_stage_root(root, results_root, STAGE_COLLECT):
            continue
        sources = _read_json_list(root / "source_validation_attempts.json", warnings)
        for source in sources:
            validation_dir = _path_from_record(source, "validation_attempt_dir")
            if validation_dir is None:
                continue
            files.update(_run_metadata_files(validation_dir, warnings, "validation"))
            source_train = _read_json_dict(validation_dir / "source_train_attempt.json", warnings)
            train_dir = _path_from_record(source_train, "train_attempt_dir")
            if train_dir is not None:
                files.update(_run_metadata_files(train_dir, warnings, "train"))
    return files


def _run_metadata_files(run_dir: Path, warnings: list[str], label: str) -> set[Path]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        warnings.append(f"missing {label} directory for metadata: {run_dir}")
        return set()
    files = set()
    for filename in RUN_METADATA_FILENAMES:
        path = run_dir / filename
        if path.is_file() or path.is_symlink():
            files.add(path.resolve())
    return files


def _is_stage_root(path: Path, results_root: Path, stage: str) -> bool:
    path = path.resolve()
    results_root = results_root.resolve()
    if not _is_relative_to(path, results_root):
        return False
    relative = path.relative_to(results_root)
    return bool(relative.parts) and relative.parts[0] == stage


def _prune_source_dir(path: Path, results_root: Path) -> bool:
    if path.name in {PYCACHE_DIRNAME, CHECKPOINT_DIRNAME}:
        return True
    return _is_relative_to(path.resolve(), results_root)


def _skip_pycache(path: Path) -> bool:
    return PYCACHE_DIRNAME in path.parts or path.suffix == ".pyc"


def _snapshot_relative_path(path: Path, study_dir: Path, results_root: Path) -> Path:
    path = path.resolve()
    if _is_relative_to(path, results_root):
        return Path("results") / path.relative_to(results_root)
    return path.relative_to(study_dir)


def _display_relative(path: Path, study_dir: Path, results_root: Path) -> Path:
    return _snapshot_relative_path(path.resolve(), study_dir.resolve(), results_root.resolve())


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def _summary_payload(
    *,
    study_name: str,
    study_dir: Path,
    results_root: Path,
    snapshot_dir: Path,
    report_attempt_id: str,
    dry_run: bool,
    scanned_files: int,
    planned_files: int,
    planned_bytes: int,
    copied_files: int,
    copied_bytes: int,
    skipped_checkpoint_dirs: int,
    ancestry_stage_counts: dict[str, int],
    ancestry_roots: Sequence[Path],
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "study_name": study_name,
        "source_study_dir": str(study_dir),
        "results_root": str(results_root),
        "snapshot_dir": str(snapshot_dir),
        "final_report_attempt_id": report_attempt_id,
        "dry_run": dry_run,
        "scanned_files": scanned_files,
        "planned_files": planned_files,
        "planned_mb": round(planned_bytes / (1024 * 1024), 3),
        "copied_files": copied_files,
        "copied_mb": round(copied_bytes / (1024 * 1024), 3),
        "skipped_checkpoint_dirs": skipped_checkpoint_dirs,
        "ancestry_stage_counts": dict(ancestry_stage_counts),
        "ancestry_roots": [str(path) for path in ancestry_roots],
        "warnings": list(warnings),
    }


def _stage_counts(roots: Sequence[Path], results_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    results_root = results_root.resolve()
    for root in roots:
        root = root.resolve()
        if not _is_relative_to(root, results_root):
            continue
        relative = root.relative_to(results_root)
        if not relative.parts:
            continue
        stage = relative.parts[0]
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


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
    else:
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


def _study_name(config: Any) -> str:
    return _config_text(config, "study.name") or _config_text(config, "experiment.name") or "pair_stability"


def _resolve_config_path(raw: str, config_path: str | Path) -> Path:
    raw_path = Path(raw)
    if raw_path.is_absolute():
        return raw_path
    cwd_candidate = raw_path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root = _find_repo_root(Path(config_path).resolve().parent)
    return (repo_root / raw_path).resolve()


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return start


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value) or "study"


def _validate_destination(study_dir: Path, results_root: Path, snapshot_dir: Path) -> None:
    if not study_dir.is_dir():
        raise NotADirectoryError(f"study directory does not exist: {study_dir}")
    if not results_root.is_dir():
        raise NotADirectoryError(f"results root does not exist: {results_root}")
    if snapshot_dir.exists():
        raise FileExistsError(f"snapshot destination already exists: {snapshot_dir}")
    for source_root in (study_dir, results_root):
        if _is_relative_to(snapshot_dir, source_root):
            raise ValueError(f"snapshot destination must not be inside source root: {source_root}")
        if _is_relative_to(source_root, snapshot_dir):
            raise ValueError(f"snapshot destination must not be an ancestor of source root: {source_root}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _print_summary(summary: SyncSummary) -> None:
    _print_status(f"study_name: {summary.study_name}")
    _print_status(f"results_root: {summary.results_root}")
    _print_status(f"snapshot_dir: {summary.snapshot_dir}")
    _print_status(f"final_report_attempt_id: {summary.final_report_attempt_id}")
    _print_status(f"dry_run: {str(summary.dry_run).lower()}")
    _print_status(f"planned_files: {summary.planned_files}")
    _print_status(f"planned_mb: {summary.planned_mb:.3f}")
    _print_status(f"copied_files: {summary.copied_files}")
    _print_status(f"copied_mb: {summary.copied_mb:.3f}")
    _print_status(f"skipped_checkpoint_dirs: {summary.skipped_checkpoint_dirs}")
    if summary.dry_run:
        _print_status(f"ancestry_roots: {len(summary.ancestry_roots)}")
        for stage, count in summary.ancestry_stage_counts.items():
            _print_status(f"  {stage}: {count}")
        _print_status("use --verbose to print every planned relative path")
    for warning in summary.warnings:
        _print_status(f"warning: {warning}")


def _print_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse sync arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        help="Parent directory that will receive <study>_snapshot_<timestamp>.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Study config; owns study.name and default results root.")
    parser.add_argument("--study-dir", type=Path, default=STUDY_DIR, help="Study directory to snapshot.")
    parser.add_argument("--results-root", type=Path, default=None, help="Override results root.")
    parser.add_argument("--report-attempt-id", default=None, help="Final report attempt id; defaults to latest.")
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Plan the snapshot without copying files; prints planned files and MiB.",
    )
    parser.add_argument("--dryrun", dest="dry_run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help="Print every planned relative path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the snapshot sync CLI."""

    args = parse_args(argv)
    mode = "dry-run snapshot" if args.dry_run else "snapshot"
    _print_status(f"[pair_stability] planning {mode} under {args.destination}")
    summary = sync_snapshot(
        destination=args.destination,
        config_path=args.config,
        study_dir=args.study_dir,
        results_root=args.results_root,
        report_attempt_id=args.report_attempt_id,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    _print_summary(summary)
    return 0


__all__ = [
    "Ancestry",
    "SyncPlan",
    "SyncSummary",
    "build_sync_plan",
    "resolve_final_report_attempt_id",
    "resolve_results_root",
    "snapshot_name",
    "sync_snapshot",
    "trace_final_report_ancestry",
]


if __name__ == "__main__":
    raise SystemExit(main())
