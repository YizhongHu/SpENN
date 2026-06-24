"""Collect validation attempts into a summary table.

Walks ``02_validation/{run_id}/*`` (the latest attempt per run id),
reads each attempt's status, evaluation metrics, source train-attempt metrics,
and recorded train-attempt provenance, and writes a ``03_collect`` attempt with
``summary.csv``, ``failures.csv``, ``collection_report.json``, and explicit
source pointers to the exact validation (and grid) attempts consumed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from artifacts import duration_from_status_file, read_metrics_map, status_of, write_csv
from run_utils import (
    STAGE_COLLECT,
    STAGE_GRID,
    STAGE_VALIDATION,
    SourceGrid,
    axis_id_labels_from_manifest,
    attempt_ids,
    grid_axes_from_manifest,
    id_for_axes,
    log_prefix,
    new_attempt_id,
    read_json,
    source_grid_from_attempt,
    source_grid_from_id,
    stage_dir,
    study_name_from_manifest,
    validation_run_dir,
    write_json,
)
from stats import as_float

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

NON_DIAGNOSTIC_DIRS = {"checkpoints", "diagnostics"}
SUCCESS_STATUSES = {"completed", "success"}

BASE_COLUMNS = (
    "run_id",
    "validation_attempt_id",
    "validation_attempt_dir",
    "status",
    "major_id",
    "minor_id",
    "config_id",
    "train_attempt_id",
    "checkpoint_path",
    "n_diagnostics",
)

TRAIN_WALL_TIME_METRIC = "train/runtime/wall_time_sec"


def latest_attempt_id(run_dir: Path) -> str | None:
    """Return the most recent attempt id directly under ``run_dir``."""

    ids = attempt_ids(run_dir)
    return ids[-1] if ids else None


def read_metrics_jsonl(path: Path) -> dict[str, Any]:
    """Flatten ``metrics.jsonl`` records into ``namespace/key -> value``."""

    return read_metrics_map(path)


def _run_parameters(attempt_dir: Path) -> dict[str, Any]:
    """Recover run parameters from resolved_config.yaml when available."""

    resolved = attempt_dir / "resolved_config.yaml"
    if resolved.is_file():
        cfg = OmegaConf.load(resolved)
        params = OmegaConf.select(cfg, "run_parameters")
        if params is not None:
            return OmegaConf.to_container(params, resolve=True)
    return {}


def _count_diagnostics(attempt_dir: Path) -> int:
    """Count evaluation task outputs recorded for this attempt."""

    index = attempt_dir / "diagnostics" / "index.json"
    if index.is_file():
        data = read_json(index)
        if isinstance(data, dict) and isinstance(data.get("artifacts"), list):
            return len(data["artifacts"])
        if isinstance(data, list):
            return len(data)
    return sum(
        1
        for child in attempt_dir.iterdir()
        if child.is_dir() and child.name not in NON_DIAGNOSTIC_DIRS
    )


def _train_attempt_dir(source: dict[str, Any]) -> Path | None:
    """Return the source train attempt directory recorded by validation."""

    train_attempt_dir = source.get("train_attempt_dir")
    if train_attempt_dir:
        return Path(str(train_attempt_dir))
    checkpoint_path = source.get("checkpoint_path")
    if checkpoint_path:
        checkpoint = Path(str(checkpoint_path))
        if checkpoint.name == "checkpoints":
            return checkpoint.parent
    return None


def _train_metrics(source: dict[str, Any]) -> dict[str, Any]:
    """Return metrics from the validation source train attempt."""

    train_attempt = _train_attempt_dir(source)
    if train_attempt is None:
        return {}
    train_metrics = read_metrics_map(train_attempt / "metrics.jsonl", prefix="train")
    if as_float(train_metrics.get(TRAIN_WALL_TIME_METRIC)) is None:
        wall_time = duration_from_status_file(train_attempt, clamp_negative=True)
        if wall_time is not None:
            train_metrics[TRAIN_WALL_TIME_METRIC] = wall_time
    return train_metrics


def _axis_metadata(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized axis metadata for collection."""

    axes = grid_axes_from_manifest(manifest)
    all_axes = tuple(axis for axis in axes["run_axes"] if isinstance(axis, str))
    labels = axis_id_labels_from_manifest(manifest, all_axes)
    return {
        **axes,
        "axis_id_labels": labels,
    }


def _point_from_sources(
    *,
    params: dict[str, Any],
    grid_job: dict[str, Any] | None,
    axes: Sequence[str],
) -> dict[str, Any]:
    """Return configured axis values from resolved config or source grid job."""

    choices = grid_job.get("choices", {}) if isinstance(grid_job, dict) else {}
    point = {}
    for axis in axes:
        if axis in params:
            point[axis] = params[axis]
        elif isinstance(choices, dict) and axis in choices:
            point[axis] = choices[axis]
        else:
            point[axis] = ""
    return point


def collect_validation_attempt(
    run_id: str,
    attempt_id: str,
    attempt_dir: Path,
    *,
    grid_job: dict[str, Any] | None,
    axis_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build one summary row from a validation attempt directory."""

    params = _run_parameters(attempt_dir)
    major_axes = tuple(axis_metadata["major_axes"])
    minor_axes = tuple(axis_metadata["minor_axes"])
    config_axes = tuple(axis_metadata["config_axes"])
    run_axes = tuple(axis_metadata["run_axes"])
    labels = axis_metadata["axis_id_labels"]
    point = _point_from_sources(params=params, grid_job=grid_job, axes=run_axes)
    source_path = attempt_dir / "source_train_attempt.json"
    source = read_json(source_path) if source_path.is_file() else {}

    row: dict[str, Any] = {column: "" for column in (*BASE_COLUMNS, *run_axes)}
    row.update(
        run_id=run_id,
        validation_attempt_id=attempt_id,
        validation_attempt_dir=str(attempt_dir),
        status=status_of(attempt_dir),
        major_id=(grid_job or {}).get("major_id") or id_for_axes(point, major_axes, labels),
        minor_id=(grid_job or {}).get("minor_id") or id_for_axes(point, minor_axes, labels),
        config_id=(grid_job or {}).get("config_id") or id_for_axes(point, config_axes, labels),
        train_attempt_id=source.get("train_attempt_id", ""),
        checkpoint_path=source.get("checkpoint_path", ""),
        n_diagnostics=_count_diagnostics(attempt_dir),
    )
    row.update(point)
    row.update(_train_metrics(source))
    row.update(read_metrics_jsonl(attempt_dir / "metrics.jsonl"))
    return row


def _latest_validation_attempts(results_root: Path) -> list[tuple[str, str, Path]]:
    """Return the latest validation attempt for each validation run id."""

    validation_root = stage_dir(results_root, STAGE_VALIDATION)
    if not validation_root.is_dir():
        return []
    attempts = []
    for run_dir in sorted(child for child in validation_root.iterdir() if child.is_dir()):
        attempt_id = latest_attempt_id(run_dir)
        if attempt_id is not None:
            attempts.append((run_dir.name, attempt_id, run_dir / attempt_id))
    return attempts


def _latest_validation_attempt(results_root: Path) -> tuple[str, str, Path] | None:
    """Return the newest validation attempt across run ids."""

    attempts = _latest_validation_attempts(results_root)
    if not attempts:
        return None
    return max(attempts, key=lambda item: item[1])


def _source_grid_from_latest_validations(results_root: Path) -> SourceGrid | None:
    """Trace the newest validation attempt back to its source grid."""

    latest = _latest_validation_attempt(results_root)
    if latest is None:
        return None
    _run_id, _attempt_id, attempt_dir = latest
    return source_grid_from_attempt(results_root, attempt_dir)


def _resolve_grid_source(results_root: Path, grid_attempt_id: str | None) -> SourceGrid | None:
    """Return explicit, traced, or latest source grid for collection."""

    if grid_attempt_id is not None:
        return source_grid_from_id(results_root, grid_attempt_id)
    traced = _source_grid_from_latest_validations(results_root)
    if traced is not None:
        return traced
    latest = stage_dir(results_root, STAGE_GRID) / "latest.json"
    if latest.is_file():
        attempt_id = read_json(latest).get("attempt_id")
        if attempt_id:
            return source_grid_from_id(results_root, str(attempt_id))
    return None


def _grid_manifest(source_grid: SourceGrid | None) -> dict[str, Any] | None:
    """Return a source grid manifest if it is available."""

    if source_grid is None or not source_grid.manifest_path.is_file():
        return None
    return source_grid.read_manifest()


def _run_ids(results_root: Path, grid_manifest: dict[str, Any] | None) -> list[str]:
    """Return run ids to collect, from the grid manifest if available."""

    if grid_manifest is not None:
        return [str(job["run_id"]) for job in grid_manifest.get("jobs", [])]
    return [run_id for run_id, _attempt_id, _attempt_dir in _latest_validation_attempts(results_root)]


def collect(
    *,
    results_root: str | Path,
    collect_attempt_id: str | None = None,
    grid_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Collect validation attempts and write a ``03_collect`` attempt."""

    results_root = Path(results_root)
    collect_attempt_id = collect_attempt_id or new_attempt_id()
    source_grid = _resolve_grid_source(results_root, grid_attempt_id)
    grid_attempt_id = None if source_grid is None else source_grid.attempt_id
    grid_manifest = _grid_manifest(source_grid)
    study = study_name_from_manifest(grid_manifest)
    axis_metadata = _axis_metadata(grid_manifest)
    job_by_run = {
        str(job.get("run_id")): job
        for job in (grid_manifest or {}).get("jobs", [])
        if isinstance(job, dict) and job.get("run_id")
    }

    rows: list[dict[str, Any]] = []
    consumed: list[dict[str, Any]] = []
    for run_id in _run_ids(results_root, grid_manifest):
        run_dir = validation_run_dir(results_root, run_id)
        attempt_id = latest_attempt_id(run_dir)
        if attempt_id is None:
            continue
        attempt_dir = run_dir / attempt_id
        attempt_source = source_grid_from_attempt(results_root, attempt_dir)
        if (
            source_grid is not None
            and attempt_source is not None
            and attempt_source.attempt_id != source_grid.attempt_id
        ):
            continue
        rows.append(
            collect_validation_attempt(
                run_id,
                attempt_id,
                attempt_dir,
                grid_job=job_by_run.get(run_id),
                axis_metadata=axis_metadata,
            )
        )
        consumed.append(
            {"run_id": run_id, "validation_attempt_id": attempt_id, "validation_attempt_dir": str(attempt_dir)}
        )

    attempt = stage_dir(results_root, STAGE_COLLECT) / collect_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    axis_columns = list(axis_metadata["run_axes"])
    base_columns = [*BASE_COLUMNS, *axis_columns]
    metric_columns = sorted({key for row in rows for key in row if key not in base_columns})
    columns = base_columns + metric_columns
    failures = [row for row in rows if str(row["status"]) not in SUCCESS_STATUSES]

    write_csv(attempt / "summary.csv", rows, columns)
    write_csv(attempt / "failures.csv", failures, columns)
    write_json(
        attempt / "source_grid_attempt.json",
        {} if source_grid is None else source_grid.to_record(),
    )
    write_json(attempt / "source_validation_attempts.json", consumed)
    report = {
        "study": study,
        "stage": STAGE_COLLECT,
        "attempt_id": collect_attempt_id,
        "grid_attempt_id": grid_attempt_id,
        "major_axes": list(axis_metadata["major_axes"]),
        "minor_axes": list(axis_metadata["minor_axes"]),
        "scan_seed_axis": axis_metadata["scan_seed_axis"],
        "config_keys": list(axis_metadata["config_axes"]),
        "axis_id_labels": axis_metadata["axis_id_labels"],
        "n_collected": len(rows),
        "n_failures": len(failures),
        "metric_columns": metric_columns,
    }
    write_json(attempt / "collection_report.json", report)
    return {"attempt_dir": str(attempt), "report": report, "rows": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse collect command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--grid-attempt-id",
        default=None,
        help="Override source grid attempt; defaults to the grid traced from the newest validation attempt.",
    )
    parser.add_argument("--attempt-id", default=None, help="Collect attempt id (defaults to now).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Collect validation attempts from the command line."""

    args = parse_args(argv)
    result = collect(
        results_root=args.results_root,
        collect_attempt_id=args.attempt_id,
        grid_attempt_id=args.grid_attempt_id,
    )
    report = result["report"]
    prefix = log_prefix(report.get("study"))
    print(
        f"{prefix} collected {report['n_collected']} runs "
        f"({report['n_failures']} failures) -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
