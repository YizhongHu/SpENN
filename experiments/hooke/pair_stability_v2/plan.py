"""Plan a staged study grid.

Expands configured major/minor/scan-seed axes into scalar override lists for
the canonical ``run.py`` entrypoint and writes a durable ``00_grid`` attempt
(manifest + commands) describing the planned train jobs.

Stage layout (under ``results_root``)::

    00_grid/{attempt_id}/{manifest.json, commands.sh, grid.yaml,
                          train_config.yaml, jobs/{run_id}.json}
    01_train/{run_id}/{attempt_id}/...
    02_validation/{run_id}/{attempt_id}/...
    03_collect/{attempt_id}/...
    04_select/{attempt_id}/...
"""

from __future__ import annotations

import argparse
import itertools
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from run_utils import (
    DEFAULT_STUDY_TIMEZONE,
    STAGE_GRID,
    STAGE_TRAIN,
    axis_value_label,
    config_snapshot_names,
    experiment_run_name,
    final_seed_sequences,
    grid_attempt_dir,
    log_prefix,
    new_attempt_id,
    resolve_timezone,
    seed_override_policy,
    seed_override_values,
    stage_dir,
    study_name,
    train_attempt_dir,
    train_run_dir,
    validation_run_dir,
    write_json,
    write_latest,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_GRID = STUDY_DIR / "configs" / "grid.yaml"


# ---------------------------------------------------------------------------
# Grid expansion and validation
# ---------------------------------------------------------------------------
def _axis_names(block: dict[str, Sequence[Any]], configured: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return deterministic axis names for one grid block."""

    if configured is not None:
        axes = tuple(str(axis) for axis in configured)
    else:
        axes = tuple(str(axis) for axis in block.keys())
    if not axes:
        raise ValueError("grid block must contain at least one axis")
    missing = [axis for axis in axes if axis not in block]
    if missing:
        raise ValueError(f"grid block is missing required axes: {', '.join(missing)}")
    return axes


def major_axes(grid_data: dict[str, Any]) -> tuple[str, ...]:
    """Return major-axis names from config, preserving YAML order by default."""

    return _axis_names(grid_data["major_grid"], grid_data.get("major_axes"))


def minor_axes(grid_data: dict[str, Any]) -> tuple[str, ...]:
    """Return minor-axis names from config, preserving YAML order by default."""

    return _axis_names(grid_data["minor_grid"], grid_data.get("minor_axes"))


def scan_seed_axis(grid_data: dict[str, Any]) -> str:
    """Return the scan seed axis name."""

    return str(grid_data.get("scan_seed_axis", "seed"))


def grid_axes(grid_data: dict[str, Any]) -> tuple[str, ...]:
    """Return the full scalar train axis order."""

    if "major_grid" in grid_data or "minor_grid" in grid_data or "scan_seeds" in grid_data:
        return (*major_axes(grid_data), *minor_axes(grid_data), scan_seed_axis(grid_data))
    return _axis_names(grid_data["grid"], grid_data.get("grid_axes"))


def _axis_points(grid: dict[str, Sequence[Any]], axes: Sequence[str]) -> list[dict[str, Any]]:
    """Expand selected axes from a grid block into dictionaries."""

    return [
        dict(zip(axes, combination, strict=True))
        for combination in itertools.product(*(list(grid[axis]) for axis in axes))
    ]


def expand_grid(grid: dict[str, Sequence[Any]], axes: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Expand a flat grid block into scalar points."""

    axes = _axis_names(grid, axes)
    return [
        dict(zip(axes, combination, strict=True))
        for combination in itertools.product(*(list(grid[axis]) for axis in axes))
    ]


def expand_split_grid(
    major_grid: dict[str, Sequence[Any]],
    minor_grid: dict[str, Sequence[Any]],
    scan_seeds: Sequence[Any],
    *,
    major_axis_names: Sequence[str],
    minor_axis_names: Sequence[str],
    seed_axis: str,
) -> list[dict[str, Any]]:
    """Expand major/minor/scan-seed blocks into scalar train points."""

    points = []
    for major in _axis_points(major_grid, major_axis_names):
        for minor in _axis_points(minor_grid, minor_axis_names):
            for seed in scan_seeds:
                points.append({**major, **minor, seed_axis: seed})
    return points


def expand_grid_spec(grid_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand either a split grid spec or a legacy flat grid spec."""

    if "major_grid" in grid_data or "minor_grid" in grid_data or "scan_seeds" in grid_data:
        return expand_split_grid(
            grid_data["major_grid"],
            grid_data["minor_grid"],
            grid_data["scan_seeds"],
            major_axis_names=major_axes(grid_data),
            minor_axis_names=minor_axes(grid_data),
            seed_axis=scan_seed_axis(grid_data),
        )
    return expand_grid(grid_data["grid"], grid_data.get("grid_axes"))


def champion_kinds(grid_data: dict[str, Any]) -> list[str]:
    """Return configured champion kinds."""

    kinds = [spec["name"] for spec in champion_specs(grid_data)]
    if not kinds:
        raise ValueError("champions must contain at least one winner kind")
    return kinds


def champion_specs(grid_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured champion selector specs.

    Each entry must be a mapping so metric semantics are visible in the grid
    snapshot and not encoded by study-local Python.
    """

    configured = grid_data.get("champions")
    if configured is None:
        raise ValueError("grid.yaml must define explicit champion selector specs")
    specs: list[dict[str, Any]] = []
    for entry in configured:
        if not isinstance(entry, dict):
            raise ValueError(f"champion entries must be mappings, got {entry!r}")
        spec = dict(entry)
        name = str(spec.get("name", "")).strip()
        selector = str(spec.get("selector", "")).strip()
        if not name:
            raise ValueError("champion specs require a non-empty name")
        if not selector:
            raise ValueError(f"champion {name!r} requires selector")
        spec["name"] = name
        spec["selector"] = selector
        specs.append(spec)
    if not specs:
        raise ValueError("champions must contain at least one selector spec")
    return specs


def champion_reference_metrics(grid_data: dict[str, Any]) -> list[dict[str, str]]:
    """Return configured extra metrics to copy into champions.csv."""

    configured = grid_data.get("champion_reference_metrics")
    if configured is None:
        return []
    metrics = []
    for entry in configured:
        if not isinstance(entry, dict):
            raise ValueError("champion_reference_metrics entries must be mappings")
        label = str(entry.get("label", "")).strip()
        metric = str(entry.get("metric", "")).strip()
        if not label or not metric:
            raise ValueError("champion_reference_metrics entries require label and metric")
        metrics.append({"label": label, "metric": metric})
    return metrics


def _config_select(config: Any, path: str, *, value: Any | None = None) -> Any:
    """Select a value from an OmegaConf config, supporting ``{value}`` templates."""

    return OmegaConf.select(config, path.format(value=value))


def choice_validation_specs(grid_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return axis validation specs from grid config."""

    configured = grid_data.get("choice_validation") or {}
    if not isinstance(configured, dict):
        raise ValueError("choice_validation must be a mapping")
    return {str(axis): dict(spec or {}) for axis, spec in configured.items()}


def choice_names(config: Any, choices_path: str) -> set[str]:
    """Return configured choice keys under ``choices_path``."""

    choices = _config_select(config, choices_path)
    if choices is None:
        raise ValueError(f"choice validation path {choices_path!r} does not exist")
    if not hasattr(choices, "keys"):
        raise ValueError(f"choice validation path {choices_path!r} is not a mapping")
    return {str(name) for name in choices.keys()}


def axis_tags(config: Any, validation_specs: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Return ``{axis: {choice: tags}}`` for axes that declare ``tags_path``."""

    tags: dict[str, dict[str, list[str]]] = {}
    for axis, spec in validation_specs.items():
        choices_path = str(spec.get("choices_path", "")).strip()
        tags_path = str(spec.get("tags_path", "")).strip()
        if not choices_path or not tags_path:
            continue
        tags[axis] = {}
        for value in choice_names(config, choices_path):
            selected = _config_select(config, tags_path, value=value) or []
            tags[axis][value] = [str(tag) for tag in selected]
    return tags


def tags_for_point(point: dict[str, Any], tags_by_axis: dict[str, dict[str, list[str]]]) -> list[str]:
    """Return tags attached to a grid point by configured tag sources."""

    tags: list[str] = []
    for axis, choices in tags_by_axis.items():
        tags.extend(choices.get(str(point.get(axis, "")), []))
    return sorted(set(tags))


def validate_grid(
    points: Sequence[dict[str, Any]],
    config: Any,
    validation_specs: dict[str, dict[str, Any]],
) -> None:
    """Fail loudly if a grid point violates configured choice validation."""

    known_by_axis = {
        axis: choice_names(config, str(spec["choices_path"]))
        for axis, spec in validation_specs.items()
        if str(spec.get("choices_path", "")).strip()
    }
    for point in points:
        for axis, known in known_by_axis.items():
            value = str(point.get(axis, ""))
            spec = validation_specs[axis]
            for suffix in spec.get("exclude_suffixes", []) or []:
                if value.endswith(str(suffix)):
                    raise ValueError(f"grid {axis} value {value!r} is excluded by suffix {suffix!r}")
            if value not in known:
                raise ValueError(f"grid {axis} value {value!r} is not under {spec['choices_path']!r}")


# ---------------------------------------------------------------------------
# Overrides and commands
# ---------------------------------------------------------------------------
def axis_id_labels(grid_data: dict[str, Any], axes: Sequence[str]) -> dict[str, str]:
    """Return axis -> id-label mapping for durable run ids."""

    configured = grid_data.get("axis_id_labels") or {}
    if not isinstance(configured, dict):
        raise ValueError("axis_id_labels must be a mapping")
    return {axis: str(configured.get(axis, axis)) for axis in axes}


def axis_override_paths(grid_data: dict[str, Any], axes: Sequence[str]) -> dict[str, str]:
    """Return axis -> OmegaConf override path mapping."""

    configured = grid_data.get("axis_overrides") or {}
    if not isinstance(configured, dict):
        raise ValueError("axis_overrides must be a mapping")
    missing = [axis for axis in axes if axis not in configured]
    if missing:
        raise ValueError(f"axis_overrides is missing required axes: {', '.join(missing)}")
    return {axis: str(configured[axis]) for axis in axes}


def _id_value(value: Any) -> str:
    """Return a compact value label for durable ids."""

    return axis_value_label(value)


def id_for(point: dict[str, Any], axes: Sequence[str], labels: dict[str, str]) -> str:
    """Return a deterministic id from selected axes."""

    return "_".join(f"{labels.get(axis, axis)}-{_id_value(point[axis])}" for axis in axes)


def _axis_value_overrides(
    point: dict[str, Any],
    *,
    axes: Sequence[str],
    override_paths: dict[str, str],
) -> list[str]:
    return [f"{override_paths[axis]}={point[axis]}" for axis in axes]


def train_overrides(
    point: dict[str, Any],
    *,
    study: str,
    run_id: str,
    attempt_id: str,
    results_root: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, str],
    seed_axis: str,
    seed_policy: dict[str, dict[str, str]] | None = None,
    timezone: str | None = None,
) -> list[str]:
    """Return scalar OmegaConf-style overrides for one train job."""

    seed_overrides = seed_override_values(
        seed_policy,
        "scan_train",
        {"scan_seed": point[seed_axis]},
    )
    overrides = [
        *_axis_value_overrides(point, axes=scalar_axes, override_paths=override_paths),
        *(f"{path}={value}" for path, value in seed_overrides.items()),
        f"run.root={stage_dir(results_root, STAGE_TRAIN)}",
        "run.layout=flat",
        f"run.run_id={run_id}/{attempt_id}",
        f"study.name={study}",
        f"study.attempt_id={attempt_id}",
        f"experiment.name={study}",
        f"experiment.run_name={experiment_run_name(study, 'train')}",
    ]
    if timezone is not None:
        overrides.append(f"run.timezone={timezone}")
    return overrides


def command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    """Return the canonical ``run.py`` command for a config and overrides."""

    return [python, "-u", "run.py", "--config", str(config), *overrides]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def build_jobs(
    points: Sequence[dict[str, Any]],
    *,
    study: str,
    attempt_id: str,
    results_root: str | Path,
    config: str | Path,
    major_axis_names: Sequence[str],
    minor_axis_names: Sequence[str],
    seed_axis: str,
    id_labels: dict[str, str],
    override_paths: dict[str, str],
    tags_by_axis: dict[str, dict[str, list[str]]],
    seed_policy: dict[str, dict[str, str]] | None = None,
    python: str = "python",
    timezone: str | None = None,
) -> list[dict[str, Any]]:
    """Return one manifest job record per grid point."""

    jobs = []
    config_axes = (*major_axis_names, *minor_axis_names)
    run_axes = (*config_axes, seed_axis)
    for point in points:
        run_id = id_for(point, run_axes, id_labels)
        major_id = id_for(point, major_axis_names, id_labels)
        minor_id = id_for(point, minor_axis_names, id_labels)
        config_id = id_for(point, config_axes, id_labels)
        overrides = train_overrides(
            point,
            study=study,
            run_id=run_id,
            attempt_id=attempt_id,
            results_root=results_root,
            scalar_axes=config_axes,
            override_paths=override_paths,
            seed_axis=seed_axis,
            seed_policy=seed_policy,
            timezone=timezone,
        )
        scan_seed_overrides = seed_override_values(
            seed_policy,
            "scan_train",
            {"scan_seed": point[seed_axis]},
        )
        jobs.append(
            {
                "run_id": run_id,
                "major_id": major_id,
                "minor_id": minor_id,
                "config_id": config_id,
                "major_choices": {axis: point[axis] for axis in major_axis_names},
                "minor_choices": {axis: point[axis] for axis in minor_axis_names},
                "scan_seed": point[seed_axis],
                "seed_overrides": {"scan_train": scan_seed_overrides},
                "train_dir": str(train_run_dir(results_root, run_id)),
                "validation_dir": str(validation_run_dir(results_root, run_id)),
                "train_attempt_dir": str(train_attempt_dir(results_root, run_id, attempt_id)),
                "overrides": overrides,
                "command": shlex.join(command_for(config, overrides, python=python)),
                "choices": {axis: point[axis] for axis in run_axes},
                "tags": tags_for_point(point, tags_by_axis),
                "submitted": False,
                "launcher": None,
                "launcher_job_id": None,
            }
        )
    return jobs


def build_manifest(
    *,
    attempt_id: str,
    created_at: str,
    config: str | Path,
    grid: str | Path,
    results_root: str | Path,
    jobs: list[dict[str, Any]],
    grid_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``00_grid`` manifest describing planned jobs."""

    grid_data = grid_data or {}
    manifest = {
        "study": study_name(grid_data.get("study")),
        "stage": STAGE_GRID,
        "attempt_id": attempt_id,
        "created_at": created_at,
        "config": str(config),
        "grid": str(grid),
        "results_root": str(results_root),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    if "major_grid" in grid_data:
        major_axis_names = major_axes(grid_data)
        minor_axis_names = minor_axes(grid_data)
        seed_axis = scan_seed_axis(grid_data)
        all_axes = (*major_axis_names, *minor_axis_names, seed_axis)
        manifest.update(
            {
                "grid_schema": "major_minor_scan",
                "major_axes": list(major_axis_names),
                "minor_axes": list(minor_axis_names),
                "scan_seed_axis": seed_axis,
                "major_grid": grid_data.get("major_grid", {}),
                "minor_grid": grid_data.get("minor_grid", {}),
                "scan_seeds": list(grid_data.get("scan_seeds", [])),
                "axis_id_labels": axis_id_labels(grid_data, all_axes),
                "axis_overrides": axis_override_paths(grid_data, (*major_axis_names, *minor_axis_names)),
                "config_snapshots": config_snapshot_names(grid_data.get("config_snapshots")),
                "choice_validation": choice_validation_specs(grid_data),
                "seed_overrides": seed_override_policy(grid_data.get("seed_overrides")),
                "final_seed_sequences": final_seed_sequences(grid_data.get("final_seed_sequences")),
                "champions": champion_specs(grid_data),
                "champion_kinds": champion_kinds(grid_data),
                "champion_reference_metrics": champion_reference_metrics(grid_data),
                "final_replicates": int(grid_data.get("final_replicates", 0) or 0),
            }
        )
    else:
        axes = grid_axes(grid_data)
        manifest.update(
            {
                "grid_schema": "flat",
                "grid_axes": list(axes),
                "axis_id_labels": axis_id_labels(grid_data, axes),
                "axis_overrides": axis_override_paths(grid_data, axes),
                "config_snapshots": config_snapshot_names(grid_data.get("config_snapshots")),
                "choice_validation": choice_validation_specs(grid_data),
            }
        )
    validation_config = grid_data.get("validation_config")
    if validation_config is not None:
        manifest["validation_config"] = str(validation_config)
    smoke_config = grid_data.get("smoke_config")
    if smoke_config is not None:
        manifest["smoke_config"] = str(smoke_config)
    return manifest


def write_grid_attempt(
    *,
    results_root: str | Path,
    attempt_id: str,
    created_at: str,
    config: str | Path,
    grid: str | Path,
    grid_data: Any,
    jobs: list[dict[str, Any]],
) -> Path:
    """Write the durable ``00_grid`` attempt and return its directory."""

    attempt = grid_attempt_dir(results_root, attempt_id)
    (attempt / "jobs").mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        attempt_id=attempt_id,
        created_at=created_at,
        config=config,
        grid=grid,
        results_root=results_root,
        jobs=jobs,
        grid_data=grid_data if isinstance(grid_data, dict) else None,
    )
    write_json(attempt / "manifest.json", manifest)

    # Snapshot the inputs that produced this plan.
    OmegaConf.save(OmegaConf.create(grid_data), attempt / "grid.yaml")
    snapshots = config_snapshot_names(
        grid_data.get("config_snapshots") if isinstance(grid_data, dict) else None
    )
    config_text = Path(config).read_text() if Path(config).exists() else ""
    (attempt / snapshots["train"]).write_text(config_text)

    # Exact commands that train.py will read from this attempt.
    study = study_name(grid_data.get("study")) if isinstance(grid_data, dict) else study_name()
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# {study} 00_grid attempt {attempt_id}", ""]
    lines += [job["command"] for job in jobs]
    (attempt / "commands.sh").write_text("\n".join(lines) + "\n")

    validation_config = grid_data.get("validation_config") if isinstance(grid_data, dict) else None
    if validation_config is not None and Path(validation_config).exists():
        (attempt / snapshots["validation"]).write_text(Path(validation_config).read_text())
    smoke_config = grid_data.get("smoke_config") if isinstance(grid_data, dict) else None
    smoke_snapshot = snapshots.get("smoke")
    if smoke_config is not None and smoke_snapshot and Path(smoke_config).exists():
        (attempt / smoke_snapshot).write_text(Path(smoke_config).read_text())

    # Per-job specs for downstream stages.
    for job in jobs:
        write_json(attempt / "jobs" / f"{job['run_id']}.json", job)

    write_latest(stage_dir(results_root, STAGE_GRID), attempt_id)
    return attempt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse planner command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default=str(DEFAULT_GRID), help="Grid YAML path.")
    parser.add_argument("--config", default=None, help="Train config path (defaults to grid.config).")
    parser.add_argument("--results-root", default=None, help="Results root (defaults to grid.results_root).")
    parser.add_argument(
        "--attempt-id", default=None, help="Attempt id in the study timezone (defaults to now)."
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_STUDY_TIMEZONE,
        help=(
            "IANA timezone owned by the planner: stamps attempt ids and "
            "overrides run.timezone (default America/New_York)."
        ),
    )
    parser.add_argument("--tags", nargs="*", default=None, help="Only include grid points with all configured tags.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of planned jobs.")
    parser.add_argument(
        "--python",
        default="python",
        help=(
            "Python executable name recorded in planned commands. The train "
            "launcher chooses the CPU/CUDA uv environment at launch time."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan the configured study grid and write a ``00_grid`` attempt."""

    args = parse_args(argv)
    grid_path = Path(args.grid)
    grid_data = OmegaConf.to_container(OmegaConf.load(grid_path), resolve=True)
    config = args.config or grid_data["config"]
    results_root = args.results_root or grid_data["results_root"]
    study = study_name(grid_data.get("study"))
    prefix = log_prefix(study)

    # The planner owns the timezone for this study: it stamps the attempt id /
    # created_at and injects run.timezone into the compiled train commands.
    tz = resolve_timezone(args.timezone)
    attempt_id = args.attempt_id or new_attempt_id(tz=tz)
    created_at = datetime.now(tz).isoformat(timespec="seconds")

    major_axis_names = major_axes(grid_data)
    minor_axis_names = minor_axes(grid_data)
    seed_axis = scan_seed_axis(grid_data)
    config_axes = (*major_axis_names, *minor_axis_names)
    all_axes = (*config_axes, seed_axis)
    points = expand_grid_spec(grid_data)
    config_obj = OmegaConf.load(config)
    validation_specs = choice_validation_specs(grid_data)
    validate_grid(points, config_obj, validation_specs)
    tags_by_axis = axis_tags(config_obj, validation_specs)
    seed_policy = seed_override_policy(grid_data.get("seed_overrides"))
    id_labels = axis_id_labels(grid_data, all_axes)
    override_paths = axis_override_paths(grid_data, config_axes)

    if args.tags:
        wanted = set(args.tags)
        kept = [p for p in points if wanted.issubset(set(tags_for_point(p, tags_by_axis)))]
        if len(kept) < len(points):
            print(f"{prefix} tag filter {sorted(wanted)}: {len(kept)}/{len(points)} jobs kept")
        points = kept
    if args.limit is not None and args.limit < len(points):
        print(f"{prefix} --limit {args.limit}: dropping {len(points) - args.limit} of {len(points)} jobs")
        points = points[: args.limit]

    jobs = build_jobs(
        points,
        study=study,
        attempt_id=attempt_id,
        results_root=results_root,
        config=config,
        major_axis_names=major_axis_names,
        minor_axis_names=minor_axis_names,
        seed_axis=seed_axis,
        id_labels=id_labels,
        override_paths=override_paths,
        tags_by_axis=tags_by_axis,
        seed_policy=seed_policy,
        python=args.python,
        timezone=args.timezone,
    )
    attempt = write_grid_attempt(
        results_root=results_root,
        attempt_id=attempt_id,
        created_at=created_at,
        config=config,
        grid=grid_path,
        grid_data=grid_data,
        jobs=jobs,
    )
    print(f"{prefix} wrote 00_grid attempt {attempt_id} with {len(jobs)} jobs -> {attempt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
