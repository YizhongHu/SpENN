"""Launch validation jobs into ``02_validation``.

Validation consumes an existing ``00_grid`` attempt and selected completed
``01_train`` attempts. It writes per-validation provenance and launches the
validation config recorded by the grid manifest.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

import launch
from run_utils import (
    STAGE_VALIDATION,
    attempt_ids,
    config_snapshot_names,
    experiment_run_name,
    grid_attempt_dir,
    log_prefix,
    stage_dir,
    seed_override_values,
    stage_job_name,
    study_name_from_manifest,
    train_attempt_dir,
    train_run_dir,
    validation_attempt_dir,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
DEFAULT_GRID = STUDY_DIR / "configs" / "grid.yaml"

def _scalar_axes(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return non-seed axes recorded in a grid manifest."""

    if manifest.get("grid_schema") == "major_minor_scan":
        return tuple(str(axis) for axis in (*manifest.get("major_axes", []), *manifest.get("minor_axes", [])))
    seed_axis = str(manifest.get("scan_seed_axis", "seed"))
    return tuple(str(axis) for axis in manifest.get("grid_axes", []) if str(axis) != seed_axis)


def _axis_override_paths(manifest: dict[str, Any], axes: Sequence[str]) -> dict[str, str]:
    """Return axis -> config override path from a grid manifest."""

    configured = manifest.get("axis_overrides")
    if not isinstance(configured, dict):
        raise ValueError("grid manifest axis_overrides must be a mapping")
    missing = [axis for axis in axes if axis not in configured]
    if missing:
        raise ValueError(f"grid manifest axis_overrides is missing axes: {', '.join(missing)}")
    return {axis: str(configured[axis]) for axis in axes}


def _axis_value_overrides(
    point: dict[str, Any],
    *,
    scalar_axes: Sequence[str],
    override_paths: dict[str, str],
) -> list[str]:
    overrides = []
    for axis in scalar_axes:
        if axis not in point:
            raise ValueError(f"job choices are missing configured axis {axis!r}")
        overrides.append(f"{override_paths[axis]}={point[axis]}")
    return overrides


def _command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    """Return the canonical ``run.py`` command for a validation config."""

    return [python, "-u", "run.py", "--config", str(config), *overrides]


def _job_timezone(job: dict[str, Any]) -> str | None:
    """Return the train job's planned run timezone override, if present."""

    for override in job.get("overrides", []):
        text = str(override)
        if text.startswith("run.timezone="):
            return text.split("=", 1)[1]
    return None


def validation_overrides(
    point: dict[str, Any],
    *,
    study: str,
    run_id: str,
    attempt_id: str,
    results_root: str | Path,
    checkpoint_path: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, str],
    seed_axis: str,
    seed_policy: dict[str, dict[str, str]] | None = None,
    timezone: str | None = None,
) -> list[str]:
    """Return scalar OmegaConf-style overrides for one validation job."""

    seed_overrides = seed_override_values(
        seed_policy,
        "validation",
        {"scan_seed": int(point[seed_axis])},
    )
    overrides = [
        *_axis_value_overrides(point, scalar_axes=scalar_axes, override_paths=override_paths),
        *(f"{path}={value}" for path, value in seed_overrides.items()),
        f"load.path={checkpoint_path}",
        f"run.root={stage_dir(results_root, STAGE_VALIDATION)}",
        "run.layout=flat",
        f"run.run_id={run_id}/{attempt_id}",
        f"study.name={study}",
        f"study.attempt_id={attempt_id}",
        f"experiment.name={study}",
        f"experiment.run_name={experiment_run_name(study, 'validation')}",
    ]
    if timezone is not None:
        overrides.append(f"run.timezone={timezone}")
    return overrides


def _is_smoke_attempt(attempt_id: str) -> bool:
    """Return whether an attempt id is marked as smoke."""

    return attempt_id.endswith("-smoke")


def latest_train_attempt_id(results_root: str | Path, run_id: str, *, smoke: bool) -> str | None:
    """Return the latest eligible train attempt for ``run_id``."""

    ids = attempt_ids(train_run_dir(results_root, run_id))
    candidates = [attempt_id for attempt_id in ids if _is_smoke_attempt(attempt_id) == smoke]
    return candidates[-1] if candidates else None


def _checkpoint_ready(train_attempt: Path) -> bool:
    """Return whether a train attempt exposes a latest checkpoint pointer."""

    return (train_attempt / "checkpoints" / "latest.json").is_file()


def _validation_config_from_grid(
    *,
    results_root: Path,
    grid_attempt_id: str,
    requested_config: str | None,
) -> str:
    if requested_config is not None:
        return requested_config
    grid_attempt = grid_attempt_dir(results_root, grid_attempt_id)
    manifest_path = grid_attempt / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        snapshots = config_snapshot_names(manifest.get("config_snapshots"))
        validation_snapshot = grid_attempt / snapshots["validation"]
        if validation_snapshot.is_file():
            return str(validation_snapshot)
        validation_config = manifest.get("validation_config")
        if validation_config:
            return str(validation_config)
    grid_snapshot = grid_attempt_dir(results_root, grid_attempt_id) / "grid.yaml"
    if grid_snapshot.is_file():
        grid_data = OmegaConf.to_container(OmegaConf.load(grid_snapshot), resolve=True)
        if isinstance(grid_data, dict):
            snapshots = config_snapshot_names(grid_data.get("config_snapshots"))
            validation_snapshot = grid_attempt / snapshots["validation"]
            if validation_snapshot.is_file():
                return str(validation_snapshot)
            config = grid_data.get("validation_config")
        else:
            config = None
        if config:
            return str(config)
    if DEFAULT_GRID.is_file():
        grid_data = OmegaConf.to_container(OmegaConf.load(DEFAULT_GRID), resolve=True)
        config = grid_data.get("validation_config") if isinstance(grid_data, dict) else None
        if config:
            return str(config)
    raise FileNotFoundError("validation config was not requested and no grid validation_config was found")


def _selected_jobs(jobs: Sequence[dict[str, Any]], *, smoke: bool) -> list[dict[str, Any]]:
    if not smoke:
        return [dict(job) for job in jobs]
    return [dict(job) for job in list(jobs)[: launch.SMOKE_JOB_LIMIT]]


def _train_attempt_id_for_job(
    *,
    args: argparse.Namespace,
    results_root: Path,
    grid_attempt_id: str,
    run_id: str,
) -> str | None:
    if args.train_attempt_id is not None:
        if not args.smoke and _is_smoke_attempt(args.train_attempt_id):
            raise ValueError("full validation refuses a smoke train attempt; pass --smoke for smoke validation")
        return args.train_attempt_id
    if args.smoke:
        smoke_id = launch.smoke_attempt_id(grid_attempt_id)
        if train_attempt_dir(results_root, run_id, smoke_id).is_dir():
            return smoke_id
        return latest_train_attempt_id(results_root, run_id, smoke=True)
    return latest_train_attempt_id(results_root, run_id, smoke=False)


def plan_validation_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    study: str,
    results_root: Path,
    grid_attempt_id: str,
    validation_config: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, str],
    seed_axis: str,
    smoke_overrides: dict[str, object],
    seed_policy: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build validation launch records and write source-train provenance."""

    selected = _selected_jobs(jobs, smoke=args.smoke)
    validation_attempt_id = args.attempt_id or (
        launch.smoke_attempt_id(grid_attempt_id) if args.smoke else grid_attempt_id
    )
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for job in selected:
        run_id = str(job["run_id"])
        train_attempt_id = _train_attempt_id_for_job(
            args=args,
            results_root=results_root,
            grid_attempt_id=grid_attempt_id,
            run_id=run_id,
        )
        if train_attempt_id is None:
            skipped.append({"run_id": run_id, "reason": "no eligible train attempt"})
            continue
        train_attempt = train_attempt_dir(results_root, run_id, train_attempt_id)
        if not _checkpoint_ready(train_attempt):
            skipped.append({"run_id": run_id, "reason": f"missing checkpoint in {train_attempt}"})
            continue

        point = dict(job.get("choices", {}))
        if seed_axis not in point and "scan_seed" in job:
            point[seed_axis] = job["scan_seed"]
        checkpoint_path = train_attempt / "checkpoints"
        validation_attempt = validation_attempt_dir(results_root, run_id, validation_attempt_id)
        source = {
            "run_id": run_id,
            "grid_attempt_id": grid_attempt_id,
            "train_attempt_id": train_attempt_id,
            "train_dir": str(train_run_dir(results_root, run_id)),
            "train_attempt_dir": str(train_attempt),
            "checkpoint_path": str(checkpoint_path),
        }
        write_json(validation_attempt / "source_train_attempt.json", source)
        write_json(
            validation_attempt / "source_grid_attempt.json",
            {
                "run_id": run_id,
                "grid_attempt_id": grid_attempt_id,
                "grid_attempt_dir": str(grid_attempt_dir(results_root, grid_attempt_id)),
            },
        )

        overrides = validation_overrides(
            point,
            study=study,
            run_id=run_id,
            attempt_id=validation_attempt_id,
            results_root=results_root,
            checkpoint_path=checkpoint_path,
            scalar_axes=scalar_axes,
            override_paths=override_paths,
            seed_axis=seed_axis,
            seed_policy=seed_policy,
            timezone=_job_timezone(job),
        )
        command = _command_for(validation_config, overrides)
        command = launch.with_study_timezone(command, timezone=_job_timezone(job))
        if args.smoke:
            command = launch.with_overrides(command, smoke_overrides)
        planned.append(
            {
                "run_id": run_id,
                "train_attempt_id": train_attempt_id,
                "validation_attempt_id": validation_attempt_id,
                "validation_attempt_dir": str(validation_attempt),
                "source_train_attempt": source,
                "command": shlex.join(command),
                "command_parts": command,
            }
        )
    return planned, skipped


def write_validation_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    grid_attempt_id: str,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write validation-stage submission provenance."""

    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        validation_attempt = Path(str(job["validation_attempt_dir"]))
        write_json(
            validation_attempt / "submission.json",
            {
                "run_id": str(job["run_id"]),
                "grid_attempt_id": grid_attempt_id,
                "train_attempt_id": str(job["train_attempt_id"]),
                "validation_attempt_id": str(job["validation_attempt_id"]),
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": str(job["command"]),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse validation command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--grid-attempt-id", default=None, help="Grid attempt to validate (defaults to latest).")
    parser.add_argument("--config", default=None, help="Validation config path (defaults to grid.validation_config).")
    parser.add_argument(
        "--train-attempt-id",
        default=None,
        help="Exact train attempt to validate. Full validation rejects smoke attempts.",
    )
    parser.add_argument(
        "--attempt-id",
        default=None,
        help="Validation attempt id (defaults to the grid attempt id, or grid-smoke with --smoke).",
    )
    launch.add_launch_arguments(
        parser,
        smoke_help=(
            "Validate the first two smoke-capable jobs with smoke-marked attempt ids, "
            "small evaluation grids, and 15-minute test partitions."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch validation jobs from existing ``00_grid`` and ``01_train`` attempts."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    results_root = launch.repo_path(args.results_root, repo_root)
    grid_attempt_id = launch.resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = launch.load_grid_manifest(results_root, grid_attempt_id)
    study = study_name_from_manifest(manifest)
    prefix = log_prefix(study)
    if args.wait_job:
        launch.submit_dependent_launcher(
            args.wait_job,
            script_path=Path(__file__).resolve(),
            argv=raw_argv,
            repo_root=repo_root,
            log_dir=stage_dir(results_root, STAGE_VALIDATION) / "slurm_logs" / "dependent_launchers",
            job_name=stage_job_name(study, "validate-launcher", smoke=args.smoke),
            partition=args.wait_launcher_partition,
            timeout_min=args.wait_launcher_timeout_min,
            study=study,
        )
        return 0
    seed_policy = manifest.get("seed_overrides")
    grid_dir = grid_attempt_dir(results_root, grid_attempt_id)
    configured_smoke_overrides = launch.load_smoke_overrides(
        "validation",
        manifest=manifest,
        attempt_dir=grid_dir,
    )
    scalar_axes = _scalar_axes(manifest)
    override_paths = _axis_override_paths(manifest, scalar_axes)
    seed_axis = str(manifest.get("scan_seed_axis", "seed"))
    validation_config = _validation_config_from_grid(
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        requested_config=args.config,
    )
    jobs, skipped = plan_validation_jobs(
        list(manifest.get("jobs", [])),
        args=args,
        study=study,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        validation_config=validation_config,
        scalar_axes=scalar_axes,
        override_paths=override_paths,
        seed_axis=seed_axis,
        smoke_overrides=configured_smoke_overrides,
        seed_policy=seed_policy,
    )
    uv_environment, uv_extras, runtime_device = launch.resolve_uv_settings(args)
    submitted_commands = [
        launch.environment_shell_command(
            job["command_parts"],
            repo_root=repo_root,
            uv_environment=uv_environment,
            uv_extras=uv_extras,
            device=runtime_device,
        )
        for job in jobs
    ]

    if skipped:
        print(f"{prefix} skipped {len(skipped)} validation jobs without eligible checkpoints")
    if not jobs:
        print(f"{prefix} no validation jobs ready for 00_grid/{grid_attempt_id}")
        return 1 if manifest.get("jobs") else 0

    row_status_paths = [Path(str(job["validation_attempt_dir"])) / "launcher_status.json" for job in jobs]
    chunk_status_dir = (
        stage_dir(results_root, STAGE_VALIDATION)
        / "chunk_status"
        / (launch.smoke_attempt_id(grid_attempt_id) if args.smoke else (args.attempt_id or grid_attempt_id))
    )

    if args.backend == "local":
        job_ids = launch.submit_local(
            submitted_commands,
            repo_root=repo_root,
            chunk_size=args.chunk_size,
            allow_partial_failures=True,
            row_status_paths=row_status_paths,
            chunk_status_dir=chunk_status_dir,
        )
    else:
        log_attempt = launch.smoke_attempt_id(grid_attempt_id) if args.smoke else (args.attempt_id or grid_attempt_id)
        job_ids = launch.submit_submitit(
            submitted_commands,
            log_dir=stage_dir(results_root, STAGE_VALIDATION) / "slurm_logs" / log_attempt,
            job_name=stage_job_name(study, "validate", smoke=args.smoke),
            slurm=launch.slurm_parameters(args, profile=args.profile, smoke=args.smoke),
            chunk_size=args.chunk_size,
            allow_partial_failures=True,
            row_status_paths=row_status_paths,
            chunk_status_dir=chunk_status_dir,
        )

    write_validation_submission_records(
        jobs,
        grid_attempt_id=grid_attempt_id,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    mode = "smoke validation" if args.smoke else "validation"
    print(f"{prefix} launched {len(job_ids)} {mode} jobs from 00_grid/{grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
