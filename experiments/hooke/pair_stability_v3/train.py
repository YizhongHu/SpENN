"""Launch train jobs from a planned ``00_grid`` attempt.

This script is intentionally a stage consumer: it reads a durable grid manifest
written by ``plan.py`` and emits training work into ``01_train``. It does not
expand grids, write ``00_grid`` attempts, or regenerate run commands.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

import launch
from utils.io import write_json
from utils.layout import (
    STAGE_TRAIN,
    grid_attempt_dir,
    stage_dir,
    write_latest,
)
from utils.naming import (
    log_prefix,
    stage_job_name,
    study_name_from_manifest,
)

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit import (  # noqa: E402
    StagePlan,
    resource_spec_from_launcher,
    stage_plan_directory,
    submit_stage_plan,
)
from experiments.toolkit.specs import tasks_from_commands  # noqa: E402

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

def _smoke_run_id(run_id: str, grid_attempt_id: str) -> str:
    """Return the flat-layout run id for a smoke train attempt."""

    return f"{run_id}/{launch.smoke_attempt_id(grid_attempt_id)}"


def _smoke_job(job: dict[str, Any], *, grid_attempt_id: str) -> dict[str, Any]:
    """Return a manifest job copy redirected to its smoke attempt directory."""

    job = dict(job)
    train_dir = Path(str(job["train_dir"]))
    job["train_attempt_dir"] = str(train_dir / launch.smoke_attempt_id(grid_attempt_id))
    return job


def _launch_jobs(jobs: Sequence[dict[str, Any]], *, grid_attempt_id: str, smoke: bool) -> list[dict[str, Any]]:
    """Return the jobs selected for this launch."""

    if not smoke:
        return [dict(job) for job in jobs]
    return [_smoke_job(job, grid_attempt_id=grid_attempt_id) for job in list(jobs)[: launch.SMOKE_JOB_LIMIT]]


def _smoke_overrides(
    job: dict[str, Any],
    *,
    grid_attempt_id: str,
    configured: dict[str, object],
) -> dict[str, object]:
    """Return smoke-only command overrides for one selected job."""

    smoke_attempt = launch.smoke_attempt_id(grid_attempt_id)
    return {
        **configured,
        "run.run_id": _smoke_run_id(str(job["run_id"]), grid_attempt_id),
        "study.attempt_id": smoke_attempt,
    }


def _execution_command(
    command: Sequence[str],
    job: dict[str, Any],
    *,
    grid_attempt_id: str,
    smoke: bool,
    smoke_overrides: dict[str, object],
) -> list[str]:
    """Return the run command after launch-mode overrides are applied."""

    if not smoke:
        return [str(part) for part in command]
    return launch.with_overrides(
        command,
        _smoke_overrides(job, grid_attempt_id=grid_attempt_id, configured=smoke_overrides),
    )


def _train_attempt_dir(job: dict[str, Any], *, manifest: dict[str, Any], repo_root: Path) -> Path:
    if job.get("train_attempt_dir"):
        return launch.repo_path(str(job["train_attempt_dir"]), repo_root)
    if job.get("train_dir"):
        return launch.repo_path(str(job["train_dir"]), repo_root) / str(manifest["attempt_id"])
    raise ValueError(f"job {job.get('run_id', '<unknown>')!r} has no train attempt path")


def build_train_stage_plan(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    grid_attempt_id: str,
    repo_root: Path,
    commands: Sequence[Sequence[str]],
    row_status_paths: Sequence[Path],
    args: argparse.Namespace,
) -> StagePlan:
    """Build a reusable toolkit stage plan for train tasks."""

    attempt_id = launch.smoke_attempt_id(grid_attempt_id) if args.smoke else grid_attempt_id
    result_dirs = [
        _train_attempt_dir(job, manifest=manifest, repo_root=repo_root)
        for job in jobs
    ]
    checkpoint_paths = [Path(result_dir) / "checkpoints" / "latest.json" for result_dir in result_dirs]
    tasks = tasks_from_commands(
        stage=STAGE_TRAIN,
        attempt_id=attempt_id,
        jobs=jobs,
        commands=commands,
        result_dirs=result_dirs,
        row_status_paths=row_status_paths,
        resources=resource_spec_from_launcher(launch, args),
        completion_policy="status_completed_with_checkpoint",
        checkpoint_paths=checkpoint_paths,
        source_attempts={"grid": grid_attempt_id},
    )
    return StagePlan(
        study=study_name_from_manifest(manifest),
        stage=STAGE_TRAIN,
        attempt_id=attempt_id,
        results_root=str(results_root),
        source_attempts={"grid": grid_attempt_id},
        timezone=manifest.get("timezone"),
        smoke=bool(args.smoke),
        metadata={
            "backend": args.backend,
            "device": launch.selected_device(args),
            "chunk_size": args.chunk_size,
        },
        tasks=tasks,
    )


def write_train_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    grid_attempt_id: str,
    repo_root: Path,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write train-stage provenance without mutating the ``00_grid`` manifest."""

    manifest_path = grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json"
    grid_dir = grid_attempt_dir(results_root, grid_attempt_id)
    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        train_attempt = _train_attempt_dir(job, manifest=manifest, repo_root=repo_root)
        source = {
            "run_id": str(job["run_id"]),
            "grid_attempt_id": grid_attempt_id,
            "grid_attempt_dir": str(grid_dir),
            "manifest_path": str(manifest_path),
        }
        write_json(train_attempt / "source_grid_attempt.json", source)
        write_json(
            train_attempt / "submission.json",
            {
                "run_id": str(job["run_id"]),
                "grid_attempt_id": grid_attempt_id,
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": job.get("command", ""),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def write_train_launch_provenance(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    grid_attempt_id: str,
    repo_root: Path,
    submitted_commands: Sequence[Sequence[str]],
    smoke: bool = False,
) -> list[Path]:
    """Create train attempt directories before scheduler execution starts."""

    manifest_path = grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json"
    grid_dir = grid_attempt_dir(results_root, grid_attempt_id)
    row_status_paths: list[Path] = []
    for index, job in enumerate(jobs):
        train_attempt = _train_attempt_dir(job, manifest=manifest, repo_root=repo_root)
        source = {
            "run_id": str(job["run_id"]),
            "grid_attempt_id": grid_attempt_id,
            "grid_attempt_dir": str(grid_dir),
            "manifest_path": str(manifest_path),
        }
        write_json(train_attempt / "source_grid_attempt.json", source)
        (train_attempt / "command.txt").write_text(
            shlex.join([str(part) for part in submitted_commands[index]]) + "\n"
        )
        write_latest(train_attempt.parent, train_attempt.name, smoke=smoke)
        row_status_paths.append(train_attempt / "launcher_status.json")
    return row_status_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse train command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--grid-attempt-id", default=None, help="Grid attempt to launch (defaults to latest).")
    launch.add_launch_arguments(
        parser,
        smoke_help=(
            "Submit two short smoke jobs with smoke-marked attempt ids, small "
            "samplers, two train steps, and 15-minute test partitions."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch train jobs from an existing ``00_grid`` attempt."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    launch.ensure_submitit_launcher_environment(
        args,
        script_path=Path(__file__).resolve(),
        argv=raw_argv,
        repo_root=repo_root,
    )
    results_root = launch.repo_path(args.results_root, repo_root)
    grid_attempt_id = launch.resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = launch.load_grid_manifest(results_root, grid_attempt_id)
    study = study_name_from_manifest(manifest)
    prefix = log_prefix(study)
    grid_dir = grid_attempt_dir(results_root, grid_attempt_id)
    configured_smoke_overrides = launch.load_smoke_overrides(
        "train",
        manifest=manifest,
        attempt_dir=grid_dir,
    )
    jobs = _launch_jobs(list(manifest.get("jobs", [])), grid_attempt_id=grid_attempt_id, smoke=args.smoke)
    commands = [
        _execution_command(
            launch.command_for_job(job),
            job,
            grid_attempt_id=grid_attempt_id,
            smoke=args.smoke,
            smoke_overrides=configured_smoke_overrides,
        )
        for job in jobs
    ]
    command_sets = launch.environment_command_sets(commands, args=args, repo_root=repo_root)
    submitted_commands = launch.summarize_command_sets(command_sets)

    if not jobs:
        print(f"{prefix} grid attempt {grid_attempt_id} has no jobs")
        return 0

    row_status_paths = write_train_launch_provenance(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        repo_root=repo_root,
        submitted_commands=submitted_commands,
        smoke=args.smoke,
    )
    log_attempt = launch.smoke_attempt_id(grid_attempt_id) if args.smoke else grid_attempt_id
    stage_plan = build_train_stage_plan(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        repo_root=repo_root,
        commands=commands,
        row_status_paths=row_status_paths,
        args=args,
    )
    execution_records = submit_stage_plan(
        launch,
        stage_plan=stage_plan,
        stage_plan_dir=stage_plan_directory(results_root, STAGE_TRAIN, log_attempt),
        command_sets=command_sets,
        submitted_commands=submitted_commands,
        args=args,
        repo_root=repo_root,
        log_dir=stage_dir(results_root, STAGE_TRAIN) / "slurm_logs" / log_attempt,
        job_name=stage_job_name(study, "train", smoke=args.smoke),
        chunk_status_dir=stage_dir(results_root, STAGE_TRAIN) / "chunk_status" / log_attempt,
    )
    job_ids = [record.launcher_job_id for record in execution_records]

    write_train_submission_records(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        repo_root=repo_root,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    mode = "smoke train" if args.smoke else "train"
    print(f"{prefix} launched {len(job_ids)} {mode} jobs from 00_grid/{grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
