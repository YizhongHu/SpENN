"""Launch Hooke pair-stability train jobs from a planned ``00_grid`` attempt.

This script is intentionally a stage consumer: it reads a durable grid manifest
written by ``plan.py`` and emits training work into ``01_train``. It does not
expand grids, write ``00_grid`` attempts, or regenerate run commands.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any, Sequence

import launch
from run_utils import STAGE_TRAIN, grid_attempt_dir, stage_dir, write_json

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

SMOKE_OVERRIDES = {
    "training.max_steps": 2,
    "training.log_every_n_steps": 1,
    "sampler_params.n_walkers": 128,
    "sampler_params.burn_in": 10,
    "sampler_params.n_steps": 5,
    "validation_sampler_params.n_walkers": 128,
    "validation_sampler_params.burn_in": 10,
    "validation_sampler_params.n_steps": 5,
    "checks.every_n_steps": 1,
    "checkpoint.every_n_steps": 1,
    "status.every_n_steps": 1,
}


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


def _smoke_overrides(job: dict[str, Any], *, grid_attempt_id: str) -> dict[str, object]:
    """Return smoke-only command overrides for one selected job."""

    smoke_attempt = launch.smoke_attempt_id(grid_attempt_id)
    return {
        **SMOKE_OVERRIDES,
        "run.run_id": _smoke_run_id(str(job["run_id"]), grid_attempt_id),
        "study.attempt_id": smoke_attempt,
    }


def _execution_command(command: Sequence[str], job: dict[str, Any], *, grid_attempt_id: str, smoke: bool) -> list[str]:
    """Return the run command after launch-mode overrides are applied."""

    if not smoke:
        return [str(part) for part in command]
    return launch.with_overrides(command, _smoke_overrides(job, grid_attempt_id=grid_attempt_id))


def _train_attempt_dir(job: dict[str, Any], *, manifest: dict[str, Any], repo_root: Path) -> Path:
    if job.get("train_attempt_dir"):
        return launch.repo_path(str(job["train_attempt_dir"]), repo_root)
    if job.get("train_dir"):
        return launch.repo_path(str(job["train_dir"]), repo_root) / str(manifest["attempt_id"])
    raise ValueError(f"job {job.get('run_id', '<unknown>')!r} has no train attempt path")


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

    args = parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    results_root = launch.repo_path(args.results_root, repo_root)
    grid_attempt_id = launch.resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = launch.load_grid_manifest(results_root, grid_attempt_id)
    jobs = _launch_jobs(list(manifest.get("jobs", [])), grid_attempt_id=grid_attempt_id, smoke=args.smoke)
    commands = [
        _execution_command(launch.command_for_job(job), job, grid_attempt_id=grid_attempt_id, smoke=args.smoke)
        for job in jobs
    ]
    uv_environment, uv_extras, runtime_device = launch.resolve_uv_settings(args)
    submitted_commands = [
        launch.environment_shell_command(
            command,
            repo_root=repo_root,
            uv_environment=uv_environment,
            uv_extras=uv_extras,
            device=runtime_device,
        )
        for command in commands
    ]

    if not jobs:
        print(f"[pair_stability] grid attempt {grid_attempt_id} has no jobs")
        return 0

    if args.backend == "local":
        job_ids = launch.submit_local(submitted_commands, repo_root=repo_root)
    else:
        log_attempt = launch.smoke_attempt_id(grid_attempt_id) if args.smoke else grid_attempt_id
        job_ids = launch.submit_submitit(
            submitted_commands,
            log_dir=stage_dir(results_root, STAGE_TRAIN) / "slurm_logs" / log_attempt,
            job_name="hooke-pair-stability-train-smoke" if args.smoke else "hooke-pair-stability-train",
            slurm=launch.slurm_parameters(args, profile=args.profile, smoke=args.smoke),
        )

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
    print(f"[pair_stability] launched {len(job_ids)} {mode} jobs from 00_grid/{grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
