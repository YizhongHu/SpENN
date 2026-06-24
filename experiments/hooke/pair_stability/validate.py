"""Launch Hooke pair-stability validation jobs into ``02_validation``.

Validation consumes an existing ``00_grid`` attempt and selected completed
``01_train`` attempts. It writes per-validation provenance and launches
``configs/pair_validation.yaml`` through the shared launch plumbing.
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
    GRID_AXES,
    STAGE_VALIDATION,
    attempt_ids,
    grid_attempt_dir,
    stage_dir,
    train_attempt_dir,
    train_run_dir,
    validation_attempt_dir,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
DEFAULT_VALIDATION_CONFIG = STUDY_DIR / "configs" / "pair_validation.yaml"

SMOKE_VALIDATION_OVERRIDES = {
    "evaluation_tasks.cusp.generator.n_points": 4,
    "evaluation_tasks.cusp.generator.n_directions": 2,
    "evaluation_tasks.cusp.generator.center_of_mass_radii": [0.0],
    "evaluation_tasks.tail.generator.n_points": 4,
    "evaluation_tasks.tail.generator.n_directions": 2,
    "evaluation_tasks.stratified_geometry.generator.n_samples": 16,
    "evaluation_tasks.hooke_orbital.generator.n_samples": 16,
    "evaluation_tasks.full_model_antisymmetry.generator.base_generator.n_samples": 8,
    "evaluation_tasks.trace_equivariance.generator.base_generator.n_samples": 4,
    "evaluation_tasks.feature_trace_stability.generator.n_samples": 8,
    "evaluation_tasks.readout_trace_stability.generator.n_samples": 8,
}


def _run_parameter_overrides(point: dict[str, Any]) -> list[str]:
    return [
        f"run_parameters.architecture={point['architecture']}",
        f"run_parameters.normalization={point['normalization']}",
        f"run_parameters.lr={point['lr']}",
        f"run_parameters.channels={int(point['channels'])}",
        f"run_parameters.seed={int(point['seed'])}",
    ]


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
    run_id: str,
    attempt_id: str,
    results_root: str | Path,
    checkpoint_path: str | Path,
    timezone: str | None = None,
) -> list[str]:
    """Return scalar OmegaConf-style overrides for one validation job."""

    overrides = [
        *_run_parameter_overrides(point),
        f"load.path={checkpoint_path}",
        f"run.root={stage_dir(results_root, STAGE_VALIDATION)}",
        "run.layout=flat",
        f"run.run_id={run_id}/{attempt_id}",
        f"study.attempt_id={attempt_id}",
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
    validation_snapshot = grid_attempt_dir(results_root, grid_attempt_id) / "pair_validation.yaml"
    if validation_snapshot.is_file():
        return str(validation_snapshot)
    grid_snapshot = grid_attempt_dir(results_root, grid_attempt_id) / "grid.yaml"
    if grid_snapshot.is_file():
        grid_data = OmegaConf.to_container(OmegaConf.load(grid_snapshot), resolve=True)
        config = grid_data.get("validation_config") if isinstance(grid_data, dict) else None
        if config:
            return str(config)
    return str(DEFAULT_VALIDATION_CONFIG)


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
    results_root: Path,
    grid_attempt_id: str,
    validation_config: str | Path,
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

        point = {axis: job["choices"][axis] for axis in GRID_AXES}
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
            run_id=run_id,
            attempt_id=validation_attempt_id,
            results_root=results_root,
            checkpoint_path=checkpoint_path,
            timezone=_job_timezone(job),
        )
        command = _command_for(validation_config, overrides)
        command = launch.with_study_timezone(command, timezone=_job_timezone(job))
        if args.smoke:
            command = launch.with_overrides(command, SMOKE_VALIDATION_OVERRIDES)
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
    if args.wait_job:
        launch.submit_dependent_launcher(
            args.wait_job,
            script_path=Path(__file__).resolve(),
            argv=raw_argv,
            repo_root=repo_root,
            log_dir=stage_dir(results_root, STAGE_VALIDATION) / "slurm_logs" / "dependent_launchers",
            job_name="hooke-pair-stability-validate-launcher-smoke"
            if args.smoke
            else "hooke-pair-stability-validate-launcher",
            partition=args.wait_launcher_partition,
            timeout_min=args.wait_launcher_timeout_min,
        )
        return 0
    grid_attempt_id = launch.resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = launch.load_grid_manifest(results_root, grid_attempt_id)
    validation_config = _validation_config_from_grid(
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        requested_config=args.config,
    )
    jobs, skipped = plan_validation_jobs(
        list(manifest.get("jobs", [])),
        args=args,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        validation_config=validation_config,
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
        print(f"[pair_stability] skipped {len(skipped)} validation jobs without eligible checkpoints")
    if not jobs:
        print(f"[pair_stability] no validation jobs ready for 00_grid/{grid_attempt_id}")
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
            job_name="hooke-pair-stability-validate-smoke" if args.smoke else "hooke-pair-stability-validate",
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
    print(f"[pair_stability] launched {len(job_ids)} {mode} jobs from 00_grid/{grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
