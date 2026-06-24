"""Launch final training from ``05_final_grid``.

Final training mirrors scan ``train.py`` but consumes final replicate rows with
explicit final model/sampler seeds. It writes ``06_final_train`` provenance for
each final run before launching the canonical ``run.py`` entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path
from typing import Any, Sequence

import launch
from run_utils import (
    STAGE_FINAL_GRID,
    STAGE_FINAL_TRAIN,
    attempt_smoke,
    experiment_run_name,
    final_grid_attempt_dir,
    final_train_attempt_dir,
    final_train_run_dir,
    latest_attempt_id,
    log_prefix,
    read_json,
    seed_override_values,
    stage_dir,
    stage_job_name,
    study_name_from_manifest,
    write_json,
    write_latest,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

def _resolve_final_grid_attempt_id(results_root: Path, requested: str | None, *, smoke: bool) -> str:
    if requested is not None:
        is_smoke = attempt_smoke(stage_dir(results_root, STAGE_FINAL_GRID), requested)
        if not smoke and is_smoke is True:
            raise ValueError("full final training refuses a smoke final grid; pass --smoke")
        return requested
    final_grid_stage = stage_dir(results_root, STAGE_FINAL_GRID)
    attempt_id = latest_attempt_id(final_grid_stage, smoke=smoke)
    if attempt_id is None:
        mode = "smoke" if smoke else "production"
        raise FileNotFoundError(f"no {mode} final-grid attempts under {final_grid_stage}")
    return attempt_id


def load_final_grid_manifest(results_root: Path, final_grid_attempt_id: str) -> dict[str, Any]:
    """Read and validate the ``05_final_grid`` manifest."""

    manifest_path = final_grid_attempt_dir(results_root, final_grid_attempt_id) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"final-grid attempt has no manifest.json: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("stage") != STAGE_FINAL_GRID:
        raise ValueError(f"manifest {manifest_path} is not a {STAGE_FINAL_GRID} manifest")
    return manifest


def load_final_jobs(results_root: Path, final_grid_attempt_id: str) -> list[dict[str, Any]]:
    """Read final jobs in CSV order, enriching from per-job JSON records."""

    grid_dir = final_grid_attempt_dir(results_root, final_grid_attempt_id)
    final_jobs_path = grid_dir / "final_jobs.csv"
    if not final_jobs_path.is_file():
        raise FileNotFoundError(f"final-grid attempt has no final_jobs.csv: {final_jobs_path}")
    with final_jobs_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = []
    for row in rows:
        job_path = grid_dir / "jobs" / f"{row['final_run_id']}.json"
        if job_path.is_file():
            jobs.append(read_json(job_path))
        else:
            jobs.append(dict(row))
    return jobs


def _selected_jobs(jobs: Sequence[dict[str, Any]], *, smoke: bool) -> list[dict[str, Any]]:
    if not smoke:
        return [dict(job) for job in jobs]
    return [dict(job) for job in list(jobs)[: launch.SMOKE_JOB_LIMIT]]


def _attempt_id(args: argparse.Namespace, *, final_grid_attempt_id: str) -> str:
    if args.attempt_id:
        return launch.smoke_attempt_id(args.attempt_id) if args.smoke else args.attempt_id
    return launch.smoke_attempt_id(final_grid_attempt_id) if args.smoke else final_grid_attempt_id


def final_scalar_axes(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return non-seed axes recorded in a final-grid manifest."""

    return tuple(str(axis) for axis in (*manifest.get("major_axes", []), *manifest.get("minor_axes", [])))


def final_axis_override_paths(manifest: dict[str, Any], axes: Sequence[str]) -> dict[str, str]:
    """Return axis -> config override path from a final-grid manifest."""

    configured = manifest.get("axis_overrides")
    if not isinstance(configured, dict):
        raise ValueError("final-grid manifest axis_overrides must be a mapping")
    missing = [axis for axis in axes if axis not in configured]
    if missing:
        raise ValueError(f"final-grid manifest axis_overrides is missing axes: {', '.join(missing)}")
    return {axis: str(configured[axis]) for axis in axes}


def _job_choices(job: dict[str, Any]) -> dict[str, Any]:
    """Return scalar final-job choices, falling back to legacy top-level fields."""

    choices = job.get("choices")
    if isinstance(choices, dict):
        return dict(choices)
    merged: dict[str, Any] = {}
    for block in (job.get("major_choices"), job.get("minor_choices")):
        if isinstance(block, dict):
            merged.update(block)
    if merged:
        return merged
    return dict(job)


def axis_value_overrides_for_job(
    job: dict[str, Any],
    *,
    scalar_axes: Sequence[str],
    override_paths: dict[str, str],
) -> list[str]:
    """Return config overrides for all scalar non-seed final-job choices."""

    choices = _job_choices(job)
    overrides = []
    for axis in scalar_axes:
        if axis not in choices:
            raise ValueError(f"final job {job.get('final_run_id', '<unknown>')!r} is missing axis {axis!r}")
        overrides.append(f"{override_paths[axis]}={choices[axis]}")
    return overrides


def final_train_overrides(
    job: dict[str, Any],
    *,
    study: str,
    final_run_id: str,
    attempt_id: str,
    results_root: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, str],
) -> list[str]:
    """Return OmegaConf overrides for one final training run."""

    stage_seed_overrides = job.get("stage_seed_overrides", {})
    seed_overrides = (
        stage_seed_overrides.get("final_train")
        if isinstance(stage_seed_overrides, dict)
        else None
    )
    if seed_overrides is None:
        seed_overrides = seed_override_values(None, "final_train", job)
    return [
        *axis_value_overrides_for_job(job, scalar_axes=scalar_axes, override_paths=override_paths),
        *(f"{path}={value}" for path, value in seed_overrides.items()),
        f"run.root={stage_dir(results_root, STAGE_FINAL_TRAIN)}",
        "run.layout=flat",
        f"run.run_id={final_run_id}/{attempt_id}",
        f"study.name={study}",
        "study.stage=06_final_train",
        f"study.attempt_id={attempt_id}",
        f"study.config_id={job['source_champion_id']}",
        f"experiment.name={study}",
        f"experiment.run_name={experiment_run_name(study, 'final_train')}",
    ]


def _command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    return [python, "-u", "run.py", "--config", str(config), *overrides]


def _command_for_job(
    job: dict[str, Any],
    *,
    config: str | Path,
    study: str,
    attempt_id: str,
    results_root: Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, str],
) -> list[str]:
    final_run_id = str(job["final_run_id"])
    command = _command_for(
        config,
        final_train_overrides(
            job,
            study=study,
            final_run_id=final_run_id,
            attempt_id=attempt_id,
            results_root=results_root,
            scalar_axes=scalar_axes,
            override_paths=override_paths,
        ),
    )
    return command


def _checkpoint_selection_record(attempt_dir: Path) -> dict[str, Any]:
    checkpoint_dir = attempt_dir / "checkpoints"
    return {
        "selection_policy": "latest_checkpoint_pointer",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_pointer": str(checkpoint_dir / "latest.json"),
        "resolved_checkpoint_dir": None,
    }


def write_final_train_provenance(
    jobs: Sequence[dict[str, Any]],
    *,
    results_root: Path,
    final_grid_attempt_id: str,
    attempt_id: str,
    commands: Sequence[Sequence[str]],
    smoke: bool = False,
) -> None:
    """Write per-final-run source pointers before launch."""

    grid_dir = final_grid_attempt_dir(results_root, final_grid_attempt_id)
    for job, command in zip(jobs, commands, strict=True):
        final_run_id = str(job["final_run_id"])
        attempt_dir = final_train_attempt_dir(results_root, final_run_id, attempt_id)
        write_json(
            attempt_dir / "source_final_grid_attempt.json",
            {
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_grid_attempt_dir": str(grid_dir),
                "final_jobs_path": str(grid_dir / "final_jobs.csv"),
            },
        )
        write_json(attempt_dir / "source_final_job.json", job)
        write_json(attempt_dir / "source_champion.json", job.get("source_champion", {}))
        write_json(attempt_dir / "selected_checkpoint.json", _checkpoint_selection_record(attempt_dir))
        (attempt_dir / "command.txt").write_text(shlex.join([str(part) for part in command]) + "\n")
        write_latest(final_train_run_dir(results_root, final_run_id), attempt_id, smoke=smoke)


def write_final_train_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    results_root: Path,
    final_grid_attempt_id: str,
    attempt_id: str,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write final-train submission records."""

    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        final_run_id = str(job["final_run_id"])
        attempt_dir = final_train_attempt_dir(results_root, final_run_id, attempt_id)
        write_json(
            attempt_dir / "submission.json",
            {
                "final_run_id": final_run_id,
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_train_attempt_id": attempt_id,
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": (attempt_dir / "command.txt").read_text().strip(),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-train launch arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-grid-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--config", default=None, help="Train config path (defaults to final-grid manifest).")
    launch.add_launch_arguments(
        parser,
        smoke_help=(
            "Launch two short final-train smoke jobs from a smoke final grid, "
            "with smoke-marked attempt ids and test partitions."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch final training jobs."""

    args = parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    results_root = launch.repo_path(args.results_root, repo_root)
    final_grid_attempt_id = _resolve_final_grid_attempt_id(
        results_root,
        args.final_grid_attempt_id,
        smoke=args.smoke,
    )
    manifest = load_final_grid_manifest(results_root, final_grid_attempt_id)
    study = study_name_from_manifest(manifest)
    prefix = log_prefix(study)
    config = args.config or manifest.get("train_config")
    if not config:
        raise ValueError("final-grid manifest does not record train_config; pass --config")
    attempt_id = _attempt_id(args, final_grid_attempt_id=final_grid_attempt_id)
    jobs = _selected_jobs(load_final_jobs(results_root, final_grid_attempt_id), smoke=args.smoke)
    if not jobs:
        raise ValueError(f"final grid attempt {final_grid_attempt_id} has no jobs")
    scalar_axes = final_scalar_axes(manifest)
    override_paths = final_axis_override_paths(manifest, scalar_axes)
    configured_smoke_overrides = launch.load_smoke_overrides(
        "final_train",
        manifest=manifest,
        attempt_dir=final_grid_attempt_dir(results_root, final_grid_attempt_id),
    )
    commands = [
        launch.with_study_timezone(
            _command_for_job(
                job,
                config=config,
                study=study,
                attempt_id=attempt_id,
                results_root=results_root,
                scalar_axes=scalar_axes,
                override_paths=override_paths,
            )
        )
        for job in jobs
    ]
    if args.smoke:
        commands = [launch.with_overrides(command, configured_smoke_overrides) for command in commands]
    write_final_train_provenance(
        jobs,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        attempt_id=attempt_id,
        commands=commands,
        smoke=args.smoke,
    )

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

    row_status_paths = [
        final_train_attempt_dir(results_root, str(job["final_run_id"]), attempt_id) / "launcher_status.json"
        for job in jobs
    ]
    chunk_status_dir = stage_dir(results_root, STAGE_FINAL_TRAIN) / "chunk_status" / attempt_id
    if args.backend == "local":
        job_ids = launch.submit_local(
            submitted_commands,
            repo_root=repo_root,
            chunk_size=args.chunk_size,
            row_status_paths=row_status_paths,
            chunk_status_dir=chunk_status_dir,
        )
    else:
        log_attempt = attempt_id
        job_ids = launch.submit_submitit(
            submitted_commands,
            log_dir=stage_dir(results_root, STAGE_FINAL_TRAIN) / "slurm_logs" / log_attempt,
            job_name=stage_job_name(study, "final-train", smoke=args.smoke),
            slurm=launch.slurm_parameters(args, profile=args.profile, smoke=args.smoke),
            chunk_size=args.chunk_size,
            row_status_paths=row_status_paths,
            chunk_status_dir=chunk_status_dir,
        )

    write_final_train_submission_records(
        jobs,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        attempt_id=attempt_id,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    mode = "smoke final-train" if args.smoke else "final-train"
    print(f"{prefix} launched {len(job_ids)} {mode} jobs from 05_final_grid/{final_grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
