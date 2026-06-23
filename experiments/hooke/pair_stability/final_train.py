"""Launch final pair-stability training from ``05_final_grid``.

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
    attempt_ids,
    final_grid_attempt_dir,
    final_train_attempt_dir,
    final_train_run_dir,
    read_json,
    stage_dir,
    write_json,
    write_latest,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
DEFAULT_TRAIN_CONFIG = STUDY_DIR / "configs" / "pair_stability.yaml"

SMOKE_FINAL_TRAIN_OVERRIDES = {
    "training.max_steps": 2,
    "training.log_every_n_steps": 1,
    "sampler_params.n_walkers": 128,
    "sampler_params.burn_in": 10,
    "sampler_params.n_steps": 5,
    "checks.every_n_steps": 1,
    "checkpoint.every_n_steps": 1,
    "status.every_n_steps": 1,
}


def _is_smoke_attempt(attempt_id: str) -> bool:
    return attempt_id.endswith("-smoke")


def _resolve_final_grid_attempt_id(results_root: Path, requested: str | None, *, smoke: bool) -> str:
    if requested is not None:
        if not smoke and _is_smoke_attempt(requested):
            raise ValueError("full final training refuses a smoke final grid; pass --smoke")
        return requested
    final_grid_stage = stage_dir(results_root, STAGE_FINAL_GRID)
    attempts = attempt_ids(final_grid_stage)
    candidates = [attempt_id for attempt_id in attempts if _is_smoke_attempt(attempt_id) == smoke]
    if not candidates:
        mode = "smoke" if smoke else "production"
        raise FileNotFoundError(f"no {mode} final-grid attempts under {final_grid_stage}")
    return candidates[-1]


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


def _run_parameter_overrides(job: dict[str, Any]) -> list[str]:
    return [
        f"run_parameters.architecture={job['architecture']}",
        f"run_parameters.normalization={job['normalization']}",
        f"run_parameters.lr={job['lr']}",
        f"run_parameters.channels={int(job['channels'])}",
        f"run_parameters.seed={int(job['final_train_model_seed'])}",
    ]


def final_train_overrides(
    job: dict[str, Any],
    *,
    final_run_id: str,
    attempt_id: str,
    results_root: str | Path,
) -> list[str]:
    """Return OmegaConf overrides for one final training run."""

    model_seed = int(job["final_train_model_seed"])
    sampler_seed = int(job["final_train_sampler_seed"])
    return [
        *_run_parameter_overrides(job),
        f"runtime.seed={model_seed}",
        f"sampler.seed={sampler_seed}",
        f"run.root={stage_dir(results_root, STAGE_FINAL_TRAIN)}",
        "run.layout=flat",
        f"run.run_id={final_run_id}/{attempt_id}",
        "study.stage=06_final_train",
        f"study.attempt_id={attempt_id}",
        f"study.config_id={job['source_champion_id']}",
        "experiment.run_name=hooke_pair_stability_final_train",
    ]


def _command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    return [python, "-u", "run.py", "--config", str(config), *overrides]


def _command_for_job(job: dict[str, Any], *, config: str | Path, attempt_id: str, results_root: Path) -> list[str]:
    final_run_id = str(job["final_run_id"])
    command = _command_for(
        config,
        final_train_overrides(
            job,
            final_run_id=final_run_id,
            attempt_id=attempt_id,
            results_root=results_root,
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
        write_latest(final_train_run_dir(results_root, final_run_id), attempt_id)


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
    config = args.config or manifest.get("train_config") or str(DEFAULT_TRAIN_CONFIG)
    attempt_id = _attempt_id(args, final_grid_attempt_id=final_grid_attempt_id)
    jobs = _selected_jobs(load_final_jobs(results_root, final_grid_attempt_id), smoke=args.smoke)
    commands = [
        launch.with_study_timezone(
            _command_for_job(job, config=config, attempt_id=attempt_id, results_root=results_root)
        )
        for job in jobs
    ]
    if args.smoke:
        commands = [launch.with_overrides(command, SMOKE_FINAL_TRAIN_OVERRIDES) for command in commands]
    write_final_train_provenance(
        jobs,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        attempt_id=attempt_id,
        commands=commands,
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

    if not jobs:
        print(f"[pair_stability] final grid attempt {final_grid_attempt_id} has no jobs")
        return 0

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
            job_name="hooke-pair-stability-final-train-smoke" if args.smoke else "hooke-pair-stability-final-train",
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
    print(f"[pair_stability] launched {len(job_ids)} {mode} jobs from 05_final_grid/{final_grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
