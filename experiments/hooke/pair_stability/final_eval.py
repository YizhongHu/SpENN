"""Launch final pair-stability evaluation from final train attempts.

Final evaluation consumes ``05_final_grid`` and completed ``06_final_train``
attempts. It records the exact final-train checkpoint directory evaluated and
launches the named ``final_eval`` suite from ``pair_validation.yaml``.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any, Sequence

import launch
from final_train import load_final_grid_manifest, load_final_jobs
from run_utils import (
    STAGE_FINAL_EVAL,
    STAGE_FINAL_GRID,
    attempt_ids,
    final_eval_attempt_dir,
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
DEFAULT_EVAL_CONFIG = STUDY_DIR / "configs" / "pair_validation.yaml"

SMOKE_FINAL_EVAL_OVERRIDES = {
    "evaluation_tasks.final_cusp.generator.n_points": 4,
    "evaluation_tasks.final_cusp.generator.n_directions": 2,
    "evaluation_tasks.final_cusp.generator.center_of_mass_radii": [0.0],
    "evaluation_tasks.final_tail.generator.n_points": 4,
    "evaluation_tasks.final_tail.generator.n_directions": 2,
    "evaluation_tasks.final_stratified_geometry.generator.n_samples": 16,
    "evaluation_tasks.final_hooke_orbital.generator.n_samples": 16,
    "evaluation_tasks.final_mcmc_energy.generator.max_samples": 32,
    "final_eval_sampler_params.n_walkers": 128,
    "final_eval_sampler_params.burn_in": 10,
    "final_eval_sampler_params.n_steps": 5,
    "final_eval_sampler_params.max_samples": 32,
    "evaluation_tasks.final_full_model_antisymmetry.generator.base_generator.n_samples": 8,
    "evaluation_tasks.final_spatial_exchange_symmetry.generator.base_generator.n_samples": 8,
    "evaluation_tasks.final_rotation_consistency.generator.base_generator.n_samples": 8,
    "evaluation_tasks.final_rotation_consistency.generator.n_rotations": 1,
    "evaluation_tasks.final_trace_equivariance.generator.base_generator.n_samples": 4,
    "evaluation_tasks.final_feature_trace_stability.generator.n_samples": 8,
    "evaluation_tasks.final_readout_trace_stability.generator.n_samples": 8,
}


def _is_smoke_attempt(attempt_id: str) -> bool:
    return attempt_id.endswith("-smoke")


def _resolve_final_grid_attempt_id(results_root: Path, requested: str | None, *, smoke: bool) -> str:
    if requested is not None:
        if not smoke and _is_smoke_attempt(requested):
            raise ValueError("full final eval refuses a smoke final grid; pass --smoke")
        return requested
    final_grid_stage = stage_dir(results_root, STAGE_FINAL_GRID)
    attempts = attempt_ids(final_grid_stage)
    candidates = [attempt_id for attempt_id in attempts if _is_smoke_attempt(attempt_id) == smoke]
    if not candidates:
        mode = "smoke" if smoke else "production"
        raise FileNotFoundError(f"no {mode} final-grid attempts under {final_grid_stage}")
    return candidates[-1]


def _selected_jobs(jobs: Sequence[dict[str, Any]], *, smoke: bool) -> list[dict[str, Any]]:
    if not smoke:
        return [dict(job) for job in jobs]
    return [dict(job) for job in list(jobs)[: launch.SMOKE_JOB_LIMIT]]


def _attempt_id(args: argparse.Namespace, *, final_grid_attempt_id: str) -> str:
    if args.attempt_id:
        return launch.smoke_attempt_id(args.attempt_id) if args.smoke else args.attempt_id
    return launch.smoke_attempt_id(final_grid_attempt_id) if args.smoke else final_grid_attempt_id


def latest_final_train_attempt_id(
    results_root: str | Path,
    final_run_id: str,
    *,
    smoke: bool,
) -> str | None:
    """Return the latest eligible final-train attempt id for ``final_run_id``."""

    ids = attempt_ids(final_train_run_dir(results_root, final_run_id))
    candidates = [attempt_id for attempt_id in ids if _is_smoke_attempt(attempt_id) == smoke]
    return candidates[-1] if candidates else None


def _final_train_attempt_id_for_job(
    *,
    args: argparse.Namespace,
    results_root: Path,
    final_run_id: str,
) -> str | None:
    if args.final_train_attempt_id is not None:
        is_smoke = _is_smoke_attempt(args.final_train_attempt_id)
        if not args.smoke and is_smoke:
            raise ValueError("full final eval refuses a smoke final-train attempt; pass --smoke")
        if args.smoke and is_smoke is False and not args.allow_production_final_train:
            raise ValueError(
                "smoke final eval refuses a production final-train attempt unless "
                "--allow-production-final-train is passed"
            )
        return args.final_train_attempt_id
    return _latest_ready_final_train_attempt_id(results_root, final_run_id, smoke=args.smoke)


def _resolved_checkpoint(train_attempt: Path) -> dict[str, Any] | None:
    selection_path = train_attempt / "selected_checkpoint.json"
    if not selection_path.is_file():
        return None
    selection = read_json(selection_path)
    pointer = Path(str(selection.get("checkpoint_pointer", "")))
    if not pointer.is_file():
        return None
    pointer_data = read_json(pointer)
    checkpoint_name = pointer_data.get("checkpoint_dir")
    if not checkpoint_name:
        return None
    checkpoint_dir = pointer.parent / str(checkpoint_name)
    if not checkpoint_dir.is_dir():
        return None
    if not (checkpoint_dir / "COMPLETE").is_file() or not (checkpoint_dir / "manifest.json").is_file():
        return None
    return {
        "selection_path": str(selection_path),
        "selection_policy": selection.get("selection_policy", ""),
        "checkpoint_pointer": str(pointer),
        "checkpoint_pointer_data": pointer_data,
        "resolved_checkpoint_dir": str(checkpoint_dir),
    }


def _latest_ready_final_train_attempt_id(
    results_root: str | Path,
    final_run_id: str,
    *,
    smoke: bool,
) -> str | None:
    """Return the newest final-train attempt with a completed selected checkpoint."""

    ids = attempt_ids(final_train_run_dir(results_root, final_run_id))
    candidates = [attempt_id for attempt_id in ids if _is_smoke_attempt(attempt_id) == smoke]
    for attempt_id in reversed(candidates):
        train_attempt = final_train_attempt_dir(results_root, final_run_id, attempt_id)
        if _resolved_checkpoint(train_attempt) is not None:
            return attempt_id
    return None


def _run_parameter_overrides(job: dict[str, Any]) -> list[str]:
    return [
        f"run_parameters.architecture={job['architecture']}",
        f"run_parameters.normalization={job['normalization']}",
        f"run_parameters.lr={job['lr']}",
        f"run_parameters.channels={int(job['channels'])}",
        f"run_parameters.seed={int(job['final_eval_seed'])}",
    ]


def final_eval_overrides(
    job: dict[str, Any],
    *,
    final_run_id: str,
    attempt_id: str,
    results_root: str | Path,
    checkpoint_dir: str | Path,
) -> list[str]:
    """Return OmegaConf overrides for one final evaluation run."""

    final_eval_seed = int(job["final_eval_seed"])
    return [
        *_run_parameter_overrides(job),
        f"runtime.seed={final_eval_seed}",
        f"evaluation.seed={final_eval_seed}",
        "evaluation.suite=final_eval",
        f"load.path={checkpoint_dir}",
        f"run.root={stage_dir(results_root, STAGE_FINAL_EVAL)}",
        "run.layout=flat",
        f"run.run_id={final_run_id}/{attempt_id}",
        "study.stage=07_final_eval",
        f"study.attempt_id={attempt_id}",
        f"study.config_id={job['source_champion_id']}",
        "experiment.run_name=hooke_pair_stability_final_eval",
    ]


def _command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    return [python, "-u", "run.py", "--config", str(config), *overrides]


def plan_final_eval_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    results_root: Path,
    final_grid_attempt_id: str,
    eval_config: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build final-eval launch records and write source provenance."""

    selected = _selected_jobs(jobs, smoke=args.smoke)
    final_eval_attempt_id = _attempt_id(args, final_grid_attempt_id=final_grid_attempt_id)
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    grid_dir = final_grid_attempt_dir(results_root, final_grid_attempt_id)
    for job in selected:
        final_run_id = str(job["final_run_id"])
        train_attempt_id = _final_train_attempt_id_for_job(
            args=args,
            results_root=results_root,
            final_run_id=final_run_id,
        )
        if train_attempt_id is None:
            skipped.append({"final_run_id": final_run_id, "reason": "no eligible final-train attempt"})
            continue
        train_attempt = final_train_attempt_dir(results_root, final_run_id, train_attempt_id)
        checkpoint = _resolved_checkpoint(train_attempt)
        if checkpoint is None:
            skipped.append({"final_run_id": final_run_id, "reason": f"missing selected checkpoint in {train_attempt}"})
            continue

        final_eval_attempt = final_eval_attempt_dir(results_root, final_run_id, final_eval_attempt_id)
        write_json(
            final_eval_attempt / "source_final_grid_attempt.json",
            {
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_grid_attempt_dir": str(grid_dir),
                "final_jobs_path": str(grid_dir / "final_jobs.csv"),
            },
        )
        write_json(
            final_eval_attempt / "source_final_train_attempt.json",
            {
                "final_run_id": final_run_id,
                "final_train_attempt_id": train_attempt_id,
                "final_train_attempt_dir": str(train_attempt),
                "checkpoint": checkpoint,
            },
        )
        write_json(final_eval_attempt / "source_final_job.json", job)
        write_json(final_eval_attempt / "source_champion.json", job.get("source_champion", {}))
        write_json(final_eval_attempt / "evaluated_checkpoint.json", checkpoint)

        command = _command_for(
            eval_config,
            final_eval_overrides(
                job,
                final_run_id=final_run_id,
                attempt_id=final_eval_attempt_id,
                results_root=results_root,
                checkpoint_dir=checkpoint["resolved_checkpoint_dir"],
            ),
        )
        command = launch.with_study_timezone(command)
        if args.smoke:
            command = launch.with_overrides(command, SMOKE_FINAL_EVAL_OVERRIDES)
        (final_eval_attempt / "command.txt").write_text(shlex.join(command) + "\n")
        write_latest(final_eval_attempt.parent, final_eval_attempt_id)
        planned.append(
            {
                "final_run_id": final_run_id,
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_train_attempt_id": train_attempt_id,
                "final_eval_attempt_id": final_eval_attempt_id,
                "final_eval_attempt_dir": str(final_eval_attempt),
                "checkpoint": checkpoint,
                "command": shlex.join(command),
                "command_parts": command,
            }
        )
    return planned, skipped


def write_final_eval_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write final-eval submission provenance."""

    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        final_eval_attempt = Path(str(job["final_eval_attempt_dir"]))
        write_json(
            final_eval_attempt / "submission.json",
            {
                "final_run_id": str(job["final_run_id"]),
                "final_grid_attempt_id": str(job["final_grid_attempt_id"]),
                "final_train_attempt_id": str(job["final_train_attempt_id"]),
                "final_eval_attempt_id": str(job["final_eval_attempt_id"]),
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": str(job["command"]),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-eval launch arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-grid-attempt-id", default=None)
    parser.add_argument("--final-train-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--config", default=None, help="Eval config path (defaults to final-grid manifest).")
    parser.add_argument(
        "--only-ready",
        action="store_true",
        help="Only launch rows with eligible completed final-train checkpoints; this is the default readiness policy.",
    )
    parser.add_argument(
        "--allow-production-final-train",
        action="store_true",
        help="With --smoke, allow explicitly requested production final-train attempts.",
    )
    launch.add_launch_arguments(
        parser,
        smoke_help=(
            "Launch final-eval smoke jobs from smoke final-train attempts with "
            "small report-grade task sizes and smoke-marked attempt ids."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch final evaluation jobs."""

    args = parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    results_root = launch.repo_path(args.results_root, repo_root)
    final_grid_attempt_id = _resolve_final_grid_attempt_id(
        results_root,
        args.final_grid_attempt_id,
        smoke=args.smoke,
    )
    manifest = load_final_grid_manifest(results_root, final_grid_attempt_id)
    eval_config = args.config or manifest.get("eval_config") or str(DEFAULT_EVAL_CONFIG)
    jobs, skipped = plan_final_eval_jobs(
        load_final_jobs(results_root, final_grid_attempt_id),
        args=args,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        eval_config=eval_config,
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
        print(f"[pair_stability] skipped {len(skipped)} final-eval jobs without ready final-train checkpoints")
    if not jobs:
        print(f"[pair_stability] no final-eval jobs ready for 05_final_grid/{final_grid_attempt_id}")
        return 1 if manifest.get("n_jobs") else 0

    attempt_id = str(jobs[0]["final_eval_attempt_id"])
    row_status_paths = [Path(str(job["final_eval_attempt_dir"])) / "launcher_status.json" for job in jobs]
    chunk_status_dir = stage_dir(results_root, STAGE_FINAL_EVAL) / "chunk_status" / attempt_id
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
        job_ids = launch.submit_submitit(
            submitted_commands,
            log_dir=stage_dir(results_root, STAGE_FINAL_EVAL) / "slurm_logs" / attempt_id,
            job_name="hooke-pair-stability-final-eval-smoke" if args.smoke else "hooke-pair-stability-final-eval",
            slurm=launch.slurm_parameters(args, profile=args.profile, smoke=args.smoke),
            chunk_size=args.chunk_size,
            allow_partial_failures=True,
            row_status_paths=row_status_paths,
            chunk_status_dir=chunk_status_dir,
        )

    write_final_eval_submission_records(
        jobs,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    mode = "smoke final-eval" if args.smoke else "final-eval"
    print(f"[pair_stability] launched {len(job_ids)} {mode} jobs from 05_final_grid/{final_grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
