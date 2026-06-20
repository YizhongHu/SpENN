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

from run_utils import (
    STAGE_GRID,
    STAGE_TRAIN,
    attempt_ids,
    grid_attempt_dir,
    read_json,
    stage_dir,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"


# ---------------------------------------------------------------------------
# Grid-attempt loading
# ---------------------------------------------------------------------------
def _repo_path(path: str | Path, repo_root: Path) -> Path:
    """Return ``path`` anchored at ``repo_root`` when it is relative."""

    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def resolve_grid_attempt_id(results_root: str | Path, grid_attempt_id: str | None) -> str:
    """Return the requested grid attempt id, defaulting to ``00_grid/latest``."""

    if grid_attempt_id is not None:
        return grid_attempt_id
    grid_stage = stage_dir(results_root, STAGE_GRID)
    latest = grid_stage / "latest.json"
    if latest.is_file():
        attempt_id = read_json(latest).get("attempt_id")
        if attempt_id:
            return str(attempt_id)
    ids = attempt_ids(grid_stage)
    if not ids:
        raise FileNotFoundError(f"no grid attempts under {grid_stage}")
    return ids[-1]


def load_grid_manifest(results_root: str | Path, grid_attempt_id: str) -> dict[str, Any]:
    """Read the ``00_grid`` manifest for ``grid_attempt_id``."""

    manifest_path = grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"grid attempt has no manifest.json: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("stage") != STAGE_GRID:
        raise ValueError(f"manifest {manifest_path} is not a {STAGE_GRID} manifest")
    return manifest


def _command_for_job(job: dict[str, Any]) -> list[str]:
    """Return the exact command stored in a manifest job."""

    command = job.get("command")
    if isinstance(command, str) and command.strip():
        return shlex.split(command)
    if isinstance(command, list) and command:
        return [str(part) for part in command]
    raise ValueError(f"job {job.get('run_id', '<unknown>')!r} has no command")


# ---------------------------------------------------------------------------
# Submission backends
# ---------------------------------------------------------------------------
def _submit_local(commands: Sequence[Sequence[str]], *, repo_root: Path) -> list[str]:
    """Run train commands sequentially in-process (smoke / local backend)."""

    import subprocess

    job_ids = []
    for index, command in enumerate(commands):
        result = subprocess.run(list(command), cwd=str(repo_root), check=False)
        job_ids.append(f"local-{index}-rc{result.returncode}")
        if result.returncode != 0:
            raise RuntimeError(f"local job {index} failed: {shlex.join(command)}")
    return job_ids


def _submit_submitit(
    commands: Sequence[Sequence[str]],
    *,
    log_dir: Path,
    job_name: str,
    slurm: dict[str, Any],
) -> list[str]:
    """Submit commands through the Submitit launcher (no bespoke sbatch)."""

    try:
        import submitit  # lazy: optional 'submitit' extra
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "submitit backend requires the optional dependency; install with "
            "`uv sync --extra submitit`"
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(name=job_name, **slurm)
    jobs = [executor.submit(submitit.helpers.CommandFunction(list(command))) for command in commands]
    return [str(job.job_id) for job in jobs]


def _train_attempt_dir(job: dict[str, Any], *, manifest: dict[str, Any], repo_root: Path) -> Path:
    if job.get("train_attempt_dir"):
        return _repo_path(str(job["train_attempt_dir"]), repo_root)
    if job.get("train_dir"):
        return _repo_path(str(job["train_dir"]), repo_root) / str(manifest["attempt_id"])
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
) -> None:
    """Write train-stage provenance without mutating the ``00_grid`` manifest."""

    manifest_path = grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json"
    grid_dir = grid_attempt_dir(results_root, grid_attempt_id)
    for job, job_id in zip(jobs, job_ids, strict=True):
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
            },
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse train-orchestrator command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--grid-attempt-id", default=None, help="Grid attempt to launch (defaults to latest).")
    parser.add_argument(
        "--backend",
        choices=["local", "submitit"],
        required=True,
        help="Training launcher backend. Planning is handled separately by plan.py.",
    )
    parser.add_argument("--repo-root", default=None, help="Repo root for command working directory.")
    parser.add_argument("--slurm-partition", default="kozinsky_gpu")
    parser.add_argument("--slurm-gpus", type=int, default=1)
    parser.add_argument("--slurm-timeout-min", type=int, default=480)
    parser.add_argument("--slurm-mem-gb", type=int, default=32)
    parser.add_argument("--slurm-cpus", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch train jobs from an existing ``00_grid`` attempt."""

    args = parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    results_root = _repo_path(args.results_root, repo_root)
    grid_attempt_id = resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = load_grid_manifest(results_root, grid_attempt_id)
    jobs = list(manifest.get("jobs", []))
    commands = [_command_for_job(job) for job in jobs]

    if not jobs:
        print(f"[pair_stability] grid attempt {grid_attempt_id} has no jobs")
        return 0

    if args.backend == "local":
        job_ids = _submit_local(commands, repo_root=repo_root)
    else:
        slurm = {
            "slurm_partition": args.slurm_partition,
            "gpus_per_node": args.slurm_gpus,
            "timeout_min": args.slurm_timeout_min,
            "mem_gb": args.slurm_mem_gb,
            "cpus_per_task": args.slurm_cpus,
            "tasks_per_node": 1,
        }
        job_ids = _submit_submitit(
            commands,
            log_dir=stage_dir(results_root, STAGE_TRAIN) / "slurm_logs" / grid_attempt_id,
            job_name="hooke-pair-stability",
            slurm=slurm,
        )

    write_train_submission_records(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        repo_root=repo_root,
        backend=args.backend,
        job_ids=job_ids,
    )
    print(f"[pair_stability] launched {len(job_ids)} train jobs from 00_grid/{grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
