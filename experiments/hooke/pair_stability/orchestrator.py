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
DEFAULT_CPU_UV_ENVIRONMENT = ".venv"
DEFAULT_CUDA_UV_ENVIRONMENT = ".venv-gpu"
DEFAULT_CPU_EXTRA = "cpu"
DEFAULT_CUDA_EXTRA = "cu126"
DEFAULT_CPU_PARTITION = "seas_compute,kozinsky_lab,sapphire"
DEFAULT_CUDA_PARTITION = "seas_gpu,kozinsky_gpu"
DEFAULT_SMOKE_CPU_PARTITION = "test"
DEFAULT_SMOKE_CUDA_PARTITION = "gpu_test"
DEFAULT_TIMEOUT_MIN = 480
DEFAULT_MEM_GB = 32
DEFAULT_CPUS = 8
SMOKE_JOB_LIMIT = 2
SMOKE_TIMEOUT_MIN = 15
SMOKE_MEM_GB = 16
SMOKE_CPUS = 4
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


def _uses_python_executable(command_part: str) -> bool:
    """Return whether ``command_part`` names a Python executable."""

    return Path(command_part).name.startswith("python")


def _activated_python_command(command: Sequence[str]) -> list[str]:
    """Run planned Python commands through the currently active environment."""

    command = [str(part) for part in command]
    if command and _uses_python_executable(command[0]):
        return ["python", *command[1:]]
    return command


def _with_runtime_device(command: Sequence[str], *, device: str) -> list[str]:
    """Return ``command`` with a final runtime.device override."""

    return _with_overrides(command, {"runtime.device": device})


def _with_overrides(command: Sequence[str], overrides: dict[str, object]) -> list[str]:
    """Return ``command`` with final scalar OmegaConf overrides appended."""

    prefixes = tuple(f"{key}=" for key in overrides)
    command = [str(part) for part in command if not str(part).startswith(prefixes)]
    command.extend(f"{key}={value}" for key, value in overrides.items())
    return command


def _smoke_attempt_id(grid_attempt_id: str) -> str:
    """Return a train attempt id that clearly marks smoke execution."""

    return f"{grid_attempt_id}-smoke"


def _smoke_run_id(run_id: str, grid_attempt_id: str) -> str:
    """Return the flat-layout run id for a smoke train attempt."""

    return f"{run_id}/{_smoke_attempt_id(grid_attempt_id)}"


def _smoke_job(job: dict[str, Any], *, grid_attempt_id: str) -> dict[str, Any]:
    """Return a manifest job copy redirected to its smoke attempt directory."""

    job = dict(job)
    train_dir = Path(str(job["train_dir"]))
    job["train_attempt_dir"] = str(train_dir / _smoke_attempt_id(grid_attempt_id))
    return job


def _launch_jobs(jobs: Sequence[dict[str, Any]], *, grid_attempt_id: str, smoke: bool) -> list[dict[str, Any]]:
    """Return the jobs selected for this launch."""

    if not smoke:
        return [dict(job) for job in jobs]
    return [_smoke_job(job, grid_attempt_id=grid_attempt_id) for job in list(jobs)[:SMOKE_JOB_LIMIT]]


def _smoke_overrides(job: dict[str, Any], *, grid_attempt_id: str) -> dict[str, object]:
    """Return smoke-only command overrides for one selected job."""

    return {
        **SMOKE_OVERRIDES,
        "run.run_id": _smoke_run_id(str(job["run_id"]), grid_attempt_id),
        "study.attempt_id": _smoke_attempt_id(grid_attempt_id),
    }


def _execution_command(command: Sequence[str], job: dict[str, Any], *, grid_attempt_id: str, smoke: bool) -> list[str]:
    """Return the run command after launch-mode overrides are applied."""

    if not smoke:
        return [str(part) for part in command]
    return _with_overrides(command, _smoke_overrides(job, grid_attempt_id=grid_attempt_id))


def _environment_shell_command(
    command: Sequence[str],
    *,
    repo_root: Path,
    uv_environment: str,
    uv_extras: Sequence[str],
    device: str,
) -> list[str]:
    """Wrap a train command in the selected uv environment setup."""

    sync_command = ["uv", "sync"]
    for extra in uv_extras:
        sync_command.extend(["--extra", str(extra)])
    activate_path = Path(uv_environment) / "bin" / "activate"
    run_command = _activated_python_command(_with_runtime_device(command, device=device))
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            f"export UV_PROJECT_ENVIRONMENT={shlex.quote(str(uv_environment))}",
            shlex.join(sync_command),
            f"source {shlex.quote(str(activate_path))}",
            f"exec {shlex.join(run_command)}",
        ]
    )
    return ["bash", "-lc", script]


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
    """Submit already prepared commands through Submitit."""

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


def _environment_defaults(profile: str) -> tuple[str, list[str], str]:
    """Return default uv environment, uv extras, and runtime device."""

    if profile == "cuda":
        return DEFAULT_CUDA_UV_ENVIRONMENT, [DEFAULT_CUDA_EXTRA], "cuda"
    return DEFAULT_CPU_UV_ENVIRONMENT, [DEFAULT_CPU_EXTRA], "cpu"


def _slurm_parameters(args: argparse.Namespace, *, profile: str, smoke: bool = False) -> dict[str, Any]:
    """Return Submitit Slurm parameters for the selected profile."""

    partition = args.slurm_partition or (
        (DEFAULT_SMOKE_CUDA_PARTITION if profile == "cuda" else DEFAULT_SMOKE_CPU_PARTITION)
        if smoke
        else (DEFAULT_CUDA_PARTITION if profile == "cuda" else DEFAULT_CPU_PARTITION)
    )
    slurm = {
        "slurm_partition": partition,
        "timeout_min": args.slurm_timeout_min or (SMOKE_TIMEOUT_MIN if smoke else DEFAULT_TIMEOUT_MIN),
        "mem_gb": args.slurm_mem_gb or (SMOKE_MEM_GB if smoke else DEFAULT_MEM_GB),
        "cpus_per_task": args.slurm_cpus or (SMOKE_CPUS if smoke else DEFAULT_CPUS),
        "tasks_per_node": 1,
    }
    if profile == "cuda":
        slurm["gpus_per_node"] = args.slurm_gpus or 1
    return slurm


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse train-orchestrator command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--grid-attempt-id", default=None, help="Grid attempt to launch (defaults to latest).")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Submit two short smoke jobs with smoke-marked attempt ids, small "
            "samplers, two train steps, and 15-minute test partitions."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["local", "submitit"],
        required=True,
        help="Training launcher backend. Planning is handled separately by plan.py.",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cpu",
        action="store_const",
        const="cpu",
        dest="profile",
        default="cpu",
        help="Run with the CPU uv environment and runtime.device=cpu (default).",
    )
    device_group.add_argument(
        "--cuda",
        action="store_const",
        const="cuda",
        dest="profile",
        help="Run with the CUDA uv environment and runtime.device=cuda.",
    )
    parser.add_argument("--repo-root", default=None, help="Repo root for command working directory.")
    parser.add_argument(
        "--slurm-partition",
        default=None,
        help=(
            "Defaults to seas_compute,kozinsky_lab,sapphire for CPU and "
            "seas_gpu,kozinsky_gpu for CUDA."
        ),
    )
    parser.add_argument("--slurm-gpus", type=int, default=None, help="CUDA only; defaults to 1 with --cuda.")
    parser.add_argument("--slurm-timeout-min", type=int, default=None)
    parser.add_argument("--slurm-mem-gb", type=int, default=None)
    parser.add_argument("--slurm-cpus", type=int, default=None)
    parser.add_argument(
        "--uv-environment",
        default=None,
        help="UV project environment path to sync and activate (defaults by --cpu/--cuda).",
    )
    parser.add_argument(
        "--uv-extra",
        action="append",
        dest="uv_extras",
        default=None,
        help="UV extra passed to uv sync; repeat for multiple extras (defaults by --cpu/--cuda).",
    )
    # Backward-compatible aliases for the first CUDA-only version of this CLI.
    parser.add_argument("--gpu-uv-environment", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-extra", action="append", dest="gpu_extras", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch train jobs from an existing ``00_grid`` attempt."""

    args = parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    results_root = _repo_path(args.results_root, repo_root)
    grid_attempt_id = resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = load_grid_manifest(results_root, grid_attempt_id)
    jobs = _launch_jobs(list(manifest.get("jobs", [])), grid_attempt_id=grid_attempt_id, smoke=args.smoke)
    commands = [
        _execution_command(_command_for_job(job), job, grid_attempt_id=grid_attempt_id, smoke=args.smoke)
        for job in jobs
    ]
    uv_environment, uv_extras, runtime_device = _environment_defaults(args.profile)
    uv_environment = args.uv_environment or (
        args.gpu_uv_environment if args.profile == "cuda" else None
    ) or uv_environment
    uv_extras = args.uv_extras or (args.gpu_extras if args.profile == "cuda" else None) or uv_extras
    submitted_commands = [
        _environment_shell_command(
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
        job_ids = _submit_local(submitted_commands, repo_root=repo_root)
    else:
        job_ids = _submit_submitit(
            submitted_commands,
            log_dir=stage_dir(results_root, STAGE_TRAIN) / "slurm_logs" / (
                _smoke_attempt_id(grid_attempt_id) if args.smoke else grid_attempt_id
            ),
            job_name="hooke-pair-stability-smoke" if args.smoke else "hooke-pair-stability",
            slurm=_slurm_parameters(args, profile=args.profile, smoke=args.smoke),
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
