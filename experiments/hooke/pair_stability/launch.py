"""Shared launch plumbing for Hooke pair-stability stage scripts.

This module owns the execution mechanics shared by ``train.py`` and
``validate.py``: CPU/CUDA profile defaults, uv environment activation, local
execution, and Submitit submission. Stage scripts own stage-specific command
construction and provenance.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence, TypeVar

from run_utils import STAGE_GRID, attempt_ids, grid_attempt_dir, read_json, stage_dir

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
DEFAULT_ARRAY_PARALLELISM = 16
DEFAULT_CHUNK_SIZE = 1
SMOKE_JOB_LIMIT = 2
SMOKE_TIMEOUT_MIN = 15
SMOKE_MEM_GB = 16
SMOKE_CPUS = 4
SMOKE_ARRAY_PARALLELISM = 2

T = TypeVar("T")


def repo_path(path: str | Path, repo_root: Path) -> Path:
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


def command_for_job(job: dict[str, Any]) -> list[str]:
    """Return the exact command stored in a manifest job."""

    command = job.get("command")
    if isinstance(command, str) and command.strip():
        return shlex.split(command)
    if isinstance(command, list) and command:
        return [str(part) for part in command]
    raise ValueError(f"job {job.get('run_id', '<unknown>')!r} has no command")


def with_overrides(command: Sequence[str], overrides: dict[str, object]) -> list[str]:
    """Return ``command`` with final scalar OmegaConf overrides appended."""

    prefixes = tuple(f"{key}=" for key in overrides)
    command = [str(part) for part in command if not str(part).startswith(prefixes)]
    command.extend(f"{key}={value}" for key, value in overrides.items())
    return command


def with_runtime_device(command: Sequence[str], *, device: str) -> list[str]:
    """Return ``command`` with a final runtime.device override."""

    return with_overrides(command, {"runtime.device": device})


def smoke_attempt_id(base_attempt_id: str) -> str:
    """Return an attempt id that clearly marks smoke execution."""

    return f"{base_attempt_id}-smoke"


def environment_defaults(profile: str) -> tuple[str, list[str], str]:
    """Return default uv environment, uv extras, and runtime device."""

    if profile == "cuda":
        return DEFAULT_CUDA_UV_ENVIRONMENT, [DEFAULT_CUDA_EXTRA], "cuda"
    return DEFAULT_CPU_UV_ENVIRONMENT, [DEFAULT_CPU_EXTRA], "cpu"


def resolve_uv_settings(args: argparse.Namespace) -> tuple[str, list[str], str]:
    """Return uv environment, uv extras, and runtime device for parsed args."""

    uv_environment, uv_extras, runtime_device = environment_defaults(args.profile)
    uv_environment = args.uv_environment or (
        args.gpu_uv_environment if args.profile == "cuda" else None
    ) or uv_environment
    uv_extras = args.uv_extras or (args.gpu_extras if args.profile == "cuda" else None) or uv_extras
    return uv_environment, list(uv_extras), runtime_device


def _uses_python_executable(command_part: str) -> bool:
    """Return whether ``command_part`` names a Python executable."""

    return Path(command_part).name.startswith("python")


def _activated_python_command(command: Sequence[str]) -> list[str]:
    """Run planned Python commands through the currently active environment."""

    command = [str(part) for part in command]
    if command and _uses_python_executable(command[0]):
        return ["python", *command[1:]]
    return command


def environment_shell_command(
    command: Sequence[str],
    *,
    repo_root: Path,
    uv_environment: str,
    uv_extras: Sequence[str],
    device: str,
) -> list[str]:
    """Wrap a run command in the selected uv environment setup."""

    sync_command = ["uv", "sync"]
    for extra in uv_extras:
        sync_command.extend(["--extra", str(extra)])
    activate_path = Path(uv_environment) / "bin" / "activate"
    run_command = _activated_python_command(with_runtime_device(command, device=device))
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


def positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def balanced_chunks(items: Sequence[T], *, chunk_size: int) -> list[list[T]]:
    """Split ``items`` into evenly sized chunks no larger than ``chunk_size``."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    items = list(items)
    if not items:
        return []
    n_chunks = (len(items) + chunk_size - 1) // chunk_size
    base_size, extra = divmod(len(items), n_chunks)
    chunks: list[list[T]] = []
    start = 0
    for index in range(n_chunks):
        size = base_size + (1 if index < extra else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def run_command_chunk(commands: Sequence[Sequence[str]], *, cwd: str | Path | None = None) -> None:
    """Run a chunk of prepared commands, reporting failures after the chunk."""

    failures = []
    for index, command in enumerate(commands):
        command = [str(part) for part in command]
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
        if result.returncode != 0:
            failures.append((index, result.returncode, shlex.join(command)))
    if failures:
        lines = [
            f"{len(failures)} of {len(commands)} command(s) failed in this chunk:",
            *[
                f"  chunk item {index}: return code {returncode}; {command}"
                for index, returncode, command in failures
            ],
        ]
        raise RuntimeError("\n".join(lines))


def _expanded_chunk_job_ids(chunk_job_ids: Sequence[str], chunks: Sequence[Sequence[Any]]) -> list[str]:
    """Return one launcher job id per original command."""

    expanded: list[str] = []
    for job_id, chunk in zip(chunk_job_ids, chunks, strict=True):
        expanded.extend([str(job_id)] * len(chunk))
    return expanded


def submit_local(
    commands: Sequence[Sequence[str]],
    *,
    repo_root: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    """Run commands sequentially in-process, grouped into balanced chunks."""

    job_ids = []
    chunks = balanced_chunks(commands, chunk_size=chunk_size)
    for index, chunk in enumerate(chunks):
        try:
            run_command_chunk(chunk, cwd=repo_root)
        except RuntimeError as exc:
            raise RuntimeError(f"local chunk {index} failed") from exc
        job_ids.extend([f"local-chunk-{index}-rc0"] * len(chunk))
    return job_ids


def submit_submitit(
    commands: Sequence[Sequence[str]],
    *,
    log_dir: Path,
    job_name: str,
    slurm: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    """Submit prepared commands through Submitit as balanced array chunks."""

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
    command_chunks = [
        [[str(part) for part in command] for command in chunk]
        for chunk in balanced_chunks(commands, chunk_size=chunk_size)
    ]
    jobs = executor.map_array(run_command_chunk, command_chunks)
    chunk_job_ids = [str(job.job_id) for job in jobs]
    return _expanded_chunk_job_ids(chunk_job_ids, command_chunks)


def slurm_parameters(args: argparse.Namespace, *, profile: str, smoke: bool = False) -> dict[str, Any]:
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
        "slurm_array_parallelism": args.slurm_array_parallelism
        or (SMOKE_ARRAY_PARALLELISM if smoke else DEFAULT_ARRAY_PARALLELISM),
    }
    if profile == "cuda":
        slurm["gpus_per_node"] = args.slurm_gpus or 1
    return slurm


def add_launch_arguments(parser: argparse.ArgumentParser, *, smoke_help: str) -> None:
    """Add shared local/Submitit and CPU/CUDA launch arguments."""

    parser.add_argument("--smoke", action="store_true", help=smoke_help)
    parser.add_argument("--backend", choices=["local", "submitit"], required=True)
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
        "--chunk-size",
        type=positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Maximum desired run commands per local/Submitit chunk. Chunks are "
            f"balanced evenly; default {DEFAULT_CHUNK_SIZE} keeps one command per array task."
        ),
    )
    parser.add_argument(
        "--slurm-array-parallelism",
        type=int,
        default=None,
        help=(
            "Maximum number of Submitit array tasks allowed to run at once "
            f"(defaults to {DEFAULT_ARRAY_PARALLELISM}, or {SMOKE_ARRAY_PARALLELISM} with --smoke)."
        ),
    )
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
    # Backward-compatible aliases for the first CUDA-only train launcher.
    parser.add_argument("--gpu-uv-environment", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-extra", action="append", dest="gpu_extras", default=None, help=argparse.SUPPRESS)
