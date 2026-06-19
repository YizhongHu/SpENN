"""Pair-stability grid orchestrator (PR8.8).

Compiles the architecture x normalization x lr x channels x seed grid into
scalar override lists for the canonical run entrypoint (``run.py``) and writes a
durable ``00_grid`` attempt (manifest + commands) describing the planned jobs.

The orchestrator does not hand-write per-variant YAML and does not emit bespoke
``sbatch`` scripts. Submission reuses the canonical ``run.py`` command path; the
optional ``submitit`` backend hands those commands to the Submitit launcher,
which owns Slurm script generation.

The orchestrator is the source of truth for the study timezone (``--timezone``,
default America/New_York): it stamps attempt ids and the manifest ``created_at``
with it, and overrides ``run.timezone`` in the compiled commands when the run
config declares a different zone.

Stage layout (under ``results_root``)::

    00_grid/{attempt_id}/{manifest.json, commands.sh, grid.yaml,
                          pair_stability.yaml, jobs/{run_id}.json}
    01_train/{run_id}/{attempt_id}/...
    02_validation/{run_id}/{attempt_id}/...
    03_collect/{attempt_id}/...
    04_select/{attempt_id}/...

``run.dir`` for a train/validation attempt is realized by the flat run layout:
``run.root = results_root/<stage>`` and ``run.run_id = <run_id>/<attempt_id>``.
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from run_utils import (
    DEFAULT_STUDY_TIMEZONE,
    GRID_AXES,
    STAGE_TRAIN,
    STAGE_VALIDATION,
    STAGE_GRID,
    grid_attempt_dir,
    new_attempt_id,
    resolve_timezone,
    run_id_for,
    stage_dir,
    train_attempt_dir,
    train_run_dir,
    validation_attempt_dir,
    validation_run_dir,
    write_json,
    write_latest,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_GRID = STUDY_DIR / "configs" / "grid.yaml"


# ---------------------------------------------------------------------------
# Grid expansion and validation
# ---------------------------------------------------------------------------
def expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand grid axes into a deterministic list of grid points."""

    axes = [list(grid[axis]) for axis in GRID_AXES]
    points = []
    for combination in itertools.product(*axes):
        point = dict(zip(GRID_AXES, combination, strict=True))
        point["lr"] = float(point["lr"])
        point["channels"] = int(point["channels"])
        point["seed"] = int(point["seed"])
        points.append(point)
    return points


def architecture_tags(config: Any) -> dict[str, list[str]]:
    """Return ``{architecture_name: tags}`` from a loaded pair_stability config."""

    arch = config.choices.architecture
    return {str(name): [str(tag) for tag in (arch[name].get("tags") or [])] for name in arch.keys()}


def normalization_names(config: Any) -> set[str]:
    """Return the set of normalization choice names from a loaded config."""

    return {str(name) for name in config.choices.normalization.keys()}


def validate_grid(points: Sequence[dict[str, Any]], config: Any) -> None:
    """Fail loudly if any grid point references an unknown or excluded choice."""

    known_arch = set(architecture_tags(config))
    known_norm = normalization_names(config)
    for point in points:
        architecture = str(point["architecture"])
        normalization = str(point["normalization"])
        if architecture.endswith("_no_envelope"):
            raise ValueError(
                f"no-envelope architecture {architecture!r} is excluded from the main scan"
            )
        if architecture not in known_arch:
            raise ValueError(f"grid architecture {architecture!r} is not in the choice library")
        if normalization not in known_norm:
            raise ValueError(f"grid normalization {normalization!r} is not in the choice library")


# ---------------------------------------------------------------------------
# Overrides and commands
# ---------------------------------------------------------------------------
def _run_parameter_overrides(point: dict[str, Any]) -> list[str]:
    return [
        f"run_parameters.architecture={point['architecture']}",
        f"run_parameters.normalization={point['normalization']}",
        f"run_parameters.lr={point['lr']}",
        f"run_parameters.channels={int(point['channels'])}",
        f"run_parameters.seed={int(point['seed'])}",
    ]


def train_overrides(
    point: dict[str, Any],
    *,
    run_id: str,
    attempt_id: str,
    results_root: str | Path,
    timezone: str | None = None,
) -> list[str]:
    """Return scalar Hydra-style overrides for one train job."""

    overrides = [
        *_run_parameter_overrides(point),
        f"run.root={stage_dir(results_root, STAGE_TRAIN)}",
        "run.layout=flat",
        f"run.run_id={run_id}/{attempt_id}",
        f"study.attempt_id={attempt_id}",
    ]
    if timezone is not None:
        overrides.append(f"run.timezone={timezone}")
    return overrides


def validation_overrides(
    point: dict[str, Any],
    *,
    run_id: str,
    attempt_id: str,
    results_root: str | Path,
    checkpoint_path: str | Path,
    timezone: str | None = None,
) -> list[str]:
    """Return scalar Hydra-style overrides for one validation job."""

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


def command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    """Return the canonical ``run.py`` command for a config and overrides."""

    return [python, "-u", "run.py", "--config", str(config), *overrides]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def build_jobs(
    points: Sequence[dict[str, Any]],
    *,
    attempt_id: str,
    results_root: str | Path,
    config: str | Path,
    tags_by_architecture: dict[str, list[str]],
    launcher: str,
    python: str = "python",
    timezone: str | None = None,
) -> list[dict[str, Any]]:
    """Return one manifest job record per grid point."""

    jobs = []
    for point in points:
        run_id = run_id_for(point)
        overrides = train_overrides(
            point, run_id=run_id, attempt_id=attempt_id, results_root=results_root, timezone=timezone
        )
        jobs.append(
            {
                "run_id": run_id,
                "train_dir": str(train_run_dir(results_root, run_id)),
                "validation_dir": str(validation_run_dir(results_root, run_id)),
                "train_attempt_dir": str(train_attempt_dir(results_root, run_id, attempt_id)),
                "overrides": overrides,
                "command": shlex.join(command_for(config, overrides, python=python)),
                "choices": {
                    "architecture": point["architecture"],
                    "normalization": point["normalization"],
                    "lr": float(point["lr"]),
                    "channels": int(point["channels"]),
                    "seed": int(point["seed"]),
                },
                "tags": list(tags_by_architecture.get(str(point["architecture"]), [])),
                "submitted": False,
                "launcher": launcher,
                "launcher_job_id": None,
            }
        )
    return jobs


def build_manifest(
    *,
    attempt_id: str,
    created_at: str,
    config: str | Path,
    grid: str | Path,
    results_root: str | Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the ``00_grid`` manifest describing planned jobs."""

    return {
        "study": "pair_stability",
        "stage": STAGE_GRID,
        "attempt_id": attempt_id,
        "created_at": created_at,
        "config": str(config),
        "grid": str(grid),
        "results_root": str(results_root),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


def write_grid_attempt(
    *,
    results_root: str | Path,
    attempt_id: str,
    created_at: str,
    config: str | Path,
    grid: str | Path,
    grid_data: Any,
    jobs: list[dict[str, Any]],
) -> Path:
    """Write the durable ``00_grid`` attempt and return its directory."""

    attempt = grid_attempt_dir(results_root, attempt_id)
    (attempt / "jobs").mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        attempt_id=attempt_id,
        created_at=created_at,
        config=config,
        grid=grid,
        results_root=results_root,
        jobs=jobs,
    )
    write_json(attempt / "manifest.json", manifest)

    # Snapshot the inputs that produced this plan.
    OmegaConf.save(OmegaConf.create(grid_data), attempt / "grid.yaml")
    config_text = Path(config).read_text() if Path(config).exists() else ""
    (attempt / "pair_stability.yaml").write_text(config_text)

    # Exact commands the orchestrator would run (or did run).
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# pair_stability 00_grid attempt {attempt_id}", ""]
    lines += [job["command"] for job in jobs]
    (attempt / "commands.sh").write_text("\n".join(lines) + "\n")

    # Per-job specs for downstream stages.
    for job in jobs:
        write_json(attempt / "jobs" / f"{job['run_id']}.json", job)

    write_latest(stage_dir(results_root, STAGE_GRID), attempt_id)
    return attempt


def plan_validation_attempt(
    job: dict[str, Any],
    *,
    results_root: str | Path,
    train_attempt_id: str,
    validation_attempt_id: str,
    validation_config: str | Path,
    python: str = "python",
    timezone: str | None = None,
) -> dict[str, Any]:
    """Plan one validation attempt and record the train attempt it consumes.

    Writes ``source_train_attempt.json`` into the validation attempt directory
    and returns the validation command plus provenance. This makes the exact
    train attempt/checkpoint a validation run consumes durable, rather than
    relying on ``run_id`` alone.
    """

    run_id = str(job["run_id"])
    point = {axis: job["choices"][axis] for axis in GRID_AXES}
    train_attempt = train_attempt_dir(results_root, run_id, train_attempt_id)
    checkpoint_path = train_attempt / "checkpoints"
    validation_attempt = validation_attempt_dir(results_root, run_id, validation_attempt_id)

    source = {
        "run_id": run_id,
        "train_attempt_id": train_attempt_id,
        "train_dir": str(train_run_dir(results_root, run_id)),
        "train_attempt_dir": str(train_attempt),
        "checkpoint_path": str(checkpoint_path),
    }
    write_json(validation_attempt / "source_train_attempt.json", source)

    overrides = validation_overrides(
        point,
        run_id=run_id,
        attempt_id=validation_attempt_id,
        results_root=results_root,
        checkpoint_path=checkpoint_path,
        timezone=timezone,
    )
    return {
        "run_id": run_id,
        "validation_attempt_dir": str(validation_attempt),
        "source_train_attempt": source,
        "command": shlex.join(command_for(validation_config, overrides, python=python)),
    }


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse orchestrator command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default=str(DEFAULT_GRID), help="Grid YAML path.")
    parser.add_argument("--config", default=None, help="Train config path (defaults to grid.config).")
    parser.add_argument("--results-root", default=None, help="Results root (defaults to grid.results_root).")
    parser.add_argument(
        "--attempt-id", default=None, help="Attempt id in the study timezone (defaults to now)."
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_STUDY_TIMEZONE,
        help=(
            "IANA timezone owned by the orchestrator: stamps attempt ids and "
            "overrides run.timezone when it differs (default America/New_York)."
        ),
    )
    parser.add_argument("--tags", nargs="*", default=None, help="Only include architectures with all of these tags.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of planned jobs.")
    parser.add_argument("--python", default=sys.executable or "python", help="Python executable for commands.")
    parser.add_argument(
        "--backend",
        choices=["plan", "local", "submitit"],
        default="plan",
        help="plan: write the grid attempt only; local/submitit: also submit train jobs.",
    )
    parser.add_argument("--repo-root", default=None, help="Repo root for command working directory.")
    parser.add_argument("--slurm-partition", default="kozinsky_gpu")
    parser.add_argument("--slurm-gpus", type=int, default=1)
    parser.add_argument("--slurm-timeout-min", type=int, default=480)
    parser.add_argument("--slurm-mem-gb", type=int, default=32)
    parser.add_argument("--slurm-cpus", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan (and optionally submit) the pair-stability grid."""

    args = parse_args(argv)
    grid_path = Path(args.grid)
    grid_data = OmegaConf.to_container(OmegaConf.load(grid_path), resolve=True)
    config = args.config or grid_data["config"]
    results_root = args.results_root or grid_data["results_root"]
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]

    # The orchestrator owns the timezone: it stamps the attempt id / created_at
    # and always injects it as a run.timezone override on the compiled commands.
    tz = resolve_timezone(args.timezone)
    attempt_id = args.attempt_id or new_attempt_id(tz=tz)
    created_at = datetime.now(tz).isoformat(timespec="seconds")

    points = expand_grid(grid_data["grid"])
    config_obj = OmegaConf.load(config)
    validate_grid(points, config_obj)
    tags_by_arch = architecture_tags(config_obj)

    if args.tags:
        wanted = set(args.tags)
        kept = [p for p in points if wanted.issubset(set(tags_by_arch.get(str(p["architecture"]), [])))]
        if len(kept) < len(points):
            print(f"[pair_stability] tag filter {sorted(wanted)}: {len(kept)}/{len(points)} jobs kept")
        points = kept
    if args.limit is not None and args.limit < len(points):
        print(f"[pair_stability] --limit {args.limit}: dropping {len(points) - args.limit} of {len(points)} jobs")
        points = points[: args.limit]

    jobs = build_jobs(
        points,
        attempt_id=attempt_id,
        results_root=results_root,
        config=config,
        tags_by_architecture=tags_by_arch,
        launcher=args.backend,
        python=args.python,
        timezone=args.timezone,
    )
    attempt = write_grid_attempt(
        results_root=results_root,
        attempt_id=attempt_id,
        created_at=created_at,
        config=config,
        grid=grid_path,
        grid_data=grid_data,
        jobs=jobs,
    )
    print(f"[pair_stability] wrote 00_grid attempt {attempt_id} with {len(jobs)} jobs -> {attempt}")

    if args.backend == "plan":
        return 0

    commands = [command_for(config, job["overrides"], python=args.python) for job in jobs]
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
            log_dir=stage_dir(results_root, STAGE_TRAIN) / "slurm_logs" / attempt_id,
            job_name="hooke-pair-stability",
            slurm=slurm,
        )

    # Record submission outcome back into the manifest.
    for job, job_id in zip(jobs, job_ids, strict=True):
        job["submitted"] = True
        job["launcher_job_id"] = job_id
    write_json(
        grid_attempt_dir(results_root, attempt_id) / "manifest.json",
        build_manifest(
            attempt_id=attempt_id,
            created_at=created_at,
            config=config,
            grid=grid_path,
            results_root=results_root,
            jobs=jobs,
        ),
    )
    print(f"[pair_stability] submitted {len(job_ids)} jobs via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
