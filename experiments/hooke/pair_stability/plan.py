"""Plan the Hooke pair-stability grid.

Expands the architecture x normalization x lr x channels x seed grid into
scalar override lists for the canonical ``run.py`` entrypoint and writes a
durable ``00_grid`` attempt (manifest + commands) describing the planned train
jobs.

Stage layout (under ``results_root``)::

    00_grid/{attempt_id}/{manifest.json, commands.sh, grid.yaml,
                          pair_stability.yaml, jobs/{run_id}.json}
    01_train/{run_id}/{attempt_id}/...
    02_validation/{run_id}/{attempt_id}/...
    03_collect/{attempt_id}/...
    04_select/{attempt_id}/...
"""

from __future__ import annotations

import argparse
import itertools
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from run_ids import GRID_AXES, run_id_for
from utils.io import write_json
from utils.layout import (
    STAGE_GRID,
    STAGE_TRAIN,
    grid_attempt_dir,
    stage_dir,
    train_attempt_dir,
    train_run_dir,
    validation_run_dir,
    write_latest,
)
from utils.time import DEFAULT_STUDY_TIMEZONE, new_attempt_id, resolve_timezone

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
    """Return scalar OmegaConf-style overrides for one train job."""

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
                "launcher": None,
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

    # Exact commands that train.py will read from this attempt.
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# pair_stability 00_grid attempt {attempt_id}", ""]
    lines += [job["command"] for job in jobs]
    (attempt / "commands.sh").write_text("\n".join(lines) + "\n")

    validation_config = grid_data.get("validation_config") if isinstance(grid_data, dict) else None
    if validation_config is not None and Path(validation_config).exists():
        (attempt / "pair_validation.yaml").write_text(Path(validation_config).read_text())

    # Per-job specs for downstream stages.
    for job in jobs:
        write_json(attempt / "jobs" / f"{job['run_id']}.json", job)

    write_latest(stage_dir(results_root, STAGE_GRID), attempt_id)
    return attempt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse planner command-line arguments."""

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
            "IANA timezone owned by the planner: stamps attempt ids and "
            "overrides run.timezone (default America/New_York)."
        ),
    )
    parser.add_argument("--tags", nargs="*", default=None, help="Only include architectures with all of these tags.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of planned jobs.")
    parser.add_argument(
        "--python",
        default="python",
        help=(
            "Python executable name recorded in planned commands. The train "
            "launcher chooses the CPU/CUDA uv environment at launch time."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan the pair-stability grid and write a ``00_grid`` attempt."""

    args = parse_args(argv)
    grid_path = Path(args.grid)
    grid_data = OmegaConf.to_container(OmegaConf.load(grid_path), resolve=True)
    config = args.config or grid_data["config"]
    results_root = args.results_root or grid_data["results_root"]

    # The planner owns the timezone for this study: it stamps the attempt id /
    # created_at and injects run.timezone into the compiled train commands.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
