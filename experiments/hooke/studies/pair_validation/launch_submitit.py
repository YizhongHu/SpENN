"""Hydra Submitit launcher for the Hooke pair validation scan."""

from __future__ import annotations

import argparse
import itertools
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf


DEFAULT_MANIFEST = "experiments/hooke/studies/pair_validation/manifest.yaml"


def _patch_hydra_argparse_for_python314() -> None:
    """Let Hydra 1.3 lazy help objects pass Python 3.14 argparse validation."""

    original = argparse.ArgumentParser._check_help

    def _check_help(self, action):  # type: ignore[no-untyped-def]
        if action.help is not None and not isinstance(action.help, str):
            return
        return original(self, action)

    if getattr(argparse.ArgumentParser._check_help, "_spenn_hydra_py314_patch", False):
        return
    _check_help._spenn_hydra_py314_patch = True
    argparse.ArgumentParser._check_help = _check_help


_patch_hydra_argparse_for_python314()


@hydra.main(version_base=None, config_path=".", config_name="launch_submitit")
def main(cfg: DictConfig) -> None:
    """Run one manifest grid point inside a Hydra multirun job."""

    manifest_path = Path(to_absolute_path(str(cfg.manifest)))
    manifest = load_manifest(manifest_path)
    jobs = manifest_jobs(manifest)
    job_index = int(cfg.job_index)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"job_index={job_index} outside manifest grid size {len(jobs)}")

    command = run_command(
        manifest=manifest,
        job=jobs[job_index],
        run_root=_run_root(manifest, cfg),
        device=str(cfg.device),
        python=str(cfg.python),
    )
    print(" ".join(command), flush=True)
    if bool(cfg.dry_run):
        return
    subprocess.run(command, cwd=get_original_cwd(), check=True)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a study manifest as plain Python data."""

    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return data


def manifest_jobs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return manifest grid jobs with the first grid axis varying fastest."""

    grid = manifest.get("grid")
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("manifest grid must be a non-empty mapping")
    axes = [(str(key), _as_sequence(value, key=str(key))) for key, value in grid.items()]
    jobs: list[dict[str, Any]] = []
    reversed_axes = list(reversed(axes))
    for reversed_values in itertools.product(*(values for _, values in reversed_axes)):
        values = list(reversed(reversed_values))
        jobs.append({key: value for (key, _), value in zip(axes, values, strict=True)})
    return jobs


def run_command(
    *,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    run_root: str,
    device: str,
    python: str = "python",
) -> list[str]:
    """Build the direct venv Python command for one manifest grid job."""

    train_config = manifest.get("train_config")
    study = manifest.get("study")
    if not train_config:
        raise ValueError("manifest train_config is required")
    if not isinstance(study, Mapping) or not study.get("name"):
        raise ValueError("manifest study.name is required")
    overrides: dict[str, Any] = {
        "run.root": run_root,
        "study.name": study["name"],
        "study.config_id": config_id(job, seed_key=str(manifest.get("seed_key", "runtime.seed"))),
        "runtime.device": device,
    }
    overrides.update(job)
    command = [
        python,
        "-u",
        "run.py",
        "--config",
        str(train_config),
    ]
    command.extend(f"{key}={_dotlist_value(value)}" for key, value in overrides.items())
    return command


def config_id(job: Mapping[str, Any], *, seed_key: str) -> str:
    """Return a deterministic non-seed config id for validation grouping."""

    parts = []
    for key, value in job.items():
        if key == seed_key:
            continue
        parts.append(f"{key.split('.')[-1]}{_slug(value)}")
    return "config_" + "_".join(parts)


def job_index_sweep(manifest: Mapping[str, Any]) -> str:
    """Return a Hydra override sweep over every manifest job index."""

    count = len(manifest_jobs(manifest))
    if count <= 0:
        raise ValueError("manifest grid produced no jobs")
    return ",".join(str(index) for index in range(count))


def hydra_overrides(manifest: Mapping[str, Any], *, device: str) -> list[str]:
    """Return Hydra Submitit overrides derived from manifest launcher metadata."""

    launcher = manifest.get("launcher") if isinstance(manifest.get("launcher"), Mapping) else {}
    slurm_profiles = launcher.get("slurm") if isinstance(launcher.get("slurm"), Mapping) else {}
    profile_name = "gpu" if device == "cuda" else "cpu"
    profile = slurm_profiles.get(profile_name) if isinstance(slurm_profiles.get(profile_name), Mapping) else {}
    job_count = len(manifest_jobs(manifest))

    overrides: list[str] = []
    if launcher.get("job_name"):
        overrides.append(f"hydra.job.name={launcher['job_name']}")
        overrides.append(f"hydra.launcher.name={launcher['job_name']}")
    if launcher.get("hydra_sweep_dir"):
        sweep_dir = str(launcher["hydra_sweep_dir"])
        overrides.append(f"hydra.sweep.dir={sweep_dir}")
        overrides.append(f"hydra.launcher.submitit_folder={sweep_dir}/.submitit/%j")

    for manifest_key, hydra_key in (
        ("partition", "hydra.launcher.partition"),
        ("gres", "hydra.launcher.gres"),
        ("cpus_per_task", "hydra.launcher.cpus_per_task"),
        ("mem_gb", "hydra.launcher.mem_gb"),
        ("timeout_min", "hydra.launcher.timeout_min"),
    ):
        value = profile.get(manifest_key)
        if value is not None:
            overrides.append(f"{hydra_key}={_hydra_override_value(value)}")

    array_parallelism = profile.get("array_parallelism")
    if array_parallelism is None:
        array_parallelism = job_count
    overrides.append(f"hydra.launcher.array_parallelism={int(array_parallelism)}")
    return overrides


def _run_root(manifest: Mapping[str, Any], cfg: DictConfig) -> str:
    value = cfg.get("run_root")
    if value:
        return str(value)
    launcher = manifest.get("launcher") if isinstance(manifest.get("launcher"), Mapping) else {}
    if launcher.get("run_root"):
        return str(launcher["run_root"])
    study = manifest.get("study") if isinstance(manifest.get("study"), Mapping) else {}
    study_name = str(study.get("name", "hooke_pair_validation"))
    return f"outputs/{study_name}"


def _as_sequence(value: Any, *, key: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"manifest grid axis {key!r} must be a sequence")
    if not value:
        raise ValueError(f"manifest grid axis {key!r} must be non-empty")
    return list(value)


def _dotlist_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:g}"
    return str(value)


def _hydra_override_value(value: Any) -> str:
    """Return a value string that Hydra will parse as one override value."""

    text = _dotlist_value(value)
    if isinstance(value, str):
        return text.replace("\\", "\\\\").replace(",", "\\,")
    return text


def _slug(value: Any) -> str:
    text = _dotlist_value(value).strip().lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-")


def _utility_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-job-index-sweep", action="store_true")
    parser.add_argument("--print-hydra-overrides", action="store_true")
    return parser


def _run_utility(argv: Sequence[str]) -> bool:
    if not any(arg.startswith("--print-") for arg in argv):
        return False
    args = _utility_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.print_job_index_sweep:
        print(job_index_sweep(manifest))
    if args.print_hydra_overrides:
        print("\n".join(hydra_overrides(manifest, device=str(args.device))))
    return True


__all__ = [
    "config_id",
    "hydra_overrides",
    "job_index_sweep",
    "load_manifest",
    "manifest_jobs",
    "run_command",
]


if __name__ == "__main__":  # pragma: no cover
    if not _run_utility(sys.argv[1:]):
        main()
