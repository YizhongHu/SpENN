"""Hydra Submitit launcher for Hooke pair final benchmark jobs."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

try:
    from launch_submitit import _hydra_override_value, _patch_hydra_argparse_for_python314, load_manifest
except ModuleNotFoundError:  # pragma: no cover - exercised by importlib file loading
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from launch_submitit import _hydra_override_value, _patch_hydra_argparse_for_python314, load_manifest

DEFAULT_MANIFEST = "experiments/hooke/studies/pair_validation/manifest.yaml"
DEFAULT_INPUTS = "experiments/hooke/studies/pair_validation/reports/final_eval_inputs.csv"
STAGES = ("final_train", "final_eval")

_patch_hydra_argparse_for_python314()


@hydra.main(version_base=None, config_path=".", config_name="launch_final_submitit")
def main(cfg: DictConfig) -> None:
    """Run one final train/eval config inside a Hydra Submitit job."""

    inputs_path = Path(to_absolute_path(str(cfg.inputs)))
    jobs = stage_jobs(
        read_inputs(inputs_path),
        stage=str(cfg.stage),
        python=str(cfg.python),
        device=str(cfg.device),
    )
    job_index = int(cfg.job_index)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"job_index={job_index} outside {cfg.stage} job count {len(jobs)}")

    job = jobs[job_index]
    if str(cfg.stage) == "final_eval" and bool(cfg.require_checkpoint) and not bool(cfg.dry_run):
        checkpoint = Path(str(job["checkpoint_path"]))
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"final_eval job requires checkpoint before submission can run: {checkpoint}"
            )

    command = list(job["command"])
    print(" ".join(command), flush=True)
    if bool(cfg.dry_run):
        return
    subprocess.run(command, cwd=get_original_cwd(), check=True)


def read_inputs(path: str | Path) -> list[dict[str, Any]]:
    """Read ``final_eval_inputs.csv`` rows."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def stage_jobs(
    inputs: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    python: str = "python",
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Return unique jobs for ``final_train`` or ``final_eval``."""

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in inputs:
        if stage == "final_train":
            command = _row_command(row, "train_command", str(row["train_config"]), python=python)
            key = (str(row["training_seed"]), str(row["train_config"]), " ".join(command))
            if key in seen:
                continue
            seen.add(key)
            config_path = str(row["train_config"])
        else:
            command = _row_command(row, "eval_command", str(row["eval_config"]), python=python)
            key = (str(row["training_seed"]), str(row["eval_seed"]), str(row["eval_config"]), " ".join(command))
            if key in seen:
                continue
            seen.add(key)
            config_path = str(row["eval_config"])
        if device:
            command = [*command, f"runtime.device={device}"]
        jobs.append(
            {
                **dict(row),
                "stage": stage,
                "config": config_path,
                "command": command,
            }
        )
    return jobs


def _row_command(row: Mapping[str, Any], key: str, config_path: str, *, python: str) -> list[str]:
    """Return a command from an inputs row, falling back to the config path."""

    command = str(row.get(key) or "").strip()
    if command:
        parts = shlex.split(command)
        if parts and parts[0] == "python" and python != "python":
            parts[0] = python
        return parts
    return [python, "-u", "run.py", "--config", config_path]


def job_index_sweep(inputs: Sequence[Mapping[str, Any]], *, stage: str) -> str:
    """Return a Hydra override sweep over final-stage job indexes."""

    count = len(stage_jobs(inputs, stage=stage))
    if count <= 0:
        raise ValueError(f"no jobs found for stage {stage!r}")
    return ",".join(str(index) for index in range(count))


def hydra_overrides(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    device: str,
    job_count: int,
) -> list[str]:
    """Return Submitit overrides for a final benchmark stage."""

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    final_eval = manifest.get("final_evaluation") if isinstance(manifest.get("final_evaluation"), Mapping) else {}
    launcher = final_eval.get("launcher") if isinstance(final_eval.get("launcher"), Mapping) else {}
    top_launcher = manifest.get("launcher") if isinstance(manifest.get("launcher"), Mapping) else {}
    slurm_profiles = _select_slurm_profiles(launcher, top_launcher, stage)
    profile_name = "gpu" if device == "cuda" else "cpu"
    profile = slurm_profiles.get(profile_name) if isinstance(slurm_profiles.get(profile_name), Mapping) else {}

    sweep_base = str(launcher.get("hydra_sweep_dir") or "slurm_logs/hooke_pair_final_v1")
    job_prefix = str(launcher.get("job_name_prefix") or "hooke-final-v1")
    job_name = f"{job_prefix}-{stage.replace('_', '-')}"
    sweep_dir = f"{sweep_base}/{stage}"

    overrides = [
        f"hydra.job.name={job_name}",
        f"hydra.launcher.name={job_name}",
        f"hydra.sweep.dir={sweep_dir}",
        f"hydra.launcher.submitit_folder={sweep_dir}/.submitit/%j",
    ]
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


def _select_slurm_profiles(
    launcher: Mapping[str, Any],
    top_launcher: Mapping[str, Any],
    stage: str,
) -> Mapping[str, Any]:
    slurm = launcher.get("slurm") if isinstance(launcher.get("slurm"), Mapping) else {}
    stage_profiles = slurm.get(stage) if isinstance(slurm.get(stage), Mapping) else {}
    if stage_profiles:
        return stage_profiles
    top_slurm = top_launcher.get("slurm") if isinstance(top_launcher.get("slurm"), Mapping) else {}
    return top_slurm


def _utility_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--stage", choices=STAGES, default="final_train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-job-index-sweep", action="store_true")
    parser.add_argument("--print-hydra-overrides", action="store_true")
    return parser


def _run_utility(argv: Sequence[str]) -> bool:
    if not any(arg.startswith("--print-") for arg in argv):
        return False
    args = _utility_parser().parse_args(argv)
    inputs = read_inputs(args.inputs)
    if args.print_job_index_sweep:
        print(job_index_sweep(inputs, stage=args.stage))
    if args.print_hydra_overrides:
        manifest = load_manifest(args.manifest)
        job_count = len(stage_jobs(inputs, stage=args.stage))
        print(
            "\n".join(
                hydra_overrides(
                    manifest,
                    stage=args.stage,
                    device=str(args.device),
                    job_count=job_count,
                )
            )
        )
    return True


__all__ = [
    "STAGES",
    "hydra_overrides",
    "job_index_sweep",
    "read_inputs",
    "stage_jobs",
]


if __name__ == "__main__":  # pragma: no cover
    if not _run_utility(sys.argv[1:]):
        main()
