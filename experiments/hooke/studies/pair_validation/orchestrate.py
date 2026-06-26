"""Orchestrator helper for Hooke pair study phases."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .study_manifest import (
        DEFAULT_MANIFEST,
        EVAL_PHASES,
        TRAIN_PHASES,
        cartesian_phase_jobs,
        command_for_job,
        final_eval_plan_dir,
        final_train_report_dir,
        job_index_sweep,
        load_jsonl,
        load_yaml,
        phase_for_launch,
        profile_device,
        profile_environment,
        select_report_dir,
        slurm_options,
    )
except ImportError:  # pragma: no cover - direct script execution
    from study_manifest import (
        DEFAULT_MANIFEST,
        EVAL_PHASES,
        TRAIN_PHASES,
        cartesian_phase_jobs,
        command_for_job,
        final_eval_plan_dir,
        final_train_report_dir,
        job_index_sweep,
        load_jsonl,
        load_yaml,
        phase_for_launch,
        profile_device,
        profile_environment,
        select_report_dir,
        slurm_options,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run orchestrator CLI."""

    args = _parse_args(argv)
    manifest = load_yaml(args.manifest)
    target_phase, smoke = phase_for_launch(args.phase, target_phase=args.target_phase)
    slurm_profile = _slurm_profile_for_launch(args.profile, smoke=smoke)
    repo_root = _repo_root_for_manifest(args.manifest, manifest)
    try:
        jobs = phase_jobs(
            manifest=manifest,
            kind=args.kind,
            phase=args.phase,
            jobs_path=args.jobs,
            selected_config_path=args.selected_config,
            target_phase=args.target_phase,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        hint = _planning_hint(args, manifest)
        if hint is not None:
            print(f"hint: {hint}", file=sys.stderr)
        return 2
    if args.print_job_index_sweep:
        print(job_index_sweep(len(jobs)))
        return 0

    selected_jobs = _selected_jobs(jobs, args.job_index)
    device = profile_device(manifest, args.profile)
    try:
        run_python = _run_python(args, manifest, repo_root=repo_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    commands = [
        command_for_job(job, device=device, python=str(run_python), repo_root=repo_root) for job in selected_jobs
    ]
    if args.backend == "slurm":
        sbatch_command = _sbatch_command(
            args,
            manifest,
            job_count=len(jobs),
            target_phase=target_phase,
            slurm_profile=slurm_profile,
            repo_root=repo_root,
            run_python=run_python,
        )
        _print_slurm_plan(
            args,
            manifest,
            commands,
            sbatch_command,
            job_count=len(jobs),
            target_phase=target_phase,
            slurm_profile=slurm_profile,
            repo_root=repo_root,
        )
        if args.dry_run:
            return 0
        _submit_sbatch(sbatch_command, repo_root=repo_root)
        return 0

    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=repo_root)
    return 0


def phase_jobs(
    *,
    manifest: Mapping[str, Any],
    kind: str,
    phase: str,
    jobs_path: str | Path | None = None,
    selected_config_path: str | Path | None = None,
    target_phase: str | None = None,
) -> list[dict[str, Any]]:
    """Return jobs for one launch phase."""

    resolved_phase, smoke = phase_for_launch(phase, target_phase=target_phase)
    if kind == "train":
        if resolved_phase not in TRAIN_PHASES:
            raise ValueError(f"train orchestrator cannot launch phase {phase!r}")
        selected = load_yaml(selected_config_path) if selected_config_path is not None else None
        if resolved_phase == "final_train" and selected is None:
            raise ValueError("final_train requires --selected-config")
        return cartesian_phase_jobs(manifest, resolved_phase, selected=selected, smoke=smoke)
    if kind == "eval":
        if resolved_phase not in EVAL_PHASES:
            raise ValueError(f"eval orchestrator cannot launch phase {phase!r}")
        if jobs_path is None:
            raise ValueError(f"{phase} requires --jobs")
        if not Path(jobs_path).exists():
            raise FileNotFoundError(f"jobs file not found: {jobs_path}")
        jobs = load_jsonl(jobs_path)
        expected = "smoke_eval" if smoke else resolved_phase
        return [row for row in jobs if str(row.get("phase")) == expected or not row.get("phase")]
    raise ValueError(f"kind must be train or eval, got {kind!r}")


def _selected_jobs(jobs: Sequence[dict[str, Any]], job_index: int | None) -> list[dict[str, Any]]:
    if job_index is None:
        return list(jobs)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"job_index={job_index} outside job count {len(jobs)}")
    return [dict(jobs[job_index])]


def _slurm_profile_for_launch(profile: str, *, smoke: bool) -> str:
    """Return the manifest profile used for Slurm resource selection."""

    if not smoke:
        return profile
    if profile == "cpu":
        return "test"
    if profile == "gpu":
        return "gpu_test"
    raise ValueError(f"unsupported smoke profile {profile!r}")


def _sbatch_command(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    *,
    job_count: int,
    target_phase: str,
    slurm_profile: str,
    repo_root: Path,
    run_python: Path,
) -> list[str]:
    """Return the sbatch command that submits this launch."""

    effective_job_count = 1 if args.job_index is not None else job_count
    options = slurm_options(
        manifest,
        phase=target_phase,
        profile=slurm_profile,
        job_count=effective_job_count,
    )
    log_dir = _resolve_path(str(options["log_dir"]), repo_root)
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={options['job_name']}",
        f"--chdir={repo_root}",
        f"--output={log_dir}/%A_%a.out" if args.job_index is None else f"--output={log_dir}/%j.out",
        f"--error={log_dir}/%A_%a.err" if args.job_index is None else f"--error={log_dir}/%j.err",
    ]
    if args.job_index is None:
        parallelism = max(1, min(int(options["array_parallelism"]), job_count))
        command.append(f"--array=0-{job_count - 1}%{parallelism}")
    for option, flag in (
        ("partition", "--partition"),
        ("gres", "--gres"),
        ("cpus_per_task", "--cpus-per-task"),
        ("mem_gb", "--mem"),
        ("timeout_min", "--time"),
    ):
        if option not in options:
            continue
        value = options[option]
        if option == "mem_gb":
            value = f"{value}G"
        command.append(f"{flag}={value}")
    command.extend(["--wrap", _array_task_wrap_command(args, run_python=run_python)])
    return command


def _array_task_wrap_command(args: argparse.Namespace, *, run_python: Path) -> str:
    """Return the command executed by one Slurm array task."""

    command = [
        str(run_python),
        str(Path(__file__).resolve()),
        "--backend",
        "local",
        "--kind",
        args.kind,
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--phase",
        args.phase,
        "--profile",
        args.profile,
        "--python",
        str(run_python),
    ]
    if args.jobs:
        command.extend(["--jobs", str(Path(args.jobs).resolve())])
    if args.selected_config:
        command.extend(["--selected-config", str(Path(args.selected_config).resolve())])
    if args.target_phase:
        command.extend(["--target-phase", str(args.target_phase)])
    command.append("--job-index")
    job_index = str(args.job_index) if args.job_index is not None else "$SLURM_ARRAY_TASK_ID"
    return f"{shlex.join(command)} {job_index}"


def _print_slurm_plan(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    commands: Sequence[Sequence[str]],
    sbatch_command: Sequence[str],
    *,
    job_count: int,
    target_phase: str,
    slurm_profile: str,
    repo_root: Path,
) -> None:
    options = slurm_options(
        manifest,
        phase=target_phase,
        profile=slurm_profile,
        job_count=1 if args.job_index is not None else job_count,
    )
    print("Slurm launch plan", flush=True)
    print(f"  kind: {args.kind}", flush=True)
    print(f"  phase: {args.phase} -> {target_phase}" if args.phase != target_phase else f"  phase: {args.phase}", flush=True)
    print(f"  requested profile: {args.profile}", flush=True)
    print(f"  slurm profile: {slurm_profile}", flush=True)
    print(f"  jobs: {len(commands)} of {job_count}", flush=True)
    print(f"  cwd: {repo_root}", flush=True)
    print(f"  log_dir: {_resolve_path(str(options['log_dir']), repo_root)}", flush=True)
    for key in ("partition", "gres", "cpus_per_task", "mem_gb", "timeout_min", "array_parallelism"):
        if key in options:
            print(f"  {key}: {options[key]}", flush=True)
    print(f"sbatch: {shlex.join(list(sbatch_command))}", flush=True)
    if commands:
        print(f"job[0]: {shlex.join(list(commands[0]))}", flush=True)


def _submit_sbatch(command: Sequence[str], *, repo_root: Path) -> None:
    """Submit a Slurm job and print the returned job id."""

    _sbatch_log_dir(command).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, check=True, cwd=repo_root, text=True, capture_output=True)
    stdout = result.stdout.strip()
    print(f"submitted Slurm job {stdout}" if stdout else "submitted Slurm job", flush=True)


def _run_python(args: argparse.Namespace, manifest: Mapping[str, Any], *, repo_root: Path) -> Path:
    """Return the Python executable for launched jobs."""

    if args.python is not None:
        path = _executable_path(args.python, repo_root)
        source = f"--python {args.python}"
    else:
        path = _resolve_path(profile_environment(manifest, args.profile), repo_root) / "bin" / "python"
        source = f"profile {args.profile!r} uv_environment"
    if not args.dry_run and not path.exists():
        raise FileNotFoundError(f"{source} expects Python at {path}; create/sync that environment or pass --python")
    return path


def _executable_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        found = shutil.which(str(path))
        return Path(found) if found is not None else path
    return _resolve_path(str(path), repo_root)


def _sbatch_log_dir(command: Sequence[str]) -> Path:
    for item in command:
        if item.startswith("--output="):
            return Path(item.removeprefix("--output=")).parent
    raise ValueError("sbatch command is missing --output")


def _planning_hint(args: argparse.Namespace, manifest: Mapping[str, Any]) -> str | None:
    """Return a planner command hint for missing row-based eval jobs."""

    if args.kind != "eval":
        return None
    resolved_phase, smoke = phase_for_launch(args.phase, target_phase=args.target_phase)
    if resolved_phase != "final_eval":
        return None
    phase = "smoke_eval" if smoke else "final_eval"
    jobs_path = args.jobs or Path(final_eval_plan_dir(manifest)) / f"{phase}_jobs.jsonl"
    plan_dir = Path(jobs_path).parent
    command = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve().with_name("plan_final.py")),
        "--manifest",
        str(args.manifest),
        "--selected-config",
        str(Path(select_report_dir(manifest)) / "selected_config.yaml"),
        "--final-train-runs",
        str(Path(final_train_report_dir(manifest)) / "final_train_runs.csv"),
        "--phase",
        phase,
        "--output-dir",
        str(plan_dir),
    ]
    return "generate the jobs file first with: " + shlex.join(command)


def _repo_root_for_manifest(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    """Infer the repo root from manifest config paths and script location."""

    config_paths = []
    configs = manifest.get("configs")
    if isinstance(configs, Mapping):
        config_paths = [Path(str(path)) for path in configs.values()]

    candidates: list[Path] = []
    candidates.append(Path.cwd().resolve())
    resolved_manifest = Path(manifest_path).resolve()
    candidates.extend([resolved_manifest.parent, *resolved_manifest.parents])
    script = Path(__file__).resolve()
    candidates.extend([script.parent, *script.parents])
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "run.py").exists() and all(
            path.is_absolute() or (candidate / path).exists() for path in config_paths
        ):
            return candidate
    return Path.cwd().resolve()


def _resolve_path(path: str, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  uv run python <study-dir>/orchestrate.py \\
    --kind train --backend local --phase smoke_train --profile cpu --dry-run

  uv run python <study-dir>/orchestrate.py \\
    --kind train --backend slurm --phase smoke_train --profile gpu

  uv run python <study-dir>/orchestrate.py \\
    --kind train --backend slurm --phase validation_train --profile gpu

  uv run python <study-dir>/orchestrate.py \\
    --kind eval --backend slurm --phase final_eval --profile gpu \\
    --jobs <report-root>/05_final_eval/plans/final_eval_jobs.jsonl
""",
    )
    parser.add_argument("--backend", choices=("local", "slurm"), default="local", help="Execution backend.")
    parser.add_argument("--kind", choices=("train", "eval"), required=True, help="Run train or eval jobs.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path, help="Study manifest path.")
    parser.add_argument("--phase", required=True, help="Manifest phase or smoke alias to launch.")
    parser.add_argument("--profile", choices=("cpu", "gpu"), default="cpu", help="Runtime device profile.")
    parser.add_argument("--jobs", type=Path, help="JSONL job rows for row-based eval phases.")
    parser.add_argument("--selected-config", type=Path, help="Selected config artifact for final_train.")
    parser.add_argument("--target-phase", help="Manifest phase used by a smoke alias.")
    parser.add_argument("--job-index", type=int, help="Run only one job index from the phase job list.")
    parser.add_argument(
        "--python",
        type=Path,
        help="Python executable for launched jobs; defaults to the profile uv_environment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing or submitting.")
    parser.add_argument("--print-job-index-sweep", action="store_true", help="Print comma-separated job indexes.")
    return parser.parse_args(argv)


__all__ = ["main", "phase_jobs"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
