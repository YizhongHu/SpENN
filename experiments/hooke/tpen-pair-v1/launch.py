"""Run the TPEN pair-v1 smoke from an existing accelerator allocation.

This module is intentionally allocation-local.  Scheduler envelopes provision
the environment and enter the allocation; this launcher only builds a typed
task plan and executes it with :class:`AllocationPoolExecutor`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Direct execution starts with this directory on ``sys.path``; add the
# checkout so the shared experiment toolkit remains importable.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.toolkit import (
    AllocationPoolExecutor,
    CompletionSpec,
    ResourceSpec,
    StagePlan,
    SubmissionRequest,
    TaskSpec,
    task_id_from_parts,
)


CONFIG_PATH = "experiments/hooke/tpen-pair-v1/configs/train.yaml"


def repository_root() -> Path:
    """Return the checkout root containing this launcher."""

    return _REPOSITORY_ROOT


def _non_empty(value: str, name: str) -> str:
    """Return a required non-empty command-line value."""

    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _visibility_values(raw: Sequence[str]) -> tuple[str, ...]:
    """Normalize visibility values while rejecting empty worker bindings."""

    values = tuple(value.strip() for value in raw)
    if not values or any(not value for value in values):
        raise ValueError("--visibility-values must contain non-empty values")
    return values


def _outside_checkout(results_root: Path, checkout: Path) -> Path:
    """Resolve a results root and reject paths inside the checkout."""

    resolved = results_root.expanduser().resolve()
    try:
        resolved.relative_to(checkout.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"results root must be outside repository checkout: {resolved}")


def _worker_count(args: argparse.Namespace, values: tuple[str, ...]) -> int:
    """Resolve and validate the allocation worker count."""

    workers = len(values) if args.workers is None else int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if workers != len(values):
        raise ValueError(
            f"worker/visibility-value count mismatch: {workers} workers, {len(values)} values"
        )
    return workers


def build_command(args: argparse.Namespace, results_root: Path) -> tuple[str, ...]:
    """Build the exact argv used for one pair-v1 smoke task."""

    return (
        _non_empty(str(args.python), "--python"),
        "run.py",
        "--config",
        CONFIG_PATH,
        f"run.root={results_root}",
        f"run.run_id={_non_empty(str(args.run_id), '--run-id')}",
        f"runtime.device={args.device}",
    )


def build_plan(args: argparse.Namespace, *, checkout: Path | None = None) -> tuple[StagePlan, SubmissionRequest]:
    """Build the pair-v1 task plan and allocation-local submission request.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed launcher arguments.
    checkout : pathlib.Path, optional
        Checkout root, injectable for tests.
    """

    checkout = (checkout or repository_root()).resolve()
    python = _non_empty(str(args.python), "--python")
    run_id = _non_empty(str(args.run_id), "--run-id")
    visibility_variable = _non_empty(str(args.visibility_variable), "--visibility-variable")
    values = _visibility_values(args.visibility_values)
    _worker_count(args, values)
    results_root = _outside_checkout(Path(args.results_root), checkout)
    run_root = results_root / run_id
    command = build_command(args, results_root)
    stage = "01_train"
    attempt_id = str(args.pass_id)
    task = TaskSpec(
        task_id=task_id_from_parts(stage=stage, run_id=run_id, attempt_id=attempt_id),
        stage=stage,
        attempt_id=attempt_id,
        run_id=run_id,
        command=command,
        result_dir=str(run_root),
        logs=(str(run_root / "launcher-status.json"),),
        resources=ResourceSpec(profile="gpu", device=args.device, threads=1, gpus=1),
        completion=CompletionSpec(policy="status_completed", status_path=str(run_root / "status.json")),
        metadata={"config": CONFIG_PATH, "runtime_device": args.device},
    )
    plan = StagePlan(
        study="tpen_pair_v1",
        stage=stage,
        attempt_id=attempt_id,
        results_root=str(results_root),
        tasks=(task,),
        smoke=True,
        metadata={"config": CONFIG_PATH},
    )
    request = SubmissionRequest(
        command_sets={"allocation": (command,)},
        submitted_commands=(command,),
    )
    return plan.validate(), request.validate(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for an allocation-local launch."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Python executable in the provisioned environment")
    parser.add_argument("--results-root", required=True, help="Run-results root outside the checkout")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cuda", "xpu"), required=True)
    parser.add_argument("--visibility-variable", required=True)
    parser.add_argument("--visibility-values", nargs="+", required=True)
    parser.add_argument("--workers", type=int, help="Expected allocation worker count")
    parser.add_argument("--deadline")
    parser.add_argument("--deadline-env-var")
    parser.add_argument("--pass-id", default="pass-1")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the plan inside the current allocation."""

    args = build_parser().parse_args(argv)
    try:
        plan, request = build_plan(args)
        if args.dry_run:
            print(" ".join(plan.tasks[0].command))
            print(f"results_root={plan.results_root}")
            print(f"visibility={args.visibility_variable}:{','.join(args.visibility_values)}")
            return 0
        AllocationPoolExecutor(
            pass_id=args.pass_id,
            n_workers=len(args.visibility_values),
            visibility_variable=args.visibility_variable,
            visibility_values=args.visibility_values,
            run_root=plan.results_root,
            working_directory=str(repository_root()),
            deadline=args.deadline,
            deadline_env_var=args.deadline_env_var,
        ).submit(plan, plan.tasks, request)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
