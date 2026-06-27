"""Shared launcher plumbing for stage-plan backed submissions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from .execution import ExecutionRecord, write_execution_records
from .executors import ExecutorOptions, LocalExecutor, SubmissionRequest, SubmititExecutor
from .resources import ResourceSpec, resource_from_profile
from .specs import StagePlan


class LaunchAdapter(Protocol):
    """Launcher module surface needed by the toolkit adapters."""

    def selected_device(self, args: Any) -> str:
        """Return the selected runtime device expression."""

    def device_profiles(self, selector: str) -> Sequence[str]:
        """Return concrete launcher profiles for a device expression."""

    def resolve_uv_settings_for_profile(self, args: Any, profile: str) -> tuple[str | None, Sequence[str], str]:
        """Return uv environment/extras and runtime device for one profile."""

    def slurm_parameters(self, args: Any, *, profile: str, smoke: bool) -> dict[str, Any]:
        """Return Slurm parameters for one profile."""

    def submit_command_sets(self, command_sets: dict[str, list[list[str]]], **kwargs: Any) -> Sequence[str]:
        """Submit command sets and return launcher job ids."""

    def claim_paths_for_statuses(
        self,
        paths: Sequence[str | Path | None] | None,
    ) -> Sequence[str | Path | None] | None:
        """Return row-claim paths for launcher status paths."""


def stage_plan_directory(results_root: str | Path, stage: str, attempt_id: str) -> Path:
    """Return the conventional durable stage-plan directory."""

    return Path(results_root) / stage / "stage_plans" / str(attempt_id)


def resource_spec_from_launcher(launcher: LaunchAdapter, args: Any) -> ResourceSpec:
    """Return a backend-neutral resource request for launcher args."""

    selector = launcher.selected_device(args)
    profiles = launcher.device_profiles(selector)
    resolved_profiles = {}
    for profile in profiles:
        uv_environment, uv_extras, _runtime_device = launcher.resolve_uv_settings_for_profile(args, profile)
        slurm = launcher.slurm_parameters(args, profile=profile, smoke=bool(args.smoke))
        resolved_profiles[profile] = resource_from_profile(
            profile=profile,
            partition=slurm.get("slurm_partition"),
            timeout_min=slurm.get("timeout_min"),
            mem_gb=slurm.get("mem_gb"),
            cpus=slurm.get("cpus_per_task"),
            gpus=slurm.get("gpus_per_node"),
            uv_environment=uv_environment,
            uv_extras=uv_extras,
        ).to_dict()
    if len(profiles) == 1:
        profile = profiles[0]
        return resource_from_profile(
            profile=profile,
            partition=resolved_profiles[profile].get("partition"),
            timeout_min=resolved_profiles[profile].get("timeout_min"),
            mem_gb=resolved_profiles[profile].get("mem_gb"),
            cpus=resolved_profiles[profile].get("threads"),
            gpus=resolved_profiles[profile].get("gpus"),
            uv_environment=resolved_profiles[profile].get("uv_environment"),
            uv_extras=resolved_profiles[profile].get("uv_extras", ()),
        )
    return resource_from_profile(
        profile=selector,
        partition=None,
        timeout_min=None,
        mem_gb=None,
        cpus=None,
        gpus=None,
        uv_environment=None,
        uv_extras=(),
        metadata={"profiles": resolved_profiles},
    )


def executor_from_launcher(
    launcher: LaunchAdapter,
    *,
    args: Any,
    repo_root: str | Path,
    log_dir: str | Path,
    job_name: str,
    chunk_status_dir: str | Path | None = None,
    allow_partial_failures: bool = False,
    claim_rows: bool = False,
):
    """Return the toolkit executor selected by launcher args."""

    options = ExecutorOptions(
        backend=str(args.backend),
        args=args,
        repo_root=repo_root,
        log_dir=log_dir,
        job_name=job_name,
        smoke=bool(args.smoke),
        chunk_size=int(args.chunk_size),
        allow_partial_failures=allow_partial_failures,
        claim_rows=claim_rows,
        chunk_status_dir=chunk_status_dir,
    )
    executor_cls = LocalExecutor if args.backend == "local" else SubmititExecutor
    return executor_cls(
        submit_command_sets=getattr(launcher, "submit_command_sets"),
        options=options,
        claim_paths_for_statuses=getattr(launcher, "claim_paths_for_statuses", None),
    )


def submit_stage_plan(
    launcher: LaunchAdapter,
    *,
    stage_plan: StagePlan,
    stage_plan_dir: str | Path,
    command_sets: dict[str, list[list[str]]],
    submitted_commands: Sequence[Sequence[str]],
    args: Any,
    repo_root: str | Path,
    log_dir: str | Path,
    job_name: str,
    chunk_status_dir: str | Path | None = None,
    allow_partial_failures: bool = False,
    claim_rows: bool = False,
) -> tuple[ExecutionRecord, ...]:
    """Write, submit, and record one stage plan."""

    plan_dir = stage_plan.write(stage_plan_dir)
    execution_records = executor_from_launcher(
        launcher,
        args=args,
        repo_root=repo_root,
        log_dir=log_dir,
        job_name=job_name,
        chunk_status_dir=chunk_status_dir,
        allow_partial_failures=allow_partial_failures,
        claim_rows=claim_rows,
    ).submit(
        stage_plan,
        stage_plan.tasks,
        SubmissionRequest(
            command_sets=command_sets,
            submitted_commands=submitted_commands,
        ),
    )
    write_execution_records(plan_dir, execution_records)
    return tuple(execution_records)
