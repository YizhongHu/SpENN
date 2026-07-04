"""Executor interfaces and launcher adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .execution import ExecutionRecord, execution_records_from_submission
from .specs import StagePlan, TaskSpec

Command = Sequence[str]
CommandSets = Mapping[str, Sequence[Command]]
LauncherSubmitter = Callable[..., Sequence[str]]
ClaimPathResolver = Callable[[Sequence[str | Path | None] | None], Sequence[str | Path | None] | None]


class Executor(Protocol):
    """Protocol implemented by local, Submitit, and future executors."""

    def submit(
        self,
        plan: StagePlan,
        tasks: Sequence[TaskSpec],
        request: "SubmissionRequest",
    ) -> Sequence[ExecutionRecord]:
        """Submit ``tasks`` from ``plan`` and return execution records."""


@dataclass(frozen=True)
class ExecutorOptions:
    """Backend-neutral options understood by executor adapters."""

    backend: str
    args: Any = None
    repo_root: str | Path | None = None
    log_dir: str | Path | None = None
    job_name: str | None = None
    smoke: bool = False
    chunk_size: int = 1
    allow_partial_failures: bool = False
    claim_rows: bool = False
    chunk_status_dir: str | Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "ExecutorOptions":
        """Validate the backend submission options and return ``self``."""

        _require_non_empty("backend", self.backend)
        if self.args is None:
            raise ValueError("executor options require launcher args")
        if self.repo_root is None:
            raise ValueError("executor options require repo_root")
        if self.log_dir is None:
            raise ValueError("executor options require log_dir")
        _require_non_empty("job_name", self.job_name or "")
        if self.chunk_size <= 0:
            raise ValueError("executor chunk_size must be positive")
        return self


@dataclass(frozen=True)
class SubmissionRequest:
    """Prepared command submission data for one stage-plan subset."""

    command_sets: CommandSets
    submitted_commands: Sequence[Command]

    def validate(self, n_tasks: int) -> "SubmissionRequest":
        """Validate command alignment against ``n_tasks`` and return ``self``."""

        command_sets = normalized_command_sets(self.command_sets)
        if n_tasks > 0 and not command_sets:
            raise ValueError("submission request requires at least one command profile")
        for profile, commands in command_sets.items():
            _require_non_empty("profile", profile)
            if len(commands) != n_tasks:
                raise ValueError(
                    f"command set {profile!r} has {len(commands)} commands for {n_tasks} tasks"
                )
        submitted = normalized_commands(self.submitted_commands, "submitted_commands")
        if len(submitted) != n_tasks:
            raise ValueError(f"submitted_commands has {len(submitted)} commands for {n_tasks} tasks")
        return self

    def command_sets_dict(self) -> dict[str, list[list[str]]]:
        """Return normalized command sets accepted by legacy launchers."""

        return normalized_command_sets(self.command_sets)

    def submitted_command_rows(self) -> list[list[str]]:
        """Return normalized provenance commands."""

        return normalized_commands(self.submitted_commands, "submitted_commands")


@dataclass(frozen=True)
class LauncherExecutor:
    """Adapter around an existing ``submit_command_sets``-style launcher."""

    submit_command_sets: LauncherSubmitter
    options: ExecutorOptions
    claim_paths_for_statuses: ClaimPathResolver | None = None

    def submit(
        self,
        plan: StagePlan,
        tasks: Sequence[TaskSpec],
        request: SubmissionRequest,
    ) -> tuple[ExecutionRecord, ...]:
        """Submit ``tasks`` through the configured launcher and return records."""

        plan.validate()
        self.options.validate()
        selected_tasks = tuple(tasks)
        _validate_task_subset(plan, selected_tasks)
        request.validate(len(selected_tasks))
        if not selected_tasks:
            return ()

        command_sets = request.command_sets_dict()
        row_status_paths = _row_status_paths(selected_tasks)
        job_ids = self.submit_command_sets(
            command_sets,
            args=self.options.args,
            backend=self.options.backend,
            repo_root=Path(str(self.options.repo_root)),
            log_dir=Path(str(self.options.log_dir)),
            job_name=str(self.options.job_name),
            smoke=self.options.smoke,
            chunk_size=self.options.chunk_size,
            allow_partial_failures=self.options.allow_partial_failures,
            row_status_paths=row_status_paths,
            chunk_status_dir=self.options.chunk_status_dir,
            claim_rows=self.options.claim_rows,
        )
        claim_paths = (
            self._claim_paths_for_statuses(row_status_paths)
            if _uses_row_claims(command_sets, self.options.claim_rows)
            else None
        )
        return execution_records_from_submission(
            tasks=selected_tasks,
            backend=self.options.backend,
            job_ids=job_ids,
            submitted_commands=request.submitted_command_rows(),
            claim_paths=claim_paths,
        )

    def _claim_paths_for_statuses(
        self,
        paths: Sequence[str | Path | None] | None,
    ) -> Sequence[str | Path | None] | None:
        resolver = self.claim_paths_for_statuses or _default_claim_paths_for_statuses
        return resolver(paths)


@dataclass(frozen=True)
class LocalExecutor(LauncherExecutor):
    """Launcher adapter for local execution."""

    def __post_init__(self) -> None:
        if self.options.backend != "local":
            raise ValueError("LocalExecutor requires options.backend == 'local'")


@dataclass(frozen=True)
class SubmititExecutor(LauncherExecutor):
    """Launcher adapter for Submitit execution."""

    def __post_init__(self) -> None:
        if self.options.backend != "submitit":
            raise ValueError("SubmititExecutor requires options.backend == 'submitit'")


def normalized_command_sets(command_sets: CommandSets) -> dict[str, list[list[str]]]:
    """Return command sets as mutable lists with validated command rows."""

    normalized: dict[str, list[list[str]]] = {}
    for profile, commands in command_sets.items():
        normalized[str(profile)] = normalized_commands(commands, f"command_sets[{profile!r}]")
    return normalized


def normalized_commands(commands: Sequence[Command], name: str) -> list[list[str]]:
    """Return command rows as ``list[list[str]]`` after shape validation."""

    if isinstance(commands, str):
        raise ValueError(f"{name} must be a sequence of commands, not a string")
    normalized: list[list[str]] = []
    for index, command in enumerate(commands):
        if isinstance(command, str):
            raise ValueError(f"{name}[{index}] must be a sequence, not a string")
        row = [str(part) for part in command]
        if not row:
            raise ValueError(f"{name}[{index}] must be non-empty")
        _require_non_empty_sequence(f"{name}[{index}]", row)
        normalized.append(row)
    return normalized


def _validate_task_subset(plan: StagePlan, tasks: Sequence[TaskSpec]) -> None:
    plan_task_ids = {task.task_id for task in plan.tasks}
    seen_task_ids: set[str] = set()
    for task in tasks:
        task.validate()
        if task.task_id not in plan_task_ids:
            raise ValueError(f"task {task.task_id!r} is not part of plan {plan.stage}/{plan.attempt_id}")
        if task.stage != plan.stage:
            raise ValueError(f"task {task.task_id!r} stage does not match plan stage")
        if task.attempt_id != plan.attempt_id:
            raise ValueError(f"task {task.task_id!r} attempt_id does not match plan attempt_id")
        if task.task_id in seen_task_ids:
            raise ValueError(f"duplicate selected task_id: {task.task_id!r}")
        seen_task_ids.add(task.task_id)


def _row_status_paths(tasks: Sequence[TaskSpec]) -> tuple[str | None, ...]:
    return tuple(task.logs[0] if task.logs else None for task in tasks)


def _uses_row_claims(command_sets: Mapping[str, Sequence[Command]], claim_rows: bool) -> bool:
    return len(command_sets) > 1 or claim_rows


def _default_claim_paths_for_statuses(
    paths: Sequence[str | Path | None] | None,
) -> list[Path | None] | None:
    if paths is None:
        return None
    return [None if path is None else Path(path).with_name("launcher_claim.json") for path in paths]


def _require_non_empty(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_empty_sequence(name: str, values: Sequence[str]) -> None:
    empty = [index for index, value in enumerate(values) if not str(value).strip()]
    if empty:
        raise ValueError(f"{name} contains empty entries at indexes: {empty}")
