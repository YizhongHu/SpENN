"""Executor interfaces and launcher adapters."""

from __future__ import annotations

import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .execution import ExecutionRecord, execution_records_from_submission
from .jsonio import write_json
from .specs import StagePlan, TaskSpec
from .task_state import (
    _deadline_guard_reached,
    allocation_deadline_unix,
    claim_paths_for_statuses as _default_claim_paths_for_statuses,
    claim_row_for_pass,
    next_attempt_dir,
    pass_claim_path,
)

Command = Sequence[str]
CommandSets = Mapping[str, Sequence[Command]]
LauncherSubmitter = Callable[..., Sequence[str]]
ClaimPathResolver = Callable[[Sequence[str | Path | None] | None], Sequence[str | Path | None] | None]


class _CompletionError(RuntimeError):
    """Declared completion remained unsatisfied after a successful command."""


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


@dataclass(frozen=True)
class AllocationPoolExecutor:
    """Execute tasks dynamically across workers in an existing allocation.

    This executor never submits scheduler jobs. Each worker owns one configured
    accelerator-visibility value, pulls tasks from a shared queue, and runs the
    task's command verbatim as an argument vector. Pass-scoped claims prevent
    duplicate execution, while completion predicates let a later pass retry
    only unfinished tasks.

    Parameters
    ----------
    pass_id : str
        Unique identifier for this pass over the plan. Retries use a new id.
    n_workers : int
        Number of long-lived worker threads in the current allocation.
    visibility_variable : str
        Per-worker environment variable, such as ``CUDA_VISIBLE_DEVICES`` or
        ``ZE_AFFINITY_MASK``.
    visibility_values : Sequence[str]
        One visibility value for each worker, in worker-index order.
    run_root : str or Path, optional
        Shared root for claims and execution receipts. Defaults to the plan's
        ``results_root``.
    working_directory : str or Path, optional
        Working directory inherited by every task command.
    deadline : str or float, optional
        Explicit allocation deadline accepted by
        :func:`allocation_deadline_unix`.
    deadline_env_var : str, optional
        Facility-provided deadline variable to resolve through the shared
        deadline helper.
    deadline_guard_min : int, default=1
        Minutes before the deadline at which workers stop claiming new tasks.
        Already-running commands are allowed to finish. Zero disables the
        guard, matching :func:`_deadline_guard_reached`.
    environment : Mapping[str, str], optional
        Extra environment entries inherited by every task. The per-worker
        visibility binding wins if the same name appears here.
    allocation_id : str, optional
        Allocation identifier written to execution records. Scheduler
        environment variables are used when omitted.
    """

    pass_id: str
    n_workers: int
    visibility_variable: str
    visibility_values: Sequence[str]
    run_root: str | Path | None = None
    working_directory: str | Path | None = None
    deadline: str | float | None = None
    deadline_env_var: str | None = None
    deadline_guard_min: int = 1
    environment: Mapping[str, str] = field(default_factory=dict)
    allocation_id: str | None = None

    def submit(
        self,
        plan: StagePlan,
        tasks: Sequence[TaskSpec],
        request: SubmissionRequest,
    ) -> tuple[ExecutionRecord, ...]:
        """Run the selected tasks and return records for claimed executions.

        Completed tasks, claims already owned by another worker, and tasks left
        unclaimed after the deadline guard are intentionally absent from the
        returned records.
        """

        plan.validate()
        selected_tasks = tuple(tasks)
        _validate_task_subset(plan, selected_tasks)
        request.validate(len(selected_tasks))
        self._validate()
        if not selected_tasks:
            return ()

        run_root = Path(self.run_root) if self.run_root is not None else Path(plan.results_root)
        base_environment = os.environ.copy()
        base_environment.update({str(key): str(value) for key, value in self.environment.items()})
        deadline_unix = allocation_deadline_unix(
            self.deadline,
            env_var=self.deadline_env_var,
            environ=base_environment,
        )
        allocation_id = self._allocation_id()

        pending: queue.Queue[tuple[int, TaskSpec]] = queue.Queue()
        for task_index, task in enumerate(selected_tasks):
            pending.put((task_index, task))

        records: dict[str, ExecutionRecord] = {}
        errors: list[Exception] = []
        result_lock = threading.Lock()
        stop_claiming = threading.Event()

        def worker(worker_index: int, visibility_value: str) -> None:
            while not stop_claiming.is_set():
                if _deadline_guard_reached(deadline_unix, self.deadline_guard_min):
                    stop_claiming.set()
                    return
                try:
                    task_index, task = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    if task.completion.is_complete():
                        continue
                    if stop_claiming.is_set() or _deadline_guard_reached(
                        deadline_unix, self.deadline_guard_min
                    ):
                        stop_claiming.set()
                        return
                    claim_path = pass_claim_path(run_root, self.pass_id, task.task_id)
                    if not claim_row_for_pass(
                        run_root,
                        self.pass_id,
                        task.task_id,
                        {
                            "allocation_id": allocation_id,
                            "worker_index": worker_index,
                            "visibility_variable": self.visibility_variable,
                            "visibility_value": visibility_value,
                        },
                    ):
                        continue
                    record = self._execute_task(
                        task=task,
                        run_root=run_root,
                        claim_path=claim_path,
                        task_index=task_index,
                        worker_index=worker_index,
                        visibility_value=visibility_value,
                        allocation_id=allocation_id,
                        base_environment=base_environment,
                    )
                    with result_lock:
                        records[task.task_id] = record
                except _CompletionError as exc:
                    with result_lock:
                        errors.append(exc)
                    continue
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)
                    stop_claiming.set()
                    return
                finally:
                    pending.task_done()

        workers = [
            threading.Thread(
                target=worker,
                args=(worker_index, str(self.visibility_values[worker_index])),
                name=f"allocation-pool-{worker_index}",
            )
            for worker_index in range(self.n_workers)
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()

        if errors:
            raise errors[0]
        return tuple(records[task.task_id] for task in selected_tasks if task.task_id in records)

    def _validate(self) -> None:
        """Validate allocation-pool configuration."""

        _require_non_empty("pass_id", self.pass_id)
        _require_non_empty("visibility_variable", self.visibility_variable)
        if self.n_workers <= 0:
            raise ValueError("allocation pool n_workers must be positive")
        if isinstance(self.visibility_values, str):
            raise ValueError("allocation pool visibility_values must be a sequence, not a string")
        if len(self.visibility_values) != self.n_workers:
            raise ValueError(
                "allocation pool visibility_values length must equal n_workers: "
                f"{len(self.visibility_values)} != {self.n_workers}"
            )
        _require_non_empty_sequence(
            "allocation pool visibility_values",
            tuple(str(value) for value in self.visibility_values),
        )
        if self.deadline_guard_min < 0:
            raise ValueError("allocation pool deadline_guard_min must be non-negative")
        if self.deadline_env_var is not None:
            _require_non_empty("deadline_env_var", self.deadline_env_var)
        if self.allocation_id is not None:
            _require_non_empty("allocation_id", self.allocation_id)

    def _allocation_id(self) -> str:
        """Return explicit or local allocation identity."""

        if self.allocation_id is not None:
            return str(self.allocation_id)
        return "allocation-pool-local"

    def _execute_task(
        self,
        *,
        task: TaskSpec,
        run_root: Path,
        claim_path: Path,
        task_index: int,
        worker_index: int,
        visibility_value: str,
        allocation_id: str,
        base_environment: Mapping[str, str],
    ) -> ExecutionRecord:
        """Run one claimed task and write immutable attempt evidence."""

        row_receipt_dir = run_root / "_allocation_pool" / claim_path.name
        attempt_dir = next_attempt_dir(row_receipt_dir)
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        attempt_status_path = attempt_dir / "status.json"
        launcher_status_path = Path(task.logs[0]) if task.logs else None
        environment = dict(base_environment)
        environment[self.visibility_variable] = visibility_value
        working_directory = str(self.working_directory) if self.working_directory is not None else None
        command_text = shlex.join(task.command)

        if launcher_status_path is not None:
            write_json(
                launcher_status_path,
                {
                    "status": "running",
                    "chunk_index": task_index,
                    "command": command_text,
                    "claim_label": self.pass_id,
                },
            )

        started_at = time.time()
        launch_error: str | None = None
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            try:
                completed = subprocess.run(
                    list(task.command),
                    cwd=working_directory,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
                returncode: int | None = int(completed.returncode)
            except OSError as exc:
                launch_error = repr(exc)
                stderr.write(launch_error + "\n")
                returncode = None
        completion_error: str | None = None
        if (
            returncode == 0
            and task.completion.policy != "none"
            and not task.completion.is_complete()
        ):
            completion_error = (
                f"command exited 0 but completion predicate {task.completion.policy!r} "
                f"was not satisfied for task {task.task_id!r}: "
                f"{task.completion.to_dict()!r}"
            )

        ended_at = time.time()

        row_status = "success" if returncode == 0 and completion_error is None else "failed"
        attempt_status = {
            "status": row_status,
            "task_id": task.task_id,
            "run_id": task.run_id,
            "stage": task.stage,
            "attempt_id": task.attempt_id,
            "backend": "allocation_pool",
            "launcher_job_id": allocation_id,
            "pass_id": self.pass_id,
            "worker_index": worker_index,
            "visibility_variable": self.visibility_variable,
            "visibility_value": visibility_value,
            "command": list(task.command),
            "command_text": command_text,
            "returncode": returncode,
            "attempt_dir": str(attempt_dir),
            "claim_path": str(claim_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "started_at_unix": started_at,
            "ended_at_unix": ended_at,
            "elapsed_sec": ended_at - started_at,
        }
        if launch_error is not None:
            attempt_status["error"] = launch_error
        if completion_error is not None:
            attempt_status["completion_error"] = completion_error
        write_json(attempt_status_path, attempt_status)

        if launcher_status_path is not None:
            launcher_status: dict[str, Any] = {
                "status": row_status,
                "chunk_index": task_index,
                "returncode": returncode,
                "command": command_text,
            }
            if launch_error is not None:
                launcher_status["error"] = launch_error
            if completion_error is not None:
                launcher_status["completion_error"] = completion_error
            write_json(launcher_status_path, launcher_status)

        if completion_error is not None:
            raise _CompletionError(completion_error)
        status_path = launcher_status_path or attempt_status_path

        return ExecutionRecord(
            task_id=task.task_id,
            run_id=task.run_id,
            stage=task.stage,
            attempt_id=task.attempt_id,
            backend="allocation_pool",
            launcher_job_id=allocation_id,
            submitted_command=tuple(task.command),
            status_path=str(status_path),
            claim_path=str(claim_path),
            metadata={
                "pass_id": self.pass_id,
                "worker_index": worker_index,
                "visibility_variable": self.visibility_variable,
                "visibility_value": visibility_value,
                "returncode": returncode,
                "attempt_dir": str(attempt_dir),
                "started_at_unix": started_at,
                "ended_at_unix": ended_at,
            },
        ).validate()


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


def _require_non_empty(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_empty_sequence(name: str, values: Sequence[str]) -> None:
    empty = [index for index, value in enumerate(values) if not str(value).strip()]
    if empty:
        raise ValueError(f"{name} contains empty entries at indexes: {empty}")
