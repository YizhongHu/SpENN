"""Durable task and stage plan specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .jsonio import read_json, read_jsonl, to_jsonable, write_json, write_jsonl
from .resources import ResourceSpec

SCHEMA_VERSION = "experiment-toolkit/v1"
STAGE_MANIFEST = "stage_manifest.json"
TASKS_JSONL = "tasks.jsonl"
COMPLETION_POLICIES = frozenset(
    {
        "none",
        "file_exists",
        "checkpoint_exists",
        "status_completed",
        "status_completed_with_checkpoint",
    }
)


@dataclass(frozen=True)
class CompletionSpec:
    """Completion predicate metadata for executor skip decisions."""

    policy: str
    status_path: str | None = None
    checkpoint_path: str | None = None
    output_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "CompletionSpec":
        """Validate the completion predicate metadata and return ``self``."""

        if self.policy not in COMPLETION_POLICIES:
            raise ValueError(f"unknown completion policy: {self.policy!r}")
        _require_path_sequence("completion.output_paths", self.output_paths)
        if self.policy == "file_exists" and not self.output_paths:
            raise ValueError("completion policy 'file_exists' requires output_paths")
        if self.policy == "checkpoint_exists" and not self.checkpoint_path:
            raise ValueError("completion policy 'checkpoint_exists' requires checkpoint_path")
        if self.policy == "status_completed" and not self.status_path:
            raise ValueError("completion policy 'status_completed' requires status_path")
        if self.policy == "status_completed_with_checkpoint":
            if not self.status_path:
                raise ValueError("completion policy 'status_completed_with_checkpoint' requires status_path")
            if not self.checkpoint_path:
                raise ValueError(
                    "completion policy 'status_completed_with_checkpoint' requires checkpoint_path"
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "policy": self.policy,
            "status_path": self.status_path,
            "checkpoint_path": self.checkpoint_path,
            "output_paths": list(self.output_paths),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CompletionSpec":
        """Build a completion spec from serialized data."""

        data = data or {}
        return cls(
            policy=str(data.get("policy") or "none"),
            status_path=_optional_str(data.get("status_path")),
            checkpoint_path=_optional_str(data.get("checkpoint_path")),
            output_paths=_string_tuple(data.get("output_paths", ()), "completion.output_paths"),
            metadata=_mapping(data.get("metadata")),
        ).validate()

    def is_complete(self) -> bool:
        """Return whether this completion predicate is currently satisfied."""

        if self.policy == "file_exists":
            return bool(self.output_paths) and all(Path(path).exists() for path in self.output_paths)
        if self.policy == "checkpoint_exists":
            return bool(self.checkpoint_path) and Path(str(self.checkpoint_path)).exists()
        if self.policy == "status_completed":
            return _status_value(self.status_path) == "completed"
        if self.policy == "status_completed_with_checkpoint":
            return _status_value(self.status_path) == "completed" and bool(
                self.checkpoint_path and Path(str(self.checkpoint_path)).exists()
            )
        return False


@dataclass(frozen=True)
class TaskSpec:
    """Logical task consumed by an execution backend."""

    task_id: str
    stage: str
    attempt_id: str
    run_id: str
    command: tuple[str, ...]
    result_dir: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=lambda: ResourceSpec(profile="cpu", device="cpu"))
    dependencies: tuple[str, ...] = ()
    completion: CompletionSpec = field(default_factory=lambda: CompletionSpec(policy="none"))
    resume: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "TaskSpec":
        """Validate the task contract and return ``self``."""

        _require_non_empty("task_id", self.task_id)
        _require_non_empty("stage", self.stage)
        _require_non_empty("attempt_id", self.attempt_id)
        _require_non_empty("run_id", self.run_id)
        _require_non_empty("result_dir", self.result_dir)
        _require_path_sequence("inputs", self.inputs)
        _require_path_sequence("outputs", self.outputs)
        _require_path_sequence("logs", self.logs)
        _require_path_sequence("dependencies", self.dependencies)
        if not self.command:
            raise ValueError(f"task {self.task_id!r} command must be non-empty")
        _require_non_empty_sequence("command", self.command)
        expected_task_id = task_id_from_parts(
            stage=self.stage,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
        )
        if self.task_id != expected_task_id:
            raise ValueError(
                f"task_id {self.task_id!r} does not match deterministic id {expected_task_id!r}"
            )
        self.resources.validate()
        self.completion.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "command": list(self.command),
            "result_dir": self.result_dir,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "logs": list(self.logs),
            "params": to_jsonable(dict(self.params)),
            "resources": self.resources.to_dict(),
            "dependencies": list(self.dependencies),
            "completion": self.completion.to_dict(),
            "resume": to_jsonable(dict(self.resume)),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        """Build a task spec from serialized data."""

        return cls(
            task_id=_required_str(data, "task_id"),
            stage=_required_str(data, "stage"),
            attempt_id=_required_str(data, "attempt_id"),
            run_id=_required_str(data, "run_id"),
            command=_string_tuple(data.get("command", ()), "command"),
            result_dir=str(data.get("result_dir") or ""),
            inputs=_string_tuple(data.get("inputs", ()), "inputs"),
            outputs=_string_tuple(data.get("outputs", ()), "outputs"),
            logs=_string_tuple(data.get("logs", ()), "logs"),
            params=_mapping(data.get("params")),
            resources=ResourceSpec.from_dict(data.get("resources")),
            dependencies=_string_tuple(data.get("dependencies", ()), "dependencies"),
            completion=CompletionSpec.from_dict(data.get("completion")),
            resume=_mapping(data.get("resume")),
            metadata=_mapping(data.get("metadata")),
        ).validate()


@dataclass(frozen=True)
class StagePlan:
    """Immutable stage-level task table."""

    study: str
    stage: str
    attempt_id: str
    results_root: str
    tasks: tuple[TaskSpec, ...]
    source_attempts: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    timezone: str | None = None
    smoke: bool = False
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        """Return the task count."""

        return len(self.tasks)

    def validate(self) -> "StagePlan":
        """Validate the stage-plan contract and return ``self``."""

        _require_non_empty("study", self.study)
        _require_non_empty("stage", self.stage)
        _require_non_empty("attempt_id", self.attempt_id)
        _require_non_empty("results_root", self.results_root)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"stage plan schema_version {self.schema_version!r} does not match {SCHEMA_VERSION!r}"
            )
        seen_task_ids: set[str] = set()
        for task in self.tasks:
            task.validate()
            if task.stage != self.stage:
                raise ValueError(
                    f"task {task.task_id!r} stage {task.stage!r} does not match plan stage {self.stage!r}"
                )
            if task.attempt_id != self.attempt_id:
                raise ValueError(
                    f"task {task.task_id!r} attempt_id {task.attempt_id!r} "
                    f"does not match plan attempt_id {self.attempt_id!r}"
                )
            if task.task_id in seen_task_ids:
                raise ValueError(f"duplicate task_id in stage plan: {task.task_id!r}")
            seen_task_ids.add(task.task_id)
        return self

    def to_manifest(self) -> dict[str, Any]:
        """Return the compact stage manifest."""

        return {
            "schema_version": self.schema_version,
            "study": self.study,
            "stage": self.stage,
            "attempt_id": self.attempt_id,
            "results_root": self.results_root,
            "source_attempts": to_jsonable(dict(self.source_attempts)),
            "created_at": self.created_at,
            "timezone": self.timezone,
            "smoke": self.smoke,
            "tasks_path": TASKS_JSONL,
            "n_tasks": self.n_tasks,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def write(self, directory: str | Path) -> Path:
        """Write the stage plan and return its directory."""

        self.validate()
        directory = Path(directory)
        write_json(directory / STAGE_MANIFEST, self.to_manifest())
        write_jsonl(directory / TASKS_JSONL, (task.to_dict() for task in self.tasks))
        return directory

    @classmethod
    def read(cls, directory: str | Path) -> "StagePlan":
        """Read a stage plan from ``directory``."""

        directory = Path(directory)
        manifest = read_json(directory / STAGE_MANIFEST)
        tasks_path = _required_str(manifest, "tasks_path")
        tasks = tuple(TaskSpec.from_dict(row) for row in read_jsonl(directory / tasks_path))
        expected_n_tasks = _required_int(manifest, "n_tasks")
        plan = cls(
            study=_required_str(manifest, "study"),
            stage=_required_str(manifest, "stage"),
            attempt_id=_required_str(manifest, "attempt_id"),
            results_root=_required_str(manifest, "results_root"),
            source_attempts=_mapping(manifest.get("source_attempts")),
            created_at=_optional_str(manifest.get("created_at")),
            timezone=_optional_str(manifest.get("timezone")),
            smoke=bool(manifest.get("smoke", False)),
            schema_version=_required_str(manifest, "schema_version"),
            metadata=_mapping(manifest.get("metadata")),
            tasks=tasks,
        )
        if expected_n_tasks != plan.n_tasks:
            raise ValueError(
                f"stage manifest n_tasks={expected_n_tasks} does not match {plan.n_tasks} task rows"
            )
        return plan.validate()


def _status_value(path: str | None) -> str | None:
    if not path:
        return None
    try:
        data = read_json(path)
    except (FileNotFoundError, ValueError):
        return None
    status = data.get("status")
    return str(status) if status else None


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required field: {key}")
    return str(value)


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return int(value)


def _require_non_empty(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_empty_sequence(name: str, values: Sequence[str]) -> None:
    empty = [index for index, value in enumerate(values) if not str(value).strip()]
    if empty:
        raise ValueError(f"{name} contains empty entries at indexes: {empty}")


def _require_path_sequence(name: str, values: Sequence[str]) -> None:
    _require_non_empty_sequence(name, values)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence, not a string")
    try:
        return tuple(str(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc


def task_id_from_parts(*, stage: str, run_id: str, attempt_id: str) -> str:
    """Return the deterministic task id used by command-backed stage tasks."""

    _require_non_empty("stage", stage)
    _require_non_empty("run_id", run_id)
    _require_non_empty("attempt_id", attempt_id)
    return f"{stage}:{run_id}:{attempt_id}"


def tasks_from_commands(
    *,
    stage: str,
    attempt_id: str,
    jobs: Sequence[Mapping[str, Any]],
    commands: Sequence[Sequence[str]],
    result_dirs: Sequence[str | Path],
    row_status_paths: Sequence[str | Path],
    resources: ResourceSpec,
    completion_policy: str,
    checkpoint_paths: Sequence[str | Path | None] | None = None,
    source_attempts: Mapping[str, Any] | None = None,
) -> tuple[TaskSpec, ...]:
    """Build task specs from stage-local job records and commands."""

    checkpoints = list(checkpoint_paths or [None] * len(jobs))
    tasks: list[TaskSpec] = []
    for index, job in enumerate(jobs):
        run_id = str(job.get("run_id") or job.get("final_run_id") or index)
        result_dir = str(result_dirs[index])
        status_path = str(row_status_paths[index])
        checkpoint_path = None if checkpoints[index] is None else str(checkpoints[index])
        task_id = task_id_from_parts(stage=stage, run_id=run_id, attempt_id=str(attempt_id))
        tasks.append(
            TaskSpec(
                task_id=task_id,
                stage=stage,
                attempt_id=str(attempt_id),
                run_id=run_id,
                command=tuple(str(part) for part in commands[index]),
                result_dir=result_dir,
                inputs=tuple(str(path) for path in _input_paths(job)),
                outputs=tuple(path for path in (checkpoint_path, result_dir) if path),
                logs=(status_path,),
                params={"source_attempts": dict(source_attempts or {})},
                resources=resources,
                completion=CompletionSpec(
                    policy=completion_policy,
                    status_path=status_path,
                    checkpoint_path=checkpoint_path,
                ),
                metadata={"job": dict(job)},
            )
        )
    return tuple(tasks)


def _input_paths(job: Mapping[str, Any]) -> list[str]:
    inputs = []
    for key in ("command", "train_attempt_dir", "validation_attempt_dir", "source_scan_run_id"):
        value = job.get(key)
        if isinstance(value, str) and value:
            inputs.append(value)
    return inputs
