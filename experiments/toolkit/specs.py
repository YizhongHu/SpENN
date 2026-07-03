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


@dataclass(frozen=True)
class CompletionSpec:
    """Completion predicate metadata for executor skip decisions."""

    policy: str
    status_path: str | None = None
    checkpoint_path: str | None = None
    output_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
            output_paths=tuple(str(path) for path in data.get("output_paths", ()) or ()),
            metadata=_mapping(data.get("metadata")),
        )

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
            task_id=str(data["task_id"]),
            stage=str(data["stage"]),
            attempt_id=str(data["attempt_id"]),
            run_id=str(data["run_id"]),
            command=tuple(str(part) for part in data.get("command", ()) or ()),
            result_dir=str(data.get("result_dir") or ""),
            inputs=tuple(str(path) for path in data.get("inputs", ()) or ()),
            outputs=tuple(str(path) for path in data.get("outputs", ()) or ()),
            logs=tuple(str(path) for path in data.get("logs", ()) or ()),
            params=_mapping(data.get("params")),
            resources=ResourceSpec.from_dict(data.get("resources")),
            dependencies=tuple(str(task_id) for task_id in data.get("dependencies", ()) or ()),
            completion=CompletionSpec.from_dict(data.get("completion")),
            resume=_mapping(data.get("resume")),
            metadata=_mapping(data.get("metadata")),
        )


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

        directory = Path(directory)
        write_json(directory / STAGE_MANIFEST, self.to_manifest())
        write_jsonl(directory / TASKS_JSONL, (task.to_dict() for task in self.tasks))
        return directory

    @classmethod
    def read(cls, directory: str | Path) -> "StagePlan":
        """Read a stage plan from ``directory``."""

        directory = Path(directory)
        manifest = read_json(directory / STAGE_MANIFEST)
        tasks = tuple(TaskSpec.from_dict(row) for row in read_jsonl(directory / str(manifest["tasks_path"])))
        return cls(
            study=str(manifest["study"]),
            stage=str(manifest["stage"]),
            attempt_id=str(manifest["attempt_id"]),
            results_root=str(manifest["results_root"]),
            source_attempts=_mapping(manifest.get("source_attempts")),
            created_at=_optional_str(manifest.get("created_at")),
            timezone=_optional_str(manifest.get("timezone")),
            smoke=bool(manifest.get("smoke", False)),
            schema_version=str(manifest.get("schema_version") or SCHEMA_VERSION),
            metadata=_mapping(manifest.get("metadata")),
            tasks=tasks,
        )


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
        task_id = f"{stage}:{run_id}:{attempt_id}"
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
