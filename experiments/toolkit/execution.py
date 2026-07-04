"""Execution records shared by replaceable experiment backends."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .jsonio import to_jsonable, write_jsonl
from .specs import TaskSpec, task_id_from_parts

EXECUTION_JSONL = "execution_records.jsonl"


@dataclass(frozen=True)
class ExecutionRecord:
    """Submission or local execution record for one logical task."""

    task_id: str
    run_id: str
    stage: str
    attempt_id: str
    backend: str
    launcher_job_id: str
    submitted_command: tuple[str, ...]
    status_path: str | None = None
    claim_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "ExecutionRecord":
        """Validate the execution-record contract and return ``self``."""

        _require_non_empty("task_id", self.task_id)
        _require_non_empty("run_id", self.run_id)
        _require_non_empty("stage", self.stage)
        _require_non_empty("attempt_id", self.attempt_id)
        _require_non_empty("backend", self.backend)
        _require_non_empty("launcher_job_id", self.launcher_job_id)
        if not self.submitted_command:
            raise ValueError(f"execution record {self.task_id!r} submitted_command must be non-empty")
        _require_non_empty_sequence("submitted_command", self.submitted_command)
        expected_task_id = task_id_from_parts(
            stage=self.stage,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
        )
        if self.task_id != expected_task_id:
            raise ValueError(
                f"execution record task_id {self.task_id!r} does not match deterministic id "
                f"{expected_task_id!r}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "attempt_id": self.attempt_id,
            "backend": self.backend,
            "launcher_job_id": self.launcher_job_id,
            "submitted_command": list(self.submitted_command),
            "submitted_command_text": shlex.join(self.submitted_command),
            "status_path": self.status_path,
            "claim_path": self.claim_path,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionRecord":
        """Build an execution record from serialized data."""

        return cls(
            task_id=_required_str(data, "task_id"),
            run_id=_required_str(data, "run_id"),
            stage=_required_str(data, "stage"),
            attempt_id=_required_str(data, "attempt_id"),
            backend=_required_str(data, "backend"),
            launcher_job_id=_required_str(data, "launcher_job_id"),
            submitted_command=_string_tuple(data.get("submitted_command", ()), "submitted_command"),
            status_path=_optional_str(data.get("status_path")),
            claim_path=_optional_str(data.get("claim_path")),
            metadata=dict(data.get("metadata")) if isinstance(data.get("metadata"), Mapping) else {},
        ).validate()


def execution_records_from_submission(
    *,
    tasks: Sequence[TaskSpec],
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
    claim_paths: Sequence[str | Path | None] | None = None,
) -> tuple[ExecutionRecord, ...]:
    """Return execution records aligned with submitted tasks."""

    claims = list(claim_paths or [None] * len(tasks))
    if len(job_ids) != len(tasks):
        raise ValueError(f"job_ids length {len(job_ids)} does not match tasks length {len(tasks)}")
    if len(submitted_commands) != len(tasks):
        raise ValueError(
            f"submitted_commands length {len(submitted_commands)} does not match tasks length {len(tasks)}"
        )
    if len(claims) != len(tasks):
        raise ValueError(f"claim_paths length {len(claims)} does not match tasks length {len(tasks)}")
    records: list[ExecutionRecord] = []
    for index, (task, job_id) in enumerate(zip(tasks, job_ids, strict=True)):
        task.validate()
        status_path = task.logs[0] if task.logs else None
        claim_path = None if claims[index] is None else str(claims[index])
        records.append(
            ExecutionRecord(
                task_id=task.task_id,
                run_id=task.run_id,
                stage=task.stage,
                attempt_id=task.attempt_id,
                backend=backend,
                launcher_job_id=str(job_id),
                submitted_command=tuple(str(part) for part in submitted_commands[index]),
                status_path=status_path,
                claim_path=claim_path,
            ).validate()
        )
    return tuple(records)


def write_execution_records(directory: str | Path, records: Sequence[ExecutionRecord]) -> Path:
    """Write execution records next to a stage plan."""

    path = Path(directory) / EXECUTION_JSONL
    write_jsonl(path, (record.validate().to_dict() for record in records))
    return path


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required field: {key}")
    return str(value)


def _require_non_empty(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be a non-empty string")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence, not a string")
    try:
        return tuple(str(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc


def _require_non_empty_sequence(name: str, values: Sequence[str]) -> None:
    empty = [index for index, value in enumerate(values) if not str(value).strip()]
    if empty:
        raise ValueError(f"{name} contains empty entries at indexes: {empty}")
