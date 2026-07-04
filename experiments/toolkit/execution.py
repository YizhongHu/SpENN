"""Execution records shared by replaceable experiment backends."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .jsonio import to_jsonable, write_jsonl
from .specs import TaskSpec

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
    records: list[ExecutionRecord] = []
    for index, (task, job_id) in enumerate(zip(tasks, job_ids, strict=True)):
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
            )
        )
    return tuple(records)


def write_execution_records(directory: str | Path, records: Sequence[ExecutionRecord]) -> Path:
    """Write execution records next to a stage plan."""

    path = Path(directory) / EXECUTION_JSONL
    write_jsonl(path, (record.to_dict() for record in records))
    return path
