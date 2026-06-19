"""Result objects for composable evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

JsonScalar: TypeAlias = bool | int | float | str | None
MetricScalar: TypeAlias = bool | int | float
EvaluationStatus: TypeAlias = Literal["success", "success_with_warnings", "failed"]
TaskStatus: TypeAlias = Literal["success", "partial_failed", "failed", "skipped"]
ComponentType: TypeAlias = Literal["generator", "calculator", "summary", "artifact", "evaluator"]


@dataclass(frozen=True)
class ArtifactRecord:
    """One evaluation artifact produced by a task or summary."""

    name: str
    kind: str
    path: str | Path
    enabled: bool = True
    expected: bool = True
    metadata: dict[str, JsonScalar] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe artifact mapping."""

        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class EvaluationFailure:
    """Structured failure captured while running an evaluation component."""

    task: str | None
    component: str | None
    component_type: ComponentType
    error_type: str
    message: str
    traceback: str | None = None

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return a JSON-safe failure mapping."""

        return {
            "task": self.task,
            "component": self.component,
            "component_type": self.component_type,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
        }


@dataclass(frozen=True)
class SummaryResult:
    """Metrics, artifacts, and optional records emitted by one summary."""

    metrics: dict[str, MetricScalar]
    artifacts: tuple[ArtifactRecord, ...] = ()
    records: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TaskResult:
    """Result for one evaluation task."""

    name: str
    namespace: str
    output_dir: str
    status: TaskStatus
    metrics: dict[str, MetricScalar]
    artifacts: tuple[ArtifactRecord, ...]
    failures: tuple[EvaluationFailure, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return compact event payload data."""

        return {
            "name": self.name,
            "namespace": self.namespace,
            "output_dir": self.output_dir,
            "status": self.status,
            "metrics": dict(self.metrics),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate result for an evaluator run."""

    status: EvaluationStatus
    metrics: dict[str, MetricScalar]
    task_results: tuple[TaskResult, ...]
    artifacts: tuple[ArtifactRecord, ...]
    failures: tuple[EvaluationFailure, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return compact event payload data."""

        return {
            "status": self.status,
            "metrics": dict(self.metrics),
            "tasks": [task.to_payload() for task in self.task_results],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "failures": [failure.to_dict() for failure in self.failures],
        }


__all__ = [
    "ArtifactRecord",
    "ComponentType",
    "EvaluationFailure",
    "EvaluationResult",
    "EvaluationStatus",
    "JsonScalar",
    "MetricScalar",
    "SummaryResult",
    "TaskResult",
    "TaskStatus",
]
