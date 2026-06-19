"""Evaluation event payload helpers."""

from __future__ import annotations

from spenn.evaluation.results import EvaluationFailure, TaskResult
from spenn.evaluation.task import EvaluationTask


def task_payload(task: EvaluationTask, *, output_dir: str | None = None) -> dict[str, object]:
    """Return the standard payload for task lifecycle events."""

    payload: dict[str, object] = {
        "task_name": task.name,
        "task_namespace": task.namespace,
    }
    if output_dir is not None:
        payload["output_dir"] = output_dir
    return payload


def component_failure_payload(
    *,
    task: EvaluationTask,
    component_name: str | None,
    failure: EvaluationFailure,
    output_dir: str | None = None,
) -> dict[str, object]:
    """Return a standard component-failure event payload."""

    return {
        **task_payload(task, output_dir=output_dir),
        "component_name": component_name,
        "failure": failure.to_dict(),
    }


def task_result_payload(task_result: TaskResult) -> dict[str, object]:
    """Return the standard payload for task completion/failure events."""

    return {"task_result": task_result.to_payload()}


__all__ = ["component_failure_payload", "task_payload", "task_result_payload"]
