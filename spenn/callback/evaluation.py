"""Callbacks for evaluation task artifacts and failures."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spenn.callback.base import Callback, Event


class ArtifactIndex(Callback):
    """Maintain a compact index of evaluation task artifacts."""

    def __init__(
        self,
        triggers: Iterable[str] = ("task_end", "task_failed", "run_end"),
        *,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.path = None if path is None else Path(path)
        self._tasks: dict[str, dict[str, Any]] = {}

    def on_task_end(self, event: Event) -> None:
        """Record a successful task result."""

        self._record_task(event)

    def on_task_failed(self, event: Event) -> None:
        """Record a failed or partially failed task result."""

        self._record_task(event)

    def on_run_end(self, event: Event) -> None:
        """Flush the artifact index at the end of the run."""

        self._write(event.context)

    def _record_task(self, event: Event) -> None:
        payload = event.payload.get("task_result")
        if not isinstance(payload, dict):
            return
        namespace = str(payload.get("namespace", payload.get("name", "")))
        if not namespace:
            return
        self._tasks[namespace] = {
            "name": payload.get("name"),
            "namespace": namespace,
            "status": payload.get("status"),
            "required": payload.get("required"),
            "artifacts": payload.get("artifacts", []),
        }
        self._write(event.context)

    def _write(self, context) -> None:
        path = self._path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": _phase_from_tasks(self._tasks.values()),
            "tasks": list(self._tasks.values()),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _path(self, context) -> Path:
        if self.path is not None:
            return self.path
        return Path(context.run_dir) / "diagnostics" / "index.json"


class FailureLog(Callback):
    """Append structured evaluation failures to ``diagnostics/failures.jsonl``."""

    def __init__(
        self,
        triggers: Iterable[str] = (
            "generator_failed",
            "calculator_failed",
            "summary_failed",
            "artifact_failed",
            "task_failed",
        ),
        *,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.path = None if path is None else Path(path)

    def on_generator_failed(self, event: Event) -> None:
        """Record a generator failure."""

        self._record_failure(event)

    def on_calculator_failed(self, event: Event) -> None:
        """Record a calculator failure."""

        self._record_failure(event)

    def on_summary_failed(self, event: Event) -> None:
        """Record a summary failure."""

        self._record_failure(event)

    def on_artifact_failed(self, event: Event) -> None:
        """Record an artifact failure."""

        self._record_failure(event)

    def on_task_failed(self, event: Event) -> None:
        """Record task-level failures from the task result payload."""

        task_result = event.payload.get("task_result")
        if not isinstance(task_result, dict):
            return
        for failure in task_result.get("failures", []) or []:
            if isinstance(failure, dict):
                self._append(event.context, failure)

    def _record_failure(self, event: Event) -> None:
        failure = event.payload.get("failure")
        if isinstance(failure, dict):
            self._append(event.context, failure)

    def _append(self, context, failure: dict[str, Any]) -> None:
        path = self._path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")

    def _path(self, context) -> Path:
        if self.path is not None:
            return self.path
        return Path(context.run_dir) / "diagnostics" / "failures.jsonl"


def _phase_from_tasks(tasks) -> str | None:
    for task in tasks:
        namespace = str(task.get("namespace", ""))
        root = namespace.split("/", 1)[0]
        if root:
            return root
    return None


__all__ = ["ArtifactIndex", "FailureLog"]
