"""Evaluation component timing callback."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from .base import Callback, Event, _attach_event_metrics, _sync_device


class EvaluationComponentTiming(Callback):
    """Measure per-component durations within each evaluation task.

    ``Evaluator._evaluate_task`` emits ``generator_start``/``generator_end``,
    ``calculator_start``/``calculator_end``, and ``summary_start``/
    ``summary_end`` around each component it runs. Durations accumulate per
    task and are logged as a single ``eval/perf/<task_name>`` record at
    ``task_end`` (or ``task_failed``, flushing whatever was measured), one key
    per component observed in that task: ``generator_time_sec``,
    ``calculator/<name>_time_sec``, and ``summary/<name>_time_sec``.

    Parameters
    ----------
    triggers : iterable of str, optional
        Event names that should trigger this callback.
    cuda_synchronize : bool, optional
        Synchronize CUDA at component boundaries for accurate device timing.
    clock : callable, optional
        Monotonic clock override for deterministic tests.
    """

    def __init__(
        self,
        triggers: Iterable[str] = (
            "generator_start",
            "generator_end",
            "calculator_start",
            "calculator_end",
            "summary_start",
            "summary_end",
            "task_end",
            "task_failed",
        ),
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._starts: dict[tuple[str, str], float] = {}
        self._durations: dict[str, dict[str, float]] = {}

    def on_generator_start(self, event: Event) -> None:
        """Record one generator start time."""

        self._record_start(self._event_key(event, "generator"))

    def on_generator_end(self, event: Event) -> None:
        """Accumulate one generator duration for the enclosing task."""

        self._record_end(self._event_key(event, "generator"))

    def on_calculator_start(self, event: Event) -> None:
        """Record one calculator start time."""

        self._record_start(self._event_key(event, "calculator"))

    def on_calculator_end(self, event: Event) -> None:
        """Accumulate one calculator duration for the enclosing task."""

        self._record_end(self._event_key(event, "calculator"))

    def on_summary_start(self, event: Event) -> None:
        """Record one summary start time."""

        self._record_start(self._event_key(event, "summary"))

    def on_summary_end(self, event: Event) -> None:
        """Accumulate one summary duration for the enclosing task."""

        self._record_end(self._event_key(event, "summary"))

    def on_task_end(self, event: Event) -> None:
        """Log all component durations recorded for the finished task."""

        self._flush(event)

    def on_task_failed(self, event: Event) -> None:
        """Log the component durations measured before the task failed."""

        self._flush(event)

    def _record_start(self, key: tuple[str, str]) -> None:
        """Record one component start time under its ``(task, metric)`` key."""

        _sync_device(self.cuda_synchronize)
        self._starts[key] = self.clock()

    def _record_end(self, key: tuple[str, str]) -> None:
        """Accumulate one component duration under its task."""

        start = self._starts.pop(key, None)
        if start is None:
            return
        _sync_device(self.cuda_synchronize)
        task_name, metric_key = key
        durations = self._durations.setdefault(task_name, {})
        durations[metric_key] = durations.get(metric_key, 0.0) + (self.clock() - start)

    def _flush(self, event: Event) -> None:
        """Log the finished task's component durations as one record."""

        task_name = self._task_name(event)
        # Unmatched starts from an aborted component must not leak across tasks.
        self._starts = {key: value for key, value in self._starts.items() if key[0] != task_name}
        metrics = self._durations.pop(task_name, None)
        if not metrics:
            return
        step = 0 if event.step is None else int(event.step)
        namespace = f"eval/perf/{task_name}"
        event.context.log(metrics, step=step, namespace=namespace)
        _attach_event_metrics(event, namespace, metrics)

    @staticmethod
    def _event_key(event: Event, component_type: str) -> tuple[str, str]:
        """Return the ``(task_name, metric_key)`` identity of one component event."""

        task_name = event.payload.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("evaluation component timing events require a non-empty 'task_name' payload entry")
        if component_type == "generator":
            return task_name, "generator_time_sec"
        component_name = event.payload.get("component_name")
        if not isinstance(component_name, str) or not component_name.strip():
            raise ValueError(
                f"{component_type} timing events require a non-empty 'component_name' payload entry"
            )
        return task_name, f"{component_type}/{component_name}_time_sec"

    @staticmethod
    def _task_name(event: Event) -> str:
        """Return the task name carried by a task boundary event."""

        name = event.payload.get("task_name")
        if name is None and isinstance(event.payload.get("task_result"), dict):
            name = event.payload["task_result"].get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("evaluation component timing task events require a non-empty task name payload entry")
        return name


__all__ = ["EvaluationComponentTiming"]
