"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from spenn.artifacts import RunContext, RunResult
from spenn.callback import Callback, Event
from spenn.logging import Logger


class Runner:
    """Base runner with callback lifecycle dispatch."""

    def __init__(
        self,
        callbacks: Iterable[Callback] | None = None,
        loggers: Iterable[Logger] | None = None,
    ) -> None:
        self.callbacks = list(callbacks or [])
        self.loggers = list(loggers or [])
        self.callback_registry = self.build_callback_registry(self.callbacks)

    def build_callback_registry(self, callbacks: Iterable[Callback]) -> dict[str, list[Callback]]:
        """Group callbacks by subscribed event name."""

        registry: dict[str, list[Callback]] = {}
        for callback in callbacks:
            for trigger in callback.triggers:
                registry.setdefault(trigger, []).append(callback)
        return registry

    def emit(
        self,
        name: str,
        context: RunContext,
        *,
        state: object | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one lifecycle event to subscribed callbacks."""

        event = Event(name=name, context=context, state=state, payload={} if payload is None else payload)
        for callback in self.callback_registry.get(name, []):
            callback.handle(event)

    def run(self, context: RunContext) -> RunResult:
        """Execute a configured run."""

        raise NotImplementedError


class Scaffold(Runner):
    """Runner that validates generic run-management plumbing."""

    def run(self, context: RunContext) -> RunResult:
        """Execute the PR1 scaffold lifecycle."""

        self.emit("run_start", context)
        context.log({"scaffold_completed": True}, step=0, namespace="scaffold")
        self.emit("run_end", context)
        return RunResult(status="completed")


class Train(Runner):
    """Placeholder for future training runner configs."""

    def run(self, context: RunContext) -> RunResult:
        """Raise until training runner support is implemented."""

        raise NotImplementedError("spenn.runner.Train will be implemented in a later PR.")


class Load(Runner):
    """Placeholder for future load/evaluate runner configs."""

    def run(self, context: RunContext) -> RunResult:
        """Raise until load runner support is implemented."""

        raise NotImplementedError("spenn.runner.Load will be implemented in a later PR.")


__all__ = ["Load", "Runner", "Scaffold", "Train"]
