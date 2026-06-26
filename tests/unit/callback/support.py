"""Shared helpers for runtime-check callback unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spenn.callback import Event


class RecordingContext:
    """Minimal RunContext stand-in that captures ``log`` calls."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(
        self,
        metrics,
        *,
        step: int | None = None,
        namespace: str = "run",
        event: str | None = None,
    ) -> None:
        self.records.append(
            {"metrics": dict(metrics), "step": step, "namespace": namespace, "event": event}
        )

    def by_namespace(self, namespace: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record["namespace"] == namespace]

    def latest(self, namespace: str) -> dict[str, Any]:
        return self.by_namespace(namespace)[-1]["metrics"]


@dataclass
class FakeState:
    """Mutable stand-in for `TrainerState` carrying step artifacts."""

    step: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    optimizer: Any = None
    sampler: Any = None
    samples: Any = None
    batch: Any = None
    local_energy: Any = None
    loss: Any = None
    wavefunction_output: Any = None
    sampler_stats: dict[str, Any] = field(default_factory=dict)


def step_event(context: Any, state: Any, step: int | None = None) -> Event:
    """Build a ``step_end`` event for `state`."""

    resolved = state.step if step is None else step
    return Event(name="step_end", context=context, state=state, payload={"step": resolved})
