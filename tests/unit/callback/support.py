"""Shared helpers for runtime-check callback unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tpen.artifacts import RunContext
from tpen.callback import Event
from tpen.events import Occurrence
from tpen.sampling import SamplerStats
from tpen.training.events import TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState


class RecordingContext(RunContext):
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
    """Mutable stand-in for `TrainerState` carrying step artifacts.

    Only the callbacks still on the legacy string path use this. A callback
    migrated to `tpen.callback.StatefulCallback` declares
    ``state_type = TrainerState``, and its ``handle_occurrence_impl`` narrows
    the parameter to that class, so under typeguard a stand-in would be
    rejected at delivery. Those tests build a real `TrainerState` with
    `training_state`.
    """

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
    sampler_stats: SamplerStats | None = None


def make_sampler_stats(
    *,
    acceptance_rate: float = 0.5,
    n_walkers: int = 8,
    burn_in: int = 0,
    n_steps: int = 1,
    proposal_scale: float = 0.1,
    geometry: dict[str, float] | None = None,
    seed: int | None = None,
) -> SamplerStats:
    """Build a `SamplerStats` record with test-friendly defaults."""

    return SamplerStats(
        acceptance_rate=acceptance_rate,
        n_walkers=n_walkers,
        burn_in=burn_in,
        n_steps=n_steps,
        proposal_scale=proposal_scale,
        geometry={} if geometry is None else geometry,
        seed=seed,
    )


def step_event(context: Any, state: Any, step: int | None = None) -> Event:
    """Build a ``step_end`` event for `state`."""

    resolved = state.step if step is None else step
    return Event(
        name="step_end",
        context=context,
        state=state,
        payload={"step": resolved},
        step=resolved,
    )


def training_state(**fields: Any) -> TrainerState:
    """Build the training domain's real state object for typed delivery.

    Every field defaults, so a test names only what its callback reads. The
    trainer mutates one instance in place all run long, which is why the
    unnamed fields keep the constructor defaults a real run would show before
    its first assignment.
    """

    return TrainerState(**fields)


def iteration_completed(step: int, *, count: int = 1) -> Occurrence[TrainingIterationCompleted]:
    """Build the occurrence the trainer emits after a completed iteration.

    Parameters
    ----------
    step : int
        Durable zero-based trainer step the iteration ran at. This is the
        coordinate a migrated callback reads -- never ``state.step``.
    count : int, optional
        One-based run-local occurrence count. Deliberately independent of
        ``step``: the two coordinates diverge on a resumed run, and a callback
        that confused them would gate its cadence on the wrong one.
    """

    return Occurrence(
        event=TrainingIterationCompleted(iteration=TrainingIteration(step=step)),
        count=count,
    )


def deliver_completed_iteration(
    callback: Any,
    context: Any,
    state: TrainerState,
    *,
    step: int,
    count: int = 1,
) -> None:
    """Hand one completed-iteration occurrence straight to ``callback``.

    This bypasses `tpen.artifacts.RunContext._dispatch_occurrence`, so it tests
    what the callback *does* once delivered, not whether delivery happens.
    Delivery itself is covered in ``test_typed_health_delivery.py``, which
    drives the real dispatcher.
    """

    callback.handle_occurrence(iteration_completed(step, count=count), context, state)
