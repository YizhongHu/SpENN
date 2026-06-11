"""Base callback event and scheduling primitives."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from spenn.artifacts import RunContext


@dataclass
class Event:
    """Lifecycle event delivered to callbacks."""

    name: str
    context: RunContext
    state: object | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def step(self) -> int | None:
        """Return the event step when available."""

        if "step" in self.payload:
            return None if self.payload["step"] is None else int(self.payload["step"])
        value = getattr(self.state, "global_step", None)
        return None if value is None else int(value)


class Callback:
    """Base class for event-triggered run callbacks.

    Parameters
    ----------
    triggers : iterable of str
        Event names that should trigger this callback.
    every_n_steps : int or None, optional
        Optional periodic step filter.
    start_step : int, optional
        First eligible step for periodic callbacks.
    max_calls : int or None, optional
        Maximum number of callback invocations (counts actual executions).
    probability : float, optional
        Probability of running when otherwise scheduled. ``1.0`` always runs,
        ``0.0`` never runs. Applied after the trigger/``every_n_steps``/
        ``start_step`` checks.
    seed : int or None, optional
        Seed for the callback-local RNG used by `probability`. Using a local
        RNG keeps probabilistic scheduling reproducible without perturbing
        global PyTorch randomness.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        every_n_steps: int | None = None,
        start_step: int = 0,
        max_calls: int | None = None,
        probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        self.triggers = tuple(triggers)
        self.every_n_steps = every_n_steps
        self.start_step = int(start_step)
        self.max_calls = max_calls
        self.probability = float(probability)
        self.seed = seed
        self._rng = random.Random(seed)
        self.num_calls = 0

    def should_run(self, event: Event) -> bool:
        """Return whether this callback should handle `event`."""

        if event.name not in self.triggers:
            return False
        if self.max_calls is not None and self.num_calls >= self.max_calls:
            return False
        if self.every_n_steps is not None:
            step = event.step
            if step is None or step < self.start_step:
                return False
            if (step - self.start_step) % self.every_n_steps != 0:
                return False
        return self._draw_probability()

    def _draw_probability(self) -> bool:
        """Apply the probability gate using the callback-local RNG."""

        if self.probability >= 1.0:
            return True
        if self.probability <= 0.0:
            return False
        return self._rng.random() < self.probability

    def handle(self, event: Event) -> None:
        """Handle an event if this callback is subscribed to it."""

        if not self.should_run(event):
            return
        method = getattr(self, f"on_{event.name}", None)
        if method is not None:
            method(event)
        self.num_calls += 1


def _attach_event_metrics(event: Event, namespace: str, metrics: Mapping[str, object]) -> None:
    by_namespace = event.payload.setdefault("metrics_by_namespace", {})
    if not isinstance(by_namespace, dict):
        return
    existing = by_namespace.setdefault(namespace, {})
    if isinstance(existing, dict):
        existing.update(metrics)
