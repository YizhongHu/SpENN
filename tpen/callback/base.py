"""Base callback event and scheduling primitives."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, final

from tpen.artifacts import RunContext
from tpen.events import DomainState, Ended, Started, StateT
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence

from .cadence import CadenceGate, SubscriptionGroup, validate_subscription_groups


@dataclass
class Event:
    """Lifecycle event delivered to callbacks."""

    name: str
    context: RunContext
    state: object | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    step: int | None = None


@dataclass
class _TypedGroupState:
    group: SubscriptionGroup
    gate: CadenceGate | None = field(init=False)
    open_pairs: dict[tuple[type[object], int], bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gate = None if self.group.cadence is None else CadenceGate(self.group.cadence)

    def reset(self) -> None:
        if self.gate is not None:
            self.gate.reset()
        self.open_pairs.clear()


_UNSET_CONTEXT = object()


class _CallbackCore:
    """Scheduling, subscription-plan, and context-reset machinery.

    This holds everything the two public callback bases share: the legacy
    string-trigger path, the typed subscription plan with its per-group cadence
    gates, and the reset performed when the owning `RunContext` identity
    changes. It owns no delivery signature of its own, because that is exactly
    what distinguishes `Callback` from `StatefulCallback`.

    Parameters
    ----------
    triggers : iterable of str, optional
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
    typed_groups : iterable of SubscriptionGroup, optional
        Typed selectors with independent occurrence-count cadence gates.
    """

    def __init__(
        self,
        triggers: Iterable[str] = (),
        every_n_steps: int | None = None,
        start_step: int = 0,
        max_calls: int | None = None,
        probability: float = 1.0,
        seed: int | None = None,
        *,
        typed_groups: Iterable[SubscriptionGroup] = (),
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
        groups = tuple(typed_groups)
        if not all(isinstance(group, SubscriptionGroup) for group in groups):
            raise TypeError("typed_groups must contain only SubscriptionGroup values")
        validate_subscription_groups(groups)
        self._typed_group_states = tuple(_TypedGroupState(group) for group in groups)
        self._typed_context: object = _UNSET_CONTEXT

    def should_run(self, event: Event) -> bool:
        """Return whether this callback should handle `event`."""

        if event.name not in self.triggers:
            return False
        if self.max_calls is not None and self.num_calls >= self.max_calls:
            return False
        if self.every_n_steps is not None:
            step = self._legacy_cadence_step(event)
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

    def _legacy_cadence_step(self, event: Event) -> int | None:
        """Return the legacy step coordinate used by `should_run`."""

        return event.step

    def handle(self, event: Event) -> None:
        """Handle an event if this callback is subscribed to it."""

        self._ensure_typed_context(event.context)
        if not self.should_run(event):
            return
        method = getattr(self, f"on_{event.name}", None)
        if method is not None:
            method(event)
        self.num_calls += 1

    def _ensure_typed_context(self, context: object) -> None:
        if not self._typed_group_states or self._typed_context is context:
            return
        self._typed_context = context
        for group_state in self._typed_group_states:
            group_state.reset()
        self._reset_typed_state()

    def _reset_typed_state(self) -> None:
        """Reset subclass caches when typed state moves to a new context."""

    @staticmethod
    def _typed_group_delivers(
        group_state: _TypedGroupState, occurrence: Occurrence[TypedEvent]
    ) -> bool:
        event = occurrence.event
        if not isinstance(event, (Started, Ended)):
            if not any(
                selector.matches(event) for selector in group_state.group.selectors
            ):
                return False
            return group_state.gate is None or group_state.gate.should_run(occurrence.count)

        operation = event.operation
        if not any(
            selector.lifecycle is not None and isinstance(operation, selector.subject)
            for selector in group_state.group.selectors
        ):
            return False
        coordinate = (type(operation), occurrence.count)
        if isinstance(event, Started):
            decision = group_state.gate is None or group_state.gate.should_run(
                occurrence.count
            )
            # Cache rejected and maxed decisions too, so Ended never redraws.
            group_state.open_pairs[coordinate] = decision
        else:
            decision = group_state.open_pairs.pop(coordinate, False)
        return decision and any(
            selector.matches(event) for selector in group_state.group.selectors
        )


class Callback(_CallbackCore):
    """Callback that observes typed occurrences without any domain state.

    Delivery is two-argument: ``(occurrence, context)``. This is the base for
    every observer that needs only the moment and the run context -- timing,
    metadata, artifact indexing. It exists as a sibling of `StatefulCallback`
    rather than its parent so that a state-free observer never has to accept and
    ignore a state parameter.

    Constructor parameters are documented on `_CallbackCore`.
    """

    @final
    def handle_occurrence(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Match, gate, and deliver one typed occurrence in group order."""

        self._ensure_typed_context(context)
        for group_state in self._typed_group_states:
            if self._typed_group_delivers(group_state, occurrence):
                self.handle_occurrence_impl(occurrence, context)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Handle one occurrence admitted by a configured typed group."""

        del occurrence, context


class StatefulCallback(_CallbackCore, Generic[StateT]):
    """Callback that observes typed occurrences together with domain state.

    Delivery is three-argument: ``(occurrence, context, state)``. The emitting
    domain passes its own `tpen.events.DomainState` beside each occurrence, and
    the dispatcher delivers it only to callbacks that declare a matching
    ``state_type``. A callback whose domain does not match is skipped, not
    failed: one run may emit several domains' states, and a callback simply has
    nothing to observe outside its own.

    This is NOT a subclass of `Callback`. The two delivery arities differ, so an
    inheritance edge would be a substitutability violation, and it would also
    make the dispatcher's ``isinstance`` discrimination meaningless.

    Constructor parameters are documented on `_CallbackCore`.

    Attributes
    ----------
    state_type : type of DomainState
        The concrete domain state class this callback observes. Every concrete
        subclass must declare it, and `__init_subclass__` rejects one that does
        not. It is the RUNTIME authority for delivery, and it is not redundant
        with the ``StatefulCallback[...]`` type argument: Python does not retain
        a subclass's type argument at runtime, so the type argument is a purely
        static claim and cannot route anything. Declaring the class explicitly
        also means renaming a state class cannot silently redirect delivery,
        the same reason `tpen.training.events.TrainingPhase` makes concrete
        phases declare ``phase_name``.
    """

    # ClassVar: shared per concrete callback class, never per instance, and
    # deliberately left unset here so this base is abstract.
    state_type: ClassVar[type[DomainState]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a subclass that declares no observable domain state."""

        super().__init_subclass__(**kwargs)
        # A stateful callback with no declared domain would silently receive
        # nothing forever, so refuse the class rather than the delivery.
        if not hasattr(cls, "state_type"):
            raise TypeError(
                f"{cls.__name__} must declare a state_type ClassVar; "
                "StatefulCallback itself is abstract"
            )
        state_type = cls.state_type
        if not isinstance(state_type, type) or not issubclass(state_type, DomainState):
            raise TypeError(
                f"{cls.__name__}.state_type must be a DomainState subclass, got "
                f"{state_type!r}"
            )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject an instance of the abstract base and forward everything else.

        `__init_subclass__` already covers every subclass, so this covers only
        the undeclared base itself, which would otherwise be accepted here and
        fail with a bare ``AttributeError`` inside dispatch, mid-run. Same
        reasoning as `tpen.training.events.TrainingPhase`, which rejects an
        undeclared phase at construction rather than at metric-key lookup.
        Arguments are forwarded unchanged; `_CallbackCore` documents them.
        """

        if not hasattr(type(self), "state_type"):
            raise TypeError(
                "StatefulCallback is abstract; declare state_type on a subclass"
            )
        super().__init__(*args, **kwargs)

    @final
    def handle_occurrence(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: StateT,
    ) -> None:
        """Match, gate, and deliver one typed occurrence with its domain state.

        The caller is responsible for confirming that ``state`` is an instance
        of this callback's ``state_type``; `tpen.artifacts.RunContext` does that
        before it ever reaches this method.
        """

        self._ensure_typed_context(context)
        for group_state in self._typed_group_states:
            if self._typed_group_delivers(group_state, occurrence):
                self.handle_occurrence_impl(occurrence, context, state)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: StateT,
    ) -> None:
        """Handle one occurrence admitted by a configured typed group."""

        del occurrence, context, state


def _legacy_event(
    *,
    name: str,
    context: RunContext,
    state: object | None = None,
    payload: dict[str, Any] | None = None,
    step: int | None = None,
) -> Event:
    """Normalize one legacy ingress event without probing runtime state."""

    event_payload = {} if payload is None else payload
    explicit_step = None if step is None else int(step)
    payload_has_step = "step" in event_payload
    payload_value = event_payload.get("step")
    payload_step = None if payload_value is None else int(payload_value)
    if explicit_step is not None and payload_has_step and payload_step != explicit_step:
        raise ValueError(
            "legacy event step mismatch: "
            f"explicit step {explicit_step} != payload step {payload_step}"
        )
    resolved_step = explicit_step if explicit_step is not None else payload_step
    return Event(
        name=name,
        context=context,
        state=state,
        payload=event_payload,
        step=resolved_step,
    )


def _attach_event_metrics(event: Event, namespace: str, metrics: Mapping[str, object]) -> None:
    by_namespace = event.payload.setdefault("metrics_by_namespace", {})
    if not isinstance(by_namespace, dict):
        return
    existing = by_namespace.setdefault(namespace, {})
    if isinstance(existing, dict):
        existing.update(metrics)
