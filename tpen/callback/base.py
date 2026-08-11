"""Base callback event and scheduling primitives."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, final

from tpen.artifacts import RunContext
from tpen.events import DomainState, Ended, Started, StateT
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence

from .cadence import CadenceGate, SubscriptionGroup, validate_subscription_groups

# Named for the same reason `spenn.status` and `spenn.bootstrap` are: a run's
# logging configuration can silence or route this channel on its own.
_LOGGER = logging.getLogger("spenn.callback")


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
    # Event shapes already reported as arriving with no state at all, so the
    # diagnostic in `StatefulCallback.handle_occurrence` fires once per shape
    # instead of once per training step.
    reported_missing_state: set[tuple[type[object], type[object] | None]] = field(
        default_factory=set
    )

    def __post_init__(self) -> None:
        self.gate = None if self.group.cadence is None else CadenceGate(self.group.cadence)

    def reset(self) -> None:
        if self.gate is not None:
            self.gate.reset()
        self.open_pairs.clear()
        self.reported_missing_state.clear()


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
        self._validate_typed_groups(groups)
        self._typed_group_states = tuple(_TypedGroupState(group) for group in groups)
        self._typed_context: object = _UNSET_CONTEXT

    def _validate_typed_groups(self, groups: tuple[SubscriptionGroup, ...]) -> None:
        """Reject a group plan this delivery shape cannot honour.

        Overridden by both public bases. It exists because ``stateless`` is a
        declaration about DELIVERY, and only the base that owns a delivery
        signature can say whether a given declaration is satisfiable there.
        """

        del groups

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
    def _typed_group_selects(
        group_state: _TypedGroupState, occurrence: Occurrence[TypedEvent]
    ) -> bool:
        """Return whether a selector names this occurrence, gating nothing.

        Kept separate from `_typed_group_delivers`, which advances the group's
        cadence gate and its open-pair table as a side effect of asking. A
        diagnostic that wants only "was this occurrence addressed to this
        group?" must not perturb the schedule by asking.
        """

        return any(
            selector.matches(occurrence.event) for selector in group_state.group.selectors
        )

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

    def _validate_typed_groups(self, groups: tuple[SubscriptionGroup, ...]) -> None:
        """Reject a ``stateless`` declaration, which is vacuous on this base.

        Every group here is already delivered without state, so the flag adds
        nothing -- but it would imply, to a reader, that some sibling group on
        the same class IS stateful, which this base cannot express. Refused at
        construction rather than ignored, so the disagreement between what the
        group declares and what the class can do surfaces as an error message
        instead of as a subtly wrong reading.
        """

        if any(group.stateless for group in groups):
            raise TypeError(
                f"{type(self).__name__} is a Callback, whose groups are all "
                "delivered without state; stateless=True is meaningful only on a "
                "StatefulCallback"
            )

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

    Delivery is decided PER GROUP, not per callback. A group that did not
    declare `tpen.callback.cadence.SubscriptionGroup.stateless` receives the
    emitting domain's own `tpen.events.DomainState` through the three-argument
    ``handle_occurrence_impl``, and only when that state is an instance of this
    callback's ``state_type``; a callback whose domain does not match is
    skipped, not failed, because one run may emit several domains' states and a
    callback simply has nothing to observe outside its own. A group that DID
    declare ``stateless`` receives the two-argument
    ``handle_stateless_occurrence_impl`` instead, and no state filter applies to
    it at all.

    That is a deliberate weakening, recorded as an amendment to ADR-E008. The
    invariant used to be "my handler always receives my domain's state"; it is
    now "each group receives state or not, as declared". It buys the one thing
    the old invariant made inexpressible: a callback that observes both its own
    domain and a boundary belonging to no domain, such as the run lifecycle,
    which carries no state and never will. The alternative was splitting such a
    class in two and changing its config-facing ``_target_``.

    Because `tpen.callback.cadence.validate_subscription_groups` rejects
    overlapping deliveries, at most one group matches any occurrence, so the
    two hooks can never both fire for a single delivery.

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

    def _validate_typed_groups(self, groups: tuple[SubscriptionGroup, ...]) -> None:
        """Reject a stateless declaration this class could only swallow.

        Two shapes are refused, both of which would otherwise reintroduce the
        silence this capability exists to remove:

        - a stateless group on a class that never overrode
          ``handle_stateless_occurrence_impl``, whose deliveries would land in
          the inherited no-op and disappear;
        - a plan whose groups are ALL stateless, which declares a ``state_type``
          that can never route anything and should be a `Callback` instead.

        A class with no groups at all is untouched: `tpen.callback.Status`
        subscribes nothing when its training line is off, and that is not the
        same claim.
        """

        stateless = tuple(group for group in groups if group.stateless)
        if not stateless:
            return
        if len(stateless) == len(groups):
            raise TypeError(
                f"every subscription group on {type(self).__name__} is stateless, so "
                "its state_type can never route a delivery; make it a Callback"
            )
        if (
            type(self).handle_stateless_occurrence_impl
            is StatefulCallback.handle_stateless_occurrence_impl
        ):
            raise TypeError(
                f"{type(self).__name__} declares a stateless subscription group but "
                "does not override handle_stateless_occurrence_impl, so those "
                "deliveries would be silently discarded"
            )

    @final
    def handle_occurrence(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: DomainState | None,
    ) -> None:
        """Match, gate, and route one typed occurrence group by group.

        ``state`` is whatever the emitting domain passed -- this callback's own
        domain state, some other domain's, or nothing at all. The discrimination
        happens HERE rather than in `tpen.artifacts.RunContext`, because the
        dispatcher cannot know which group a given occurrence will match and the
        answer differs between them. The narrow ``StateT`` annotation therefore
        lives on ``handle_occurrence_impl``, which is reached only after the
        instance check below.

        The state check runs BEFORE `_typed_group_delivers` for a stateful
        group, which is not incidental: asking that question advances the
        group's cadence gate, so checking it first is what keeps a foreign
        domain's occurrences from consuming schedule the way they never did when
        the dispatcher filtered whole callbacks.
        """

        self._ensure_typed_context(context)
        for index, group_state in enumerate(self._typed_group_states):
            if group_state.group.stateless:
                if self._typed_group_delivers(group_state, occurrence):
                    self.handle_stateless_occurrence_impl(occurrence, context)
                continue
            if not isinstance(state, self.state_type):
                self._report_missing_state(index, group_state, occurrence, state)
                continue
            if self._typed_group_delivers(group_state, occurrence):
                self.handle_occurrence_impl(occurrence, context, state)

    def _report_missing_state(
        self,
        index: int,
        group_state: _TypedGroupState,
        occurrence: Occurrence[TypedEvent],
        state: DomainState | None,
    ) -> None:
        """Log the one skip shape that is a wiring error rather than routine.

        The two skips are told apart by whether ANY state arrived, and the
        distinction is not cosmetic:

        - some other domain's state arrived. Routine, and silent. A run emits
          several domains' boundaries and this callback is not their audience;
          it happens on most occurrences of a mixed run, so a diagnostic here
          would be pure noise.
        - nothing arrived, on a boundary this group actually selected. Nobody
          could have been the audience, so either the emitter omitted its
          ``state=`` or the group should have declared ``stateless=True``. That
          is a mistake every time, and it is precisely the silence that trapped
          `tpen.callback.Status` and `tpen.callback.ArtifactIndex`.

        Reported at WARNING rather than raised, because raising would break the
        pinned behaviour that a boundary emitted without state delivers nothing
        rather than killing the run -- a callback misconfiguration must not take
        down training. Reported once per event shape per group, because the
        selected boundary may be a per-step one.
        """

        if state is not None or not self._typed_group_selects(group_state, occurrence):
            return
        event = occurrence.event
        # Typed discrimination rather than attribute probing: a lifecycle event
        # is keyed by its operation type too, so Started[A] and Started[B] are
        # reported separately.
        if isinstance(event, (Started, Ended)):
            key: tuple[type[object], type[object] | None] = (
                type(event),
                type(event.operation),
            )
            shape = f"{type(event).__name__}[{type(event.operation).__name__}]"
        else:
            key = (type(event), None)
            shape = type(event).__name__
        if key in group_state.reported_missing_state:
            return
        group_state.reported_missing_state.add(key)
        _LOGGER.warning(
            "%s subscription group %d requires %s, but %s was emitted with no domain "
            "state at all, so nothing was delivered. Either the emitter omitted "
            "state=, or the group should declare stateless=True. Reported once per "
            "event shape per group.",
            type(self).__name__,
            index,
            self.state_type.__name__,
            shape,
        )

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: StateT,
    ) -> None:
        """Handle one occurrence admitted by a configured typed group."""

        del occurrence, context, state

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Handle one occurrence admitted by a group that declared no state.

        Deliberately NOT named ``handle_occurrence_impl``: the two hooks differ
        in arity, and one name carrying two signatures on one class would make
        it impossible to tell, at an override site, which route the author meant
        -- and a wrong guess would be silent. `_validate_typed_groups` refuses a
        stateless group unless this is overridden, so the inherited no-op is
        reachable only by a class that declared no stateless group.
        """

        del occurrence, context


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
