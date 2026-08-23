"""Typed event and operation contracts for one TPEN run."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ContextManager, Generic, Protocol, TypeVar


class Event:
    """Base type for an instantaneous run event."""


class Operation:
    """Base type for a scoped run operation."""


class DomainState:
    """Marker base for the mutable run-state object one domain owns.

    This base is deliberately EMPTY, and that is a design constraint rather
    than an unfinished class. The domains agree on nothing: training is
    coordinated by an integer step, while evaluation is coordinated by a task
    namespace *string*, and neither has a meaningful value for the other's
    coordinate. Hoisting ``step``, ``metrics``, or ``model`` here would invent
    a shared vocabulary that no second domain can honour, and every domain's
    handler would then have to defend against fields that are structurally
    absent for it.

    Its whole job is to mark "I am some domain's state object", so that
    `EventEmitter.emit` and `EventEmitter.scope` can accept one typed optional
    argument and a typed handler can declare which domain it observes. Named,
    typed field access lives on the concrete subclass.
    """


@dataclass(frozen=True)
class TrainingTiming:
    """Whole-iteration timing published on the typed training state.

    This value lives in the torch-free event vocabulary so timing producers
    and consumers can share it without importing the training stack.
    """

    step_time_sec: float
    step_time_sec_rolling_mean: float
    step_device_time_sec: float | None = None

    def __post_init__(self) -> None:
        """Validate host and optional device durations at the typed boundary."""

        for name, value in (
            ("step_time_sec", self.step_time_sec),
            ("step_time_sec_rolling_mean", self.step_time_sec_rolling_mean),
            ("step_device_time_sec", self.step_device_time_sec),
        ):
            if value is None and name == "step_device_time_sec":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real duration or None")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


class TrainingTimingState(DomainState):
    """Typed state capability exposing whole-iteration training timing.

    The concrete training state subclasses this capability and owns the
    mutable storage. Keeping the capability here lets timing callbacks import
    their contract without importing ``tpen.training`` or Torch.
    """

    timing: TrainingTiming | None


EventT = TypeVar("EventT", bound=Event)
OperationT = TypeVar("OperationT", bound=Operation)
StateT = TypeVar("StateT", bound=DomainState)


@dataclass(frozen=True)
class Occurrence(Generic[EventT]):
    """One run-local occurrence of a concrete event type.

    Parameters
    ----------
    event : EventT
        Typed event value.
    count : int
        One-based count for the concrete event or operation type in the
        current run context.
    monotonic_time : float or None, optional
        Process-local monotonic time captured by the emitter before artifact
        persistence and callback delivery. It is deliberately an occurrence
        envelope field, not domain-event data, and is not a cross-run clock.
        ``None`` preserves direct test construction outside a `RunContext`.
    """

    event: EventT
    count: int
    monotonic_time: float | None = None

@dataclass(frozen=True)
class Started(Event, Generic[OperationT]):
    """Record entry into a typed operation scope."""

    operation: OperationT


@dataclass(frozen=True)
class Ended(Event, Generic[OperationT]):
    """Record exit from a typed operation scope.

    Parameters
    ----------
    operation : OperationT
        Scoped operation that ended.
    succeeded : bool, optional
        Whether the scope body returned normally. A boundary still arrives
        after a body exception once its `Started` boundary was delivered.
    """

    operation: OperationT
    succeeded: bool = True

@dataclass(frozen=True)
class Subscription:
    """Select one typed event or one boundary of a typed operation.

    Parameters
    ----------
    subject : type[Event] or type[Operation]
        Base event or operation type to select. Matching uses ``isinstance``,
        so a base subject includes its subclasses.
    lifecycle : type[Started], type[Ended], or None, optional
        Bare lifecycle class for operation subscriptions. Instantaneous event
        subscriptions must use ``None``.
    """

    subject: type[Event] | type[Operation]
    lifecycle: type[Started] | type[Ended] | None = None

    def __post_init__(self) -> None:
        subject = self.subject
        if not isinstance(subject, type):
            raise TypeError("subscription subject must be an Event or Operation type")
        is_event = issubclass(subject, Event)
        is_operation = issubclass(subject, Operation)
        if not is_event and not is_operation:
            raise TypeError("subscription subject must be an Event or Operation subclass")
        if issubclass(subject, (Started, Ended)):
            raise TypeError("Started and Ended cannot be subscription subjects")
        if is_event and is_operation:
            raise TypeError("subscription subject cannot be both an Event and an Operation")
        if is_event and self.lifecycle is not None:
            raise ValueError("Event subscriptions require lifecycle=None")
        if is_operation and self.lifecycle is not Started and self.lifecycle is not Ended:
            raise ValueError("Operation subscriptions require bare Started or Ended lifecycle")

    @classmethod
    def of(cls, subject: type[Event]) -> Subscription:
        """Return an instantaneous event subscription."""

        return cls(subject=subject)

    @classmethod
    def started(cls, subject: type[Operation]) -> Subscription:
        """Return a subscription to starts of ``subject`` operations."""

        return cls(subject=subject, lifecycle=Started)

    @classmethod
    def ended(cls, subject: type[Operation]) -> Subscription:
        """Return a subscription to ends of ``subject`` operations."""

        return cls(subject=subject, lifecycle=Ended)

    def matches(self, event: Event) -> bool:
        """Return whether ``event`` is a delivered boundary for this selector."""

        if self.lifecycle is None:
            return not isinstance(event, (Started, Ended)) and isinstance(event, self.subject)
        return isinstance(event, self.lifecycle) and isinstance(event.operation, self.subject)


def started(subject: type[Operation]) -> Subscription:
    """Return a subscription to starts of ``subject`` operations."""

    return Subscription.started(subject)


def ended(subject: type[Operation]) -> Subscription:
    """Return a subscription to ends of ``subject`` operations."""

    return Subscription.ended(subject)


class EventEmitter(Protocol):
    """Protocol for run-local typed event emission.

    ``state`` is the emitting domain's state object. It is optional and
    defaults to ``None``, so a boundary with no domain state to offer emits
    exactly as before. The state travels beside the occurrence and never
    inside it: an event says WHEN something happened, and no data is attached
    to the event value or to its durable record.
    """

    def emit(
        self, event: EventT, *, state: DomainState | None = None
    ) -> Occurrence[EventT]:
        """Emit and return the next occurrence of ``event``."""

        ...

    def scope(
        self, operation: OperationT, *, state: DomainState | None = None
    ) -> ContextManager[Occurrence[Started[OperationT]]]:
        """Bracket ``operation`` with paired started and ended records.

        Both boundaries carry the same ``state`` reference, so a handler at the
        ended boundary observes whatever the scope body mutated.
        """

        ...


__all__ = [
    "DomainState",
    "Ended",
    "Event",
    "EventEmitter",
    "Occurrence",
    "Operation",
    "Started",
    "Subscription",
    "TrainingTiming",
    "TrainingTimingState",
    "ended",
    "started",
]
