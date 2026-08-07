"""Typed event and operation contracts for one TPEN run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ContextManager, Generic, Protocol, TypeVar


class Event:
    """Base type for an instantaneous run event."""


class Operation:
    """Base type for a scoped run operation."""


EventT = TypeVar("EventT", bound=Event)
OperationT = TypeVar("OperationT", bound=Operation)


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
    """

    event: EventT
    count: int


@dataclass(frozen=True)
class Started(Event, Generic[OperationT]):
    """Record entry into a typed operation scope."""

    operation: OperationT


@dataclass(frozen=True)
class Ended(Event, Generic[OperationT]):
    """Record exit from a typed operation scope, regardless of outcome."""

    operation: OperationT


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
    """Protocol for run-local typed event emission."""

    def emit(self, event: EventT) -> Occurrence[EventT]:
        """Emit and return the next occurrence of ``event``."""

        ...

    def scope(
        self, operation: OperationT
    ) -> ContextManager[Occurrence[Started[OperationT]]]:
        """Bracket ``operation`` with paired started and ended records."""

        ...


__all__ = [
    "Ended",
    "Event",
    "EventEmitter",
    "Occurrence",
    "Operation",
    "Started",
    "Subscription",
    "ended",
    "started",
]
