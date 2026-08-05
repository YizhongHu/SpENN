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


__all__ = ["Ended", "Event", "EventEmitter", "Occurrence", "Operation", "Started"]
