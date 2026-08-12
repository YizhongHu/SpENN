"""Run metadata lifecycle callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tpen.artifacts import RunContext, write_json
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.run_events import RunCompleted, RunFailed, RunStarted

from .base import Callback
from .cadence import SubscriptionGroup


class Metadata(Callback):
    """Write run metadata at each boundary of the run's own lifecycle.

    Data-free: it reads `tpen.artifacts.RunMetadata` off the context and, on a
    failure, the two fields `tpen.run_events.RunFailed` carries. Nothing here
    needs a domain state object, so this stays a plain
    `tpen.callback.Callback`.

    ``triggers`` is gone from the signature. ADR-E002 forbids a config from
    naming the events a callback answers to, and a config that still passes the
    key now fails loudly with a duplicate-argument ``TypeError`` rather than
    quietly re-enabling the legacy string path alongside the typed one.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination for ``metadata.json``.
    **kwargs
        Forwarded to `tpen.callback.Callback`.
    """

    def __init__(self, output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(
            # All three selectors in ONE group: `validate_subscription_groups`
            # rejects overlapping deliveries across groups, and one group is
            # also the honest shape here, since the three share a single
            # ungated decision.
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    )
                ),
            ),
            **kwargs,
        )
        self.output_path = Path(output_path)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Record the run's status at whichever lifecycle boundary just fired."""

        event = occurrence.event
        if isinstance(event, RunStarted):
            self._write(context, status="running")
            return
        if isinstance(event, RunCompleted):
            self._write(context, status="completed")
            return
        if isinstance(event, RunFailed):
            self._write(context, status="failed", failure=event)

    def _write(
        self, context: RunContext, *, status: str, failure: RunFailed | None = None
    ) -> None:
        metadata = context.metadata
        metadata.status = status
        data = metadata.to_dict()
        data["status"] = status
        if failure is not None:
            # Named field reads on a frozen typed event. The legacy path took a
            # live exception out of an untyped payload and re-derived these two
            # strings here; `tpen.run` already had both, so they now ride the
            # event that reports the moment.
            data["exception_type"] = failure.exception_type
            data["exception_message"] = failure.exception_message
        write_json(self.output_path, data)



__all__ = ["Metadata"]
