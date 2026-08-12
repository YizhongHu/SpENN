"""Typed events owned by the checkpoint domain."""

from __future__ import annotations

from dataclasses import dataclass

from tpen.events import Event

from .restore import RestoreReport


@dataclass(frozen=True)
class CheckpointRestored(Event):
    """A checkpoint restore finished and reported what it loaded.

    This event lives with the checkpoint domain rather than with either runner
    because both `tpen.runner.Train` and `tpen.runner.Evaluate` reach the same
    moment, and `RestoreReport` is defined here.

    Unusually for this program, it has no callback subscriber and is not
    expected to grow one: the legacy ``checkpoint_restored`` string has two
    emitters and zero handlers repo-wide. It is typed anyway because its
    consumer is the durable record. The human's D3 ruling put restored
    checkpoint identity on the typed evaluation lifecycle rather than smuggling
    it into a metric key or a `tpen.logging.LogRecord` field, and that identity
    reaches ``occurrences.jsonl`` only through this event. Field-wise
    serialization of a nested non-`Event` dataclass is the prerequisite: without
    it the report would serialize as a bare type marker and ``completed_updates``
    -- the whole point of carrying it -- would be silently dropped.

    Parameters
    ----------
    report : RestoreReport
        What the restore resolved and loaded, including the resume cursor and
        the applied-update counter that identify the restored model version.
    """

    report: RestoreReport


__all__ = ["CheckpointRestored"]
