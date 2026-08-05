"""Typed operations owned by the training domain."""

from __future__ import annotations

from dataclasses import dataclass

from tpen.events import Operation


@dataclass(frozen=True)
class CollectSamples(Operation):
    """Collect the walkers used by one training iteration.

    Parameters
    ----------
    step : int
        Durable trainer step associated with the sample collection. This is
        distinct from the operation's run-local occurrence count.
    """

    step: int


__all__ = ["CollectSamples"]
