"""Typed operations owned by the training domain."""

from __future__ import annotations

from dataclasses import dataclass

from tpen.events import Event, Operation


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


@dataclass(frozen=True)
class TrainingIteration(Operation):
    """One attempted trainer iteration at a durable zero-based step.

    Parameters
    ----------
    step : int
        Durable zero-based trainer step attempted by this scope.
    """

    step: int


@dataclass(frozen=True)
class TrainingIterationCompleted(Event):
    """Signal successful completion of one training iteration.

    Parameters
    ----------
    iteration : TrainingIteration
        Successfully completed iteration identity.
    """

    iteration: TrainingIteration


__all__ = ["CollectSamples", "TrainingIteration", "TrainingIterationCompleted"]
