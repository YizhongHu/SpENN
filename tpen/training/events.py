"""Typed operations owned by the training domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tpen.events import Event, Operation


@dataclass(frozen=True)
class TrainingPhase(Operation):
    """One timed stage inside a single training iteration.

    Concrete subclasses declare ``phase_name``, the durable metric fragment
    that timing callbacks turn into keys such as ``sampling_time_sec``. Those
    fragments are part of the public metric surface, so they live next to the
    phase type rather than in a separate name table.

    This base class deliberately leaves ``phase_name`` unset, so it is abstract:
    constructing it, or any subclass that forgot to declare ``phase_name``,
    fails loudly instead of silently producing an unnamed timing metric.

    Parameters
    ----------
    step : int
        Durable zero-based trainer step this phase belongs to. This is distinct
        from the operation's run-local occurrence count.

    Raises
    ------
    TypeError
        If the concrete type declares no ``phase_name``.
    """

    # ClassVar: neither a dataclass field nor a serialized occurrence field.
    phase_name: ClassVar[str]

    step: int

    def __post_init__(self) -> None:
        # A phase with no durable metric fragment cannot be timed, so reject it
        # at construction rather than at metric-key lookup.
        if not hasattr(type(self), "phase_name"):
            raise TypeError(
                f"{type(self).__name__} must declare a phase_name ClassVar; "
                "TrainingPhase itself is abstract"
            )


@dataclass(frozen=True)
class CollectSamples(TrainingPhase):
    """Collect the walkers used by one training iteration."""

    phase_name: ClassVar[str] = "sampling"


@dataclass(frozen=True)
class BuildBatch(TrainingPhase):
    """Materialize the electron batch consumed by one training iteration."""

    phase_name: ClassVar[str] = "batch_build"


@dataclass(frozen=True)
class LocalEnergy(TrainingPhase):
    """Evaluate the Hamiltonian local energy for one training iteration."""

    phase_name: ClassVar[str] = "local_energy"


@dataclass(frozen=True)
class Forward(TrainingPhase):
    """Evaluate the wavefunction model for one training iteration."""

    phase_name: ClassVar[str] = "forward"


@dataclass(frozen=True)
class Objective(TrainingPhase):
    """Form the differentiable VMC objective for one training iteration."""

    phase_name: ClassVar[str] = "objective"


@dataclass(frozen=True)
class Backward(TrainingPhase):
    """Backpropagate the objective for one training iteration."""

    phase_name: ClassVar[str] = "backward"


@dataclass(frozen=True)
class OptimizerUpdate(TrainingPhase):
    """Apply one optimizer update for one training iteration."""

    phase_name: ClassVar[str] = "optimizer_step"


@dataclass(frozen=True)
class Metrics(TrainingPhase):
    """Assemble the post-step training metrics for one training iteration."""

    phase_name: ClassVar[str] = "post_step_metrics"


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


@dataclass(frozen=True)
class UpdateCompleted(Event):
    """Signal that one optimizer update was applied and has returned.

    Emitted after the enclosing `OptimizerUpdate` scope closes, so it always
    follows that scope's ``Ended`` record.

    Parameters
    ----------
    iteration : TrainingIteration
        Iteration whose optimizer update completed.
    """

    iteration: TrainingIteration


@dataclass(frozen=True)
class UpdateSkipped(Event):
    """Signal that one iteration deliberately applied no optimizer update.

    Parameters
    ----------
    iteration : TrainingIteration
        Iteration that skipped its optimizer update.
    """

    iteration: TrainingIteration


__all__ = [
    "Backward",
    "BuildBatch",
    "CollectSamples",
    "Forward",
    "LocalEnergy",
    "Metrics",
    "Objective",
    "OptimizerUpdate",
    "TrainingIteration",
    "TrainingIterationCompleted",
    "TrainingPhase",
    "UpdateCompleted",
    "UpdateSkipped",
]
