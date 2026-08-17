"""Trace the trainable scalars owned by post-readout log-amplitude factors."""

from __future__ import annotations

from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class FactorScalars(StatefulCallback[TrainerState]):
    """Log each post-readout factor's trainable scalars under ``train/factors``.

    Why this exists, stated as the measurement it enables. A convergence
    assessment run on the loss can only see parameters the loss is sensitive
    to. A cusp range parameter is not one of them at the resolution that
    matters: it can sit well away from its optimum while costing far less
    energy than the window-to-window scatter of a training trace, so the loss
    plateaus on schedule while the model is still structurally in transit. The
    remedy is not a better energy statistic, it is measuring the scalars
    directly -- the same reasoning that makes initialization bias a test on the
    sampled MEAN rather than on ``tau`` or split-Rhat, because those cannot see
    it either.

    Constrained values are what a factor reports and therefore what lands here.
    A range parameter held positive through a softplus moves on a raw axis that
    is not the physical one, so a trace of raw values can show motion where the
    effective parameter has settled, or stillness where it has not. Factors that
    reparameterize report both, and the metric names distinguish them.

    The scalars come from each factor's own `scalar_diagnostics` contract, not
    from walking ``named_parameters()``. A factor names its own quantities; a
    consumer that instead scraped a parameter container would have to guess
    which entry meant what, and would silently start reporting a different
    quantity the moment a factor gained a parameter.

    Parameters
    ----------
    every_n_steps : int, optional
        Durable-step cadence, consumed by `pop_step_cadence`.
    fail_fast : bool, optional
        Raise when a factor reports a non-finite scalar. Off by default: this
        is a trace, and `DataIntegrity` owns run-halting finiteness policy.

    Notes
    -----
    A model whose factors own no scalars logs ``n_scalars`` of ``0`` and
    nothing else. That is a real observation -- the factors were asked and had
    nothing to report -- and is distinguishable from the callback not running,
    which emits no row at all.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, *, fail_fast: bool = False, **kwargs: Any) -> None:
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(TrainingIterationCompleted),)),
            ),
            **kwargs,
        )
        # Durable trainer step, not run-local occurrence count: the latter
        # restarts after a restore and would resample the cadence.
        self._steps = StepCadenceGate(cadence)
        self.fail_fast = bool(fail_fast)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Read every factor's declared scalars and log them."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        # The coordinate rides the typed event; `state.step` is assigned at the
        # end of the trainer loop body and is stale at this boundary.
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        scalars = collect_factor_scalars(state.model)
        metrics: dict[str, Any] = dict(scalars)
        metrics["n_scalars"] = len(scalars)
        nonfinite = sorted(name for name, value in scalars.items() if not _is_finite(value))
        metrics["nonfinite_scalar_count"] = len(nonfinite)
        metrics["passed"] = not nonfinite
        context.log(metrics, step=step, namespace="train/factors")
        if nonfinite and self.fail_fast:
            raise ValueError(f"factor scalars are not finite: {nonfinite}")


def collect_factor_scalars(model: Any) -> dict[str, float]:
    """Return the flat scalar mapping declared by a model's factors.

    Parameters
    ----------
    model : Any
        Wave function exposing a ``factors`` sequence. A model without one has
        no post-readout factors and reports nothing.

    Returns
    -------
    dict
        Scalars keyed ``factors.<index>.<name>``. The index prefix keeps two
        factors of the same class distinguishable, which a bare name would not.
    """

    factors = getattr(model, "factors", None)
    if factors is None:
        return {}
    scalars: dict[str, float] = {}
    for index, factor in enumerate(factors):
        declared = getattr(factor, "scalar_diagnostics", None)
        if not callable(declared):
            continue
        for name, value in declared().items():
            scalars[f"factors.{index}.{name}"] = float(value)
    return scalars


def _is_finite(value: float) -> bool:
    """Return whether one scalar is finite, without importing torch."""

    return value == value and value not in (float("inf"), float("-inf"))


__all__ = ["FactorScalars", "collect_factor_scalars"]
