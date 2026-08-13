"""Sampler health callback."""

from __future__ import annotations

from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from ..base import StatefulCallback
from ..cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class SamplerHealth(StatefulCallback[TrainerState]):
    """Expose sampler statistics under ``checks/sampler`` with optional bounds.

    Reads the typed `tpen.sampling.SamplerStats` record on ``state.sampler_stats``
    and lets that record compose the check key set, so ``checks/sampler`` names
    are spelled in exactly one place. A state carrying no diagnostics still
    reports ``passed``. When acceptance-rate bounds are configured and violated,
    ``passed`` is ``False``; it raises only if ``fail_fast`` is set.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(
        self,
        *,
        fail_fast: bool = False,
        min_acceptance_rate: float | None = None,
        max_acceptance_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            # Subscriptions are class-owned, never configured: this callback
            # observes completed training iterations and nothing else.
            typed_groups=(
                SubscriptionGroup(
                    selectors=(Subscription.of(TrainingIterationCompleted),)
                ),
            ),
            **kwargs,
        )
        # `cadence=None` on the group above is deliberate: a group `Cadence`
        # gates on the run-local occurrence count, which restarts after a
        # restore. Scheduling uses the durable trainer step instead.
        self._steps = StepCadenceGate(cadence)
        self.fail_fast = bool(fail_fast)
        self.min_acceptance_rate = None if min_acceptance_rate is None else float(min_acceptance_rate)
        self.max_acceptance_rate = None if max_acceptance_rate is None else float(max_acceptance_rate)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Log available sampler diagnostics and check crude bounds."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        # The coordinate rides the typed event, never `state.step`; see the note
        # on `tpen.training.state.TrainerState`'s value fields in the trainer.
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        # The state is typed now, so the record is read by name. ``None`` is a
        # declared value of the field, not a missing attribute.
        stats = state.sampler_stats

        metrics: dict[str, Any] = {} if stats is None else stats.as_check_metrics()

        failure = None if stats is None else self._acceptance_failure(stats.acceptance_rate)

        metrics["passed"] = failure is None
        context.log(metrics, step=step, namespace="checks/sampler")
        if self.fail_fast and failure is not None:
            raise RuntimeError(f"SamplerHealth failed at step {step}: {failure}")

    def _acceptance_failure(self, acceptance_rate: float) -> str | None:
        """Return a bound-violation description, or ``None`` when in bounds."""

        if self.min_acceptance_rate is not None and acceptance_rate < self.min_acceptance_rate:
            return f"acceptance_rate={acceptance_rate} below min_acceptance_rate={self.min_acceptance_rate}"
        if self.max_acceptance_rate is not None and acceptance_rate > self.max_acceptance_rate:
            return f"acceptance_rate={acceptance_rate} above max_acceptance_rate={self.max_acceptance_rate}"
        return None


__all__ = ["SamplerHealth"]
