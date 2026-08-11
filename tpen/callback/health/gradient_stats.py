"""Gradient health callback."""

from __future__ import annotations

from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.dependencies import require_torch
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from ..base import StatefulCallback
from ..cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class GradientStats(StatefulCallback[TrainerState]):
    """Track gradient health once each training iteration completes.

    Reads parameter gradients from ``state.model`` and logs norm/finite metrics
    under ``checks/gradient``. With ``check_finite`` it fails on non-finite
    gradients, and with ``max_global_grad_norm`` it fails when the global norm
    is exceeded. It does not require convergence or small gradients.

    Notes
    -----
    The observed gradients are the ones ``optimizer.step()`` consumed, still
    live because the trainer clears them with ``optimizer.zero_grad`` part-way
    through the *following* iteration rather than after the update. That
    ordering is a real contract of this callback's output and is pinned by
    ``tests/unit/training/test_gradient_observation_contract.py``: moving the
    clear to just after ``optimizer.step()`` empties every metric here without
    changing training at all.

    They are also the *clipped* gradients when ``gradient_clip_norm`` is
    configured, because ``clip_grad_norm_`` mutates ``.grad`` in place before
    the update. Observing at an earlier boundary would silently republish
    unclipped numbers under the same metric names.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(
        self,
        *,
        fail_fast: bool = False,
        max_global_grad_norm: float | None = None,
        check_finite: bool = True,
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
        self.max_global_grad_norm = None if max_global_grad_norm is None else float(max_global_grad_norm)
        self.check_finite = bool(check_finite)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Summarize gradients of the model parameters."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        # The coordinate rides the typed event, never `state.step`: the state's
        # value fields are assigned at the end of the trainer's loop body, so
        # they are stale at any earlier boundary and still hold the constructor
        # default ``-1`` on the first iteration.
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        torch = require_torch(feature="GradientStats callback")
        grads = [
            param.grad.detach().reshape(-1)
            for param in state.model.parameters()
            if param.grad is not None
        ]
        metrics: dict[str, Any] = {
            "n_grad_tensors": len(grads),
            "n_grad_elements": 0,
            "global_grad_norm": 0.0,
            "max_abs_grad": 0.0,
            "mean_abs_grad": 0.0,
            "nonfinite_grad_fraction": 0.0,
        }
        global_norm = 0.0
        nonfinite_fraction = 0.0
        if grads:
            flat = torch.cat(grads)
            n_elements = int(flat.numel())
            finite_mask = torch.isfinite(flat)
            n_finite = int(finite_mask.sum().item())
            nonfinite_fraction = float((n_elements - n_finite) / n_elements) if n_elements else 0.0
            finite_values = flat[finite_mask]
            global_norm = float(finite_values.norm().item()) if n_finite else 0.0
            abs_finite = finite_values.abs()
            metrics["n_grad_elements"] = n_elements
            metrics["global_grad_norm"] = global_norm
            metrics["max_abs_grad"] = float(abs_finite.max().item()) if n_finite else 0.0
            metrics["mean_abs_grad"] = float(abs_finite.mean().item()) if n_finite else 0.0
            metrics["nonfinite_grad_fraction"] = nonfinite_fraction

        failure: str | None = None
        if self.check_finite and nonfinite_fraction > 0.0:
            failure = f"nonfinite_grad_fraction={nonfinite_fraction} exceeds 0.0"
        elif self.max_global_grad_norm is not None and global_norm > self.max_global_grad_norm:
            failure = f"global_grad_norm={global_norm} exceeds max_global_grad_norm={self.max_global_grad_norm}"

        metrics["passed"] = failure is None
        context.log(metrics, step=step, namespace="checks/gradient")
        if self.fail_fast and failure is not None:
            raise RuntimeError(f"GradientStats failed at step {step}: {failure}")



__all__ = ["GradientStats"]
