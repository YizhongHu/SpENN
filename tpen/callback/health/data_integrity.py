"""Training data integrity health callback."""

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


class DataIntegrity(StatefulCallback[TrainerState]):
    """Hard guardrail catching invalid training tensors at iteration end.

    Inspects the batch, wavefunction output (``sign``/``logabs``), local energy,
    and loss on the `TrainerState`, logging finite/validity metrics under
    ``checks/data_integrity``. In ``fail_fast`` mode a failed required check
    raises a clear `RuntimeError` instead of silently continuing.

    Notes
    -----
    Two independent axes meet here and must not be conflated. This callback
    *runs* after the optimizer update, but the tensors it describes are the
    pre-update ones that produced that update. That is the correct pairing: the
    check reports on the data the step consumed, not on the model the step
    produced.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(
        self,
        *,
        fail_fast: bool = False,
        max_nonfinite_energy_fraction: float = 0.0,
        max_nonfinite_logabs_fraction: float = 0.0,
        check_loss: bool = True,
        check_wavefunction_output: bool = True,
        check_batch: bool = True,
        strict_sign_values: bool = True,
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
        self.max_nonfinite_energy_fraction = float(max_nonfinite_energy_fraction)
        self.max_nonfinite_logabs_fraction = float(max_nonfinite_logabs_fraction)
        self.check_loss = bool(check_loss)
        self.check_wavefunction_output = bool(check_wavefunction_output)
        self.check_batch = bool(check_batch)
        self.strict_sign_values = bool(strict_sign_values)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Validate the most recent training step's tensors."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        # The coordinate rides the typed event, never `state.step`; see the note
        # on `tpen.training.state.TrainerState`'s value fields in the trainer.
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        metrics: dict[str, Any] = {}
        failures: list[str] = []

        # The state is typed now, so its fields are read by name. ``None`` is a
        # declared value of each field, not a missing attribute. The probes that
        # remain below inspect the *contents* -- a batch or output object whose
        # own contract this callback verifies rather than assumes.
        local_energy = state.local_energy
        if local_energy is not None:
            finite, total = _finite_counts(local_energy)
            energy_fraction = _nonfinite_fraction(finite, total)
            metrics["local_energy_finite_count"] = finite
            metrics["local_energy_total_count"] = total
            metrics["local_energy_nonfinite_fraction"] = energy_fraction
            if energy_fraction > self.max_nonfinite_energy_fraction:
                failures.append(
                    f"local_energy_nonfinite_fraction={energy_fraction} exceeds "
                    f"max_nonfinite_energy_fraction={self.max_nonfinite_energy_fraction}"
                )

        if self.check_wavefunction_output:
            output = state.wavefunction_output
            if output is not None:
                finite, total = _finite_counts(output.logabs)
                logabs_fraction = _nonfinite_fraction(finite, total)
                metrics["logabs_finite_count"] = finite
                metrics["logabs_total_count"] = total
                metrics["logabs_nonfinite_fraction"] = logabs_fraction
                if logabs_fraction > self.max_nonfinite_logabs_fraction:
                    failures.append(
                        f"logabs_nonfinite_fraction={logabs_fraction} exceeds "
                        f"max_nonfinite_logabs_fraction={self.max_nonfinite_logabs_fraction}"
                    )
                if self.strict_sign_values:
                    sign_fraction = _sign_invalid_fraction(output.sign)
                    metrics["sign_invalid_fraction"] = sign_fraction
                    if sign_fraction > 0.0:
                        failures.append(f"sign_invalid_fraction={sign_fraction} exceeds 0.0")
                # Schema invariants belong to the typed output object;
                # DataIntegrity only decides when to check and whether to fail.
                validate = getattr(output, "validate", None)
                if not callable(validate):
                    metrics["output_validated"] = False
                    failures.append(
                        f"wavefunction output type {type(output).__name__} does not expose validate()"
                    )
                else:
                    kwargs: dict[str, Any] = {}
                    batch = state.batch
                    if batch is not None:
                        sample_shape = getattr(batch, "sample_shape", None)
                        batch_size = getattr(batch, "batch_size", None)
                        if sample_shape is not None:
                            kwargs["sample_shape"] = tuple(sample_shape)
                        if batch_size is not None:
                            kwargs["batch_size"] = int(batch_size)
                    try:
                        validate(**kwargs)
                    except Exception as exc:
                        metrics["output_validated"] = False
                        failures.append(
                            f"WavefunctionOutput.validate() failed with {type(exc).__name__}: {exc}"
                        )
                    else:
                        metrics["output_validated"] = True

        if self.check_loss:
            loss = state.loss
            if loss is not None:
                torch = require_torch(feature="DataIntegrity callback")
                loss_is_finite = bool(torch.isfinite(loss).all().item())
                metrics["loss_is_finite"] = loss_is_finite
                if not loss_is_finite:
                    failures.append("loss is not finite")

        if self.check_batch:
            batch = state.batch
            if batch is not None:
                validate = getattr(batch, "validate", None)
                if not callable(validate):
                    metrics["batch_validated"] = False
                    failures.append(f"batch type {type(batch).__name__} does not expose validate()")
                else:
                    try:
                        validate()
                    except Exception as exc:
                        metrics["batch_validated"] = False
                        failures.append(f"batch.validate() failed with {type(exc).__name__}: {exc}")
                    else:
                        metrics["batch_validated"] = True

                validity_metrics = getattr(batch, "validity_metrics", None)
                if callable(validity_metrics):
                    for key, value in validity_metrics().items():
                        metrics[f"batch_{key}"] = value

        passed = not failures
        metrics["passed"] = passed
        context.log(metrics, step=step, namespace="checks/data_integrity")
        if self.fail_fast and not passed:
            raise RuntimeError(f"DataIntegrity failed at step {step}: {failures[0]}")


def _finite_counts(tensor: object) -> tuple[int, int]:
    """Return ``(finite_count, total_count)`` for `tensor`."""

    torch = require_torch(feature="DataIntegrity callback")
    total = int(tensor.numel())
    finite = int(torch.isfinite(tensor).sum().item()) if total else 0
    return finite, total


def _nonfinite_fraction(finite: int, total: int) -> float:
    """Return the non-finite fraction; an empty tensor (``total == 0``) is invalid (1.0).

    Empty and fully-nonfinite tensors share the fraction 1.0; the paired finite/
    total counts logged alongside disambiguate the two cases.
    """

    return float((total - finite) / total) if total > 0 else 1.0


def _sign_invalid_fraction(sign: object) -> float:
    """Return the fraction of sign entries not in the exact set ``{-1, 0, 1}``.

    Wavefunction signs are treated as semantic/discrete (real tensors), so the
    check is exact rather than tolerant. An empty tensor is invalid (1.0).
    """

    torch = require_torch(feature="DataIntegrity callback")
    n = int(sign.numel())
    if n == 0:
        return 1.0
    valid = torch.isfinite(sign) & ((sign == -1) | (sign == 0) | (sign == 1))
    return float(int((~valid).sum().item()) / n)



__all__ = ["DataIntegrity"]
