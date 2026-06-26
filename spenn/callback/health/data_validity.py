"""Training data validity health callback."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from spenn.dependencies import require_torch
from ..base import Callback, Event


class DataValidity(Callback):
    """Hard guardrail catching invalid training tensors at ``step_end``.

    Inspects the batch, wavefunction output (``sign``/``logabs``), local energy,
    and loss on the `TrainerState`, logging finite/validity metrics under
    ``checks/data_validity``. In ``fail_fast`` mode a failed required check
    raises a clear `RuntimeError` instead of silently continuing.
    """

    def __init__(
        self,
        triggers: Iterable[str],
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
        super().__init__(triggers, **kwargs)
        self.fail_fast = bool(fail_fast)
        self.max_nonfinite_energy_fraction = float(max_nonfinite_energy_fraction)
        self.max_nonfinite_logabs_fraction = float(max_nonfinite_logabs_fraction)
        self.check_loss = bool(check_loss)
        self.check_wavefunction_output = bool(check_wavefunction_output)
        self.check_batch = bool(check_batch)
        self.strict_sign_values = bool(strict_sign_values)

    def on_step_end(self, event: Event) -> None:
        """Validate the most recent training step's tensors."""

        state = event.state
        metrics: dict[str, Any] = {}
        failures: list[str] = []

        local_energy = getattr(state, "local_energy", None)
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
            output = getattr(state, "wavefunction_output", None)
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
                # DataValidity only decides when to check and whether to fail.
                validate = getattr(output, "validate", None)
                if not callable(validate):
                    metrics["output_validated"] = False
                    failures.append(
                        f"wavefunction output type {type(output).__name__} does not expose validate()"
                    )
                else:
                    kwargs: dict[str, Any] = {}
                    batch = getattr(state, "batch", None)
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
            loss = getattr(state, "loss", None)
            if loss is not None:
                torch = require_torch(feature="DataValidity callback")
                loss_is_finite = bool(torch.isfinite(loss).all().item())
                metrics["loss_is_finite"] = loss_is_finite
                if not loss_is_finite:
                    failures.append("loss is not finite")

        if self.check_batch:
            batch = getattr(state, "batch", None)
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
        event.context.log(metrics, step=state.step, namespace="checks/data_validity")
        if self.fail_fast and not passed:
            raise RuntimeError(f"DataValidity failed at step {state.step}: {failures[0]}")


def _finite_counts(tensor: object) -> tuple[int, int]:
    """Return ``(finite_count, total_count)`` for `tensor`."""

    torch = require_torch(feature="DataValidity callback")
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

    torch = require_torch(feature="DataValidity callback")
    n = int(sign.numel())
    if n == 0:
        return 1.0
    valid = torch.isfinite(sign) & ((sign == -1) | (sign == 0) | (sign == 1))
    return float(int((~valid).sum().item()) / n)



__all__ = ["DataValidity"]
