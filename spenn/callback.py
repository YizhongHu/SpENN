"""Callback primitives for configured SpENN runs."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import torch
from omegaconf import OmegaConf

from spenn.artifacts import RunContext, write_json


@dataclass
class Event:
    """Lifecycle event delivered to callbacks."""

    name: str
    context: RunContext
    state: object | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def step(self) -> int | None:
        """Return the event step when available."""

        if "step" in self.payload:
            return None if self.payload["step"] is None else int(self.payload["step"])
        value = getattr(self.state, "global_step", None)
        return None if value is None else int(value)


class Callback:
    """Base class for event-triggered run callbacks.

    Parameters
    ----------
    triggers : iterable of str
        Event names that should trigger this callback.
    every_n_steps : int or None, optional
        Optional periodic step filter.
    start_step : int, optional
        First eligible step for periodic callbacks.
    max_calls : int or None, optional
        Maximum number of callback invocations (counts actual executions).
    probability : float, optional
        Probability of running when otherwise scheduled. ``1.0`` always runs,
        ``0.0`` never runs. Applied after the trigger/``every_n_steps``/
        ``start_step`` checks.
    seed : int or None, optional
        Seed for the callback-local RNG used by `probability`. Using a local
        RNG keeps probabilistic scheduling reproducible without perturbing
        global PyTorch randomness.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        every_n_steps: int | None = None,
        start_step: int = 0,
        max_calls: int | None = None,
        probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        self.triggers = tuple(triggers)
        self.every_n_steps = every_n_steps
        self.start_step = int(start_step)
        self.max_calls = max_calls
        self.probability = float(probability)
        self.seed = seed
        self._rng = random.Random(seed)
        self.num_calls = 0

    def should_run(self, event: Event) -> bool:
        """Return whether this callback should handle `event`."""

        if event.name not in self.triggers:
            return False
        if self.max_calls is not None and self.num_calls >= self.max_calls:
            return False
        if self.every_n_steps is not None:
            step = event.step
            if step is None or step < self.start_step:
                return False
            if (step - self.start_step) % self.every_n_steps != 0:
                return False
        return self._draw_probability()

    def _draw_probability(self) -> bool:
        """Apply the probability gate using the callback-local RNG."""

        if self.probability >= 1.0:
            return True
        if self.probability <= 0.0:
            return False
        return self._rng.random() < self.probability

    def handle(self, event: Event) -> None:
        """Handle an event if this callback is subscribed to it."""

        if not self.should_run(event):
            return
        method = getattr(self, f"on_{event.name}", None)
        if method is not None:
            method(event)
        self.num_calls += 1


class ConfigSnapshot(Callback):
    """Write a re-runnable run configuration at run start."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Write ``config.yaml``."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(event.context.source_cfg, self.output_path, resolve=False)


class ResolvedConfigSnapshot(Callback):
    """Write the fully resolved run configuration at run start."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Write ``resolved_config.yaml``."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(event.context.cfg, self.output_path, resolve=True)


class Metadata(Callback):
    """Write run metadata during lifecycle transitions."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Record running metadata."""

        self._write(event, status="running")

    def on_run_end(self, event: Event) -> None:
        """Record completed metadata."""

        self._write(event, status="completed")

    def on_exception(self, event: Event) -> None:
        """Record failed metadata."""

        self._write(event, status="failed")

    def _write(self, event: Event, *, status: str) -> None:
        metadata = event.context.metadata
        metadata.status = status
        data = metadata.to_dict()
        data["status"] = status
        exception = event.payload.get("exception")
        if exception is not None:
            data["exception_type"] = type(exception).__name__
            data["exception_message"] = str(exception)
        write_json(self.output_path, data)


class Status(Callback):
    """Write lifecycle status for one run."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)
        self.start_time: str | None = None

    def on_run_start(self, event: Event) -> None:
        """Record run start."""

        self.start_time = _now()
        self._write(
            status="running",
            current_event=event.name,
            end_time=None,
            exception_type=None,
            exception_message=None,
        )

    def on_run_end(self, event: Event) -> None:
        """Record successful completion."""

        self._write(
            status="completed",
            current_event=event.name,
            end_time=_now(),
            exception_type=None,
            exception_message=None,
        )

    def on_exception(self, event: Event) -> None:
        """Record run failure."""

        exception = event.payload.get("exception")
        self._write(
            status="failed",
            current_event=event.name,
            end_time=_now(),
            exception_type=None if exception is None else type(exception).__name__,
            exception_message=None if exception is None else str(exception),
        )

    def _write(
        self,
        *,
        status: str,
        current_event: str,
        end_time: str | None,
        exception_type: str | None,
        exception_message: str | None,
    ) -> None:
        write_json(
            self.output_path,
            {
                "status": status,
                "start_time": self.start_time,
                "end_time": end_time,
                "current_event": current_event,
                "exception_type": exception_type,
                "exception_message": exception_message,
            },
        )


class ReportSkeleton(Callback):
    """Write a human-readable scaffold report."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_end(self, event: Event) -> None:
        """Write ``report.md`` for a scaffold run."""

        run_id = event.context.metadata.run_id
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            "\n".join(
                [
                    "# SpENN Scaffold Run",
                    "",
                    f"- Run ID: `{run_id}`",
                    f"- Run directory: `{event.context.run_dir}`",
                    "- Status: completed",
                    "",
                    "This scaffold run only exercised generic run management.",
                    "No Hooke physics, VMC training, diagnostics, sampling, or plotting were run.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


class Checkpoint(Callback):
    """Write training checkpoints from the loop `TrainerState`.

    Reads ``event.state`` (a `spenn.training.state.TrainerState`) and writes a
    ``torch.save`` payload to ``output_dir/step_<step>.pt`` and
    ``output_dir/latest.pt``. PR3 only writes checkpoints; it does not resume.

    Parameters
    ----------
    triggers : iterable of str
        Event names that should trigger checkpointing (typically ``step_end``).
    output_dir : str or pathlib.Path
        Directory into which checkpoints are written.
    **kwargs
        Forwarded to `Callback` (e.g. ``every_n_steps``).
    """

    def __init__(self, triggers: Iterable[str], output_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_dir = Path(output_dir)

    def on_step_end(self, event: Event) -> None:
        """Write the current step's checkpoint."""

        import torch

        state = event.state
        sampler = getattr(state, "sampler", None)
        payload = {
            "step": state.step,
            "model_state_dict": state.model.state_dict(),
            "optimizer_state_dict": state.optimizer.state_dict(),
            "sampler_state_dict": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
            "metrics": state.metrics,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.output_dir / f"step_{state.step}.pt")
        torch.save(payload, self.output_dir / "latest.pt")


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
        check_batch_tensors: bool = True,
        strict_sign_values: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.fail_fast = bool(fail_fast)
        self.max_nonfinite_energy_fraction = float(max_nonfinite_energy_fraction)
        self.max_nonfinite_logabs_fraction = float(max_nonfinite_logabs_fraction)
        self.check_loss = bool(check_loss)
        self.check_wavefunction_output = bool(check_wavefunction_output)
        self.check_batch_tensors = bool(check_batch_tensors)
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

        if self.check_loss:
            loss = getattr(state, "loss", None)
            if loss is not None:
                loss_is_finite = bool(torch.isfinite(loss).all().item())
                metrics["loss_is_finite"] = loss_is_finite
                if not loss_is_finite:
                    failures.append("loss is not finite")

        if self.check_batch_tensors:
            batch = getattr(state, "batch", None)
            if batch is not None:
                count = _nonfinite_tensor_count(batch)
                metrics["batch_nonfinite_tensor_count"] = count
                if count > 0:
                    failures.append(f"batch_nonfinite_tensor_count={count} exceeds 0")

        passed = not failures
        metrics["passed"] = passed
        event.context.log(metrics, step=state.step, namespace="checks/data_validity")
        if self.fail_fast and not passed:
            raise RuntimeError(f"DataValidity failed at step {state.step}: {failures[0]}")


class GradientStats(Callback):
    """Track gradient health at ``step_end`` (after ``optimizer.step()``).

    Reads parameter gradients from ``state.model`` and logs norm/finite metrics
    under ``checks/gradient``. With ``check_finite`` it fails on non-finite
    gradients, and with ``max_global_grad_norm`` it fails when the global norm
    is exceeded. It does not require convergence or small gradients.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        fail_fast: bool = False,
        max_global_grad_norm: float | None = None,
        check_finite: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.fail_fast = bool(fail_fast)
        self.max_global_grad_norm = None if max_global_grad_norm is None else float(max_global_grad_norm)
        self.check_finite = bool(check_finite)

    def on_step_end(self, event: Event) -> None:
        """Summarize gradients of the model parameters."""

        state = event.state
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
        event.context.log(metrics, step=state.step, namespace="checks/gradient")
        if self.fail_fast and failure is not None:
            raise RuntimeError(f"GradientStats failed at step {state.step}: {failure}")


class SamplerHealth(Callback):
    """Expose sampler statistics under ``checks/sampler`` with optional bounds.

    Reads ``state.sampler_stats`` (falling back to ``sampler.*`` keys in
    ``state.metrics``) and logs only the stats actually available. When
    acceptance-rate bounds are configured and violated, ``passed`` is ``False``;
    it raises only if ``fail_fast`` is set.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        fail_fast: bool = False,
        min_acceptance_rate: float | None = None,
        max_acceptance_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.fail_fast = bool(fail_fast)
        self.min_acceptance_rate = None if min_acceptance_rate is None else float(min_acceptance_rate)
        self.max_acceptance_rate = None if max_acceptance_rate is None else float(max_acceptance_rate)

    def on_step_end(self, event: Event) -> None:
        """Log available sampler diagnostics and check crude bounds."""

        state = event.state
        stats = dict(getattr(state, "sampler_stats", None) or {})
        if not stats:
            source = getattr(state, "metrics", None) or {}
            prefix = "sampler."
            stats = {key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)}

        metrics: dict[str, Any] = {
            key: stats[key] for key in ("acceptance_rate", "n_walkers", "n_steps", "burn_in") if key in stats
        }

        failure: str | None = None
        acceptance_rate = stats.get("acceptance_rate")
        if acceptance_rate is not None:
            if self.min_acceptance_rate is not None and acceptance_rate < self.min_acceptance_rate:
                failure = (
                    f"acceptance_rate={acceptance_rate} below min_acceptance_rate={self.min_acceptance_rate}"
                )
            elif self.max_acceptance_rate is not None and acceptance_rate > self.max_acceptance_rate:
                failure = (
                    f"acceptance_rate={acceptance_rate} above max_acceptance_rate={self.max_acceptance_rate}"
                )

        metrics["passed"] = failure is None
        event.context.log(metrics, step=state.step, namespace="checks/sampler")
        if self.fail_fast and failure is not None:
            raise RuntimeError(f"SamplerHealth failed at step {state.step}: {failure}")


class RuntimeEquivariance(Callback):
    """Schedule a checker-driven runtime equivariance check.

    The callback owns *when* to run (triggers, ``every_n_steps``,
    ``probability``); the injected ``checker`` owns *how* to check and returns a
    `spenn.testing.runtime.CheckResult`. Its metrics are logged under
    ``checks/<result.name>``; a failed result raises only in ``fail_fast`` mode.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        checker: Any,
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.checker = checker
        self.fail_fast = bool(fail_fast)

    def on_step_end(self, event: Event) -> None:
        """Run the checker against the current state and log its result."""

        state = event.state
        result = self.checker.run(state)
        event.context.log(result.metrics, step=state.step, namespace=f"checks/{result.name}")
        if self.fail_fast and not result.passed:
            raise RuntimeError(
                f"RuntimeEquivariance check {result.name!r} failed at step {state.step}: {result.metrics}"
            )


class ReferenceEnergy(Callback):
    """Log reference-energy comparison metrics from training metrics.

    Reads ``state.metrics[source_metric]`` and logs ``reference_energy``,
    ``energy_error``, and ``abs_energy_error`` under ``namespace``. Keeps
    reference comparison an explicit run choice rather than trainer policy.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        reference_energy: float,
        source_metric: str = "energy_mean",
        namespace: str = "reference",
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.reference_energy = float(reference_energy)
        self.source_metric = source_metric
        self.namespace = namespace

    def on_step_end(self, event: Event) -> None:
        """Compute and log reference-energy metrics for the current step."""

        from spenn.physics.hamiltonian import reference_energy_metrics

        state = event.state
        if state is None or state.metrics is None:
            return
        if self.source_metric not in state.metrics:
            raise KeyError(
                f"ReferenceEnergy expected metric {self.source_metric!r} "
                f"in TrainerState.metrics at step {state.step}."
            )
        metrics = reference_energy_metrics(
            energy_mean=state.metrics[self.source_metric],
            reference_energy=self.reference_energy,
        )
        event.context.log(metrics, step=state.step, namespace=self.namespace)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _finite_counts(tensor: torch.Tensor) -> tuple[int, int]:
    """Return ``(finite_count, total_count)`` for `tensor`."""

    total = int(tensor.numel())
    finite = int(torch.isfinite(tensor).sum().item()) if total else 0
    return finite, total


def _nonfinite_fraction(finite: int, total: int) -> float:
    """Return the non-finite fraction; an empty tensor (``total == 0``) is invalid (1.0).

    Empty and fully-nonfinite tensors share the fraction 1.0; the paired finite/
    total counts logged alongside disambiguate the two cases.
    """

    return float((total - finite) / total) if total > 0 else 1.0


def _sign_invalid_fraction(sign: torch.Tensor) -> float:
    """Return the fraction of sign entries not in the exact set ``{-1, 0, 1}``.

    Wavefunction signs are treated as semantic/discrete (real tensors), so the
    check is exact rather than tolerant. An empty tensor is invalid (1.0).
    """

    n = int(sign.numel())
    if n == 0:
        return 1.0
    valid = torch.isfinite(sign) & ((sign == -1) | (sign == 0) | (sign == 1))
    return float(int((~valid).sum().item()) / n)


def _iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor leaf in a dataclass/mapping/sequence/tensor tree."""

    if isinstance(obj, torch.Tensor):
        yield obj
        return
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            yield from _iter_tensors(getattr(obj, f.name))
        return
    if isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_tensors(value)
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for value in obj:
            yield from _iter_tensors(value)


def _nonfinite_tensor_count(obj: Any) -> int:
    """Return the number of tensor leaves that are empty or contain a non-finite value.

    Empty leaves count as invalid, matching `_nonfinite_fraction`'s treatment of
    empty tensors.
    """

    count = 0
    for tensor in _iter_tensors(obj):
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
            count += 1
    return count


__all__ = [
    "Callback",
    "Checkpoint",
    "ConfigSnapshot",
    "DataValidity",
    "Event",
    "GradientStats",
    "Metadata",
    "ReferenceEnergy",
    "ReportSkeleton",
    "ResolvedConfigSnapshot",
    "RuntimeEquivariance",
    "SamplerHealth",
    "Status",
]
