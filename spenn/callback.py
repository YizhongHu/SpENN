"""Callback primitives for configured SpENN runs."""

from __future__ import annotations

import logging
import json
import os
import random
import socket
import sys
import time
import warnings
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf

from spenn.artifacts import RunContext, write_json
from spenn.naming import camel_to_snake


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
    """Write lifecycle status artifacts and terminal status lines."""

    def __init__(
        self,
        triggers: Iterable[str],
        output_path: str | Path | None = None,
        *,
        terminal: bool = True,
        logger_name: str = "spenn.status",
        include: Sequence[str] | None = None,
        color: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = None if output_path is None else Path(output_path)
        self.terminal = bool(terminal)
        self.logger = logging.getLogger(logger_name)
        self.include = tuple(_DEFAULT_STATUS_METRICS if include is None else include)
        self.color = _validate_terminal_choice(color, name="color")
        self.start_time: str | None = None

    def on_run_start(self, event: Event) -> None:
        """Record run start."""

        self.start_time = _now()
        self._log_status(_format_run_start(event), kind="run")
        self._write(
            status="running",
            current_event=event.name,
            end_time=None,
            exception_type=None,
            exception_message=None,
        )

    def on_run_end(self, event: Event) -> None:
        """Record successful completion."""

        self._log_status(_format_run_end(event), kind="completed")
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
        self._log_status(_format_run_failure(event, exception), kind="failed")
        self._write(
            status="failed",
            current_event=event.name,
            end_time=_now(),
            exception_type=None if exception is None else type(exception).__name__,
            exception_message=None if exception is None else str(exception),
        )

    def on_step_end(self, event: Event) -> None:
        """Write one compact training status line."""

        line = _format_train_status(event, self.include)
        if line is not None:
            self._log_status(line, kind="train")

    def on_evaluate_end(self, event: Event) -> None:
        """Write one compact evaluation status line."""

        line = _format_evaluate_status(event)
        if line is not None:
            self._log_status(line, kind="eval")

    def _log_status(self, line: str, *, kind: str) -> None:
        if not self.terminal:
            return
        self.logger.info(_color_status_line(line, kind=kind, color=self.color))

    def _write(
        self,
        *,
        status: str,
        current_event: str,
        end_time: str | None,
        exception_type: str | None,
        exception_message: str | None,
    ) -> None:
        if self.output_path is None:
            return
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


def configure_terminal_logging(
    *,
    enabled: bool = True,
    level: str = "info",
    color: str = "auto",
    logger_name: str = "spenn",
) -> None:
    """Configure the package terminal logging channel.

    Parameters
    ----------
    enabled : bool, optional
        If ``False``, leave logging configuration unchanged.
    level : str, optional
        Logging level name.
    color : {"auto", "always", "never"}, optional
        Accepted for config validation and consistency with `Status`.
    logger_name : str, optional
        Logger subtree to configure.
    """

    if not enabled:
        return
    _validate_terminal_choice(color, name="color")
    logger = logging.getLogger(logger_name)
    logger.setLevel(_logging_level(level))
    for handler in logger.handlers:
        if getattr(handler, "_spenn_terminal_handler", False):
            handler.setLevel(_logging_level(level))
            return
    handler = logging.StreamHandler()
    handler._spenn_terminal_handler = True
    handler.setLevel(_logging_level(level))
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


class RunTiming(Callback):
    """Measure whole-run timestamps and wall-clock duration."""

    def __init__(
        self,
        triggers: Iterable[str] = ("run_start", "run_end", "exception", "run_failed"),
        *,
        log_start_end_timestamps: bool = True,
        log_wall_time: bool = True,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.log_start_end_timestamps = bool(log_start_end_timestamps)
        self.log_wall_time = bool(log_wall_time)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self.wall_clock = time.time if wall_clock is None else wall_clock
        self._start_perf: float | None = None

    def on_run_start(self, event: Event) -> None:
        """Record run start timing."""

        _sync_cuda(self.cuda_synchronize)
        self._start_perf = self.clock()
        if self.log_start_end_timestamps:
            event.context.log({"start_time_unix": self.wall_clock()}, step=0, namespace="runtime")

    def on_run_end(self, event: Event) -> None:
        """Log whole-run elapsed time at normal completion."""

        self._log_end(event, failed=False)

    def on_exception(self, event: Event) -> None:
        """Log whole-run elapsed time on failure without swallowing the exception."""

        self._log_end(event, failed=True)

    def on_run_failed(self, event: Event) -> None:
        """Alias for runtimes that emit ``run_failed``."""

        self._log_end(event, failed=True)

    def _log_end(self, event: Event, *, failed: bool) -> None:
        _sync_cuda(self.cuda_synchronize)
        now = self.clock()
        metrics: dict[str, float | bool] = {}
        if self.log_start_end_timestamps:
            metrics["end_time_unix"] = self.wall_clock()
        if self.log_wall_time and self._start_perf is not None:
            metrics["wall_time_sec"] = now - self._start_perf
        if failed:
            metrics["failed"] = True
        if metrics:
            event.context.log(metrics, step=0, namespace="runtime")
            _attach_event_metrics(event, "runtime", metrics)


class TrainStepTiming(Callback):
    """Measure training step durations."""

    def __init__(
        self,
        triggers: Iterable[str] = ("step_start", "step_end"),
        *,
        rolling_window: int = 20,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        if rolling_window <= 0:
            raise ValueError(f"rolling_window must be positive, got {rolling_window}")
        self.rolling_window = int(rolling_window)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._starts: dict[int, float] = {}
        self._durations: deque[float] = deque(maxlen=self.rolling_window)

    def on_step_start(self, event: Event) -> None:
        """Record the start time for one training step."""

        step = event.step
        if step is None:
            return
        _sync_cuda(self.cuda_synchronize)
        self._starts[int(step)] = self.clock()

    def on_step_end(self, event: Event) -> None:
        """Log step duration and rolling mean."""

        step = event.step
        if step is None or int(step) not in self._starts:
            return
        _sync_cuda(self.cuda_synchronize)
        duration = self.clock() - self._starts.pop(int(step))
        self._durations.append(duration)
        metrics = {
            "step_time_sec": duration,
            "step_time_sec_rolling_mean": sum(self._durations) / len(self._durations),
        }
        event.context.log(metrics, step=int(step), namespace="train/perf")
        _attach_event_metrics(event, "train/perf", metrics)


class EvaluationTiming(Callback):
    """Measure evaluation wall time."""

    def __init__(
        self,
        triggers: Iterable[str] = ("evaluate_start", "evaluate_end", "eval_start", "eval_end", "exception", "eval_failed"),
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._start: float | None = None

    def on_evaluate_start(self, event: Event) -> None:
        """Record evaluation start."""

        self._start_timing()

    def on_eval_start(self, event: Event) -> None:
        """Alias for ``evaluate_start``."""

        self._start_timing()

    def on_evaluate_end(self, event: Event) -> None:
        """Log evaluation duration."""

        self._log_end(event, failed=False)

    def on_eval_end(self, event: Event) -> None:
        """Alias for ``evaluate_end``."""

        self._log_end(event, failed=False)

    def on_exception(self, event: Event) -> None:
        """Log elapsed evaluation time on failure when evaluation had started."""

        self._log_end(event, failed=True)

    def on_eval_failed(self, event: Event) -> None:
        """Alias for failed evaluation events."""

        self._log_end(event, failed=True)

    def _start_timing(self) -> None:
        _sync_cuda(self.cuda_synchronize)
        self._start = self.clock()

    def _log_end(self, event: Event, *, failed: bool) -> None:
        if self._start is None:
            return
        _sync_cuda(self.cuda_synchronize)
        metrics: dict[str, float | bool] = {"wall_time_sec": self.clock() - self._start}
        if failed:
            metrics["failed"] = True
        step = 0 if event.step is None else int(event.step)
        event.context.log(metrics, step=step, namespace="eval/perf")
        _attach_event_metrics(event, "eval/perf", metrics)
        self._start = None


class DiagnosticTiming(Callback):
    """Measure per-diagnostic evaluation durations."""

    def __init__(
        self,
        triggers: Iterable[str] = ("diagnostic_start", "diagnostic_end", "diagnostic_failed"),
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._starts: dict[tuple[int, str], float] = {}

    def on_diagnostic_start(self, event: Event) -> None:
        """Record one diagnostic start time."""

        key = self._event_key(event)
        _sync_cuda(self.cuda_synchronize)
        self._starts[key] = self.clock()

    def on_diagnostic_end(self, event: Event) -> None:
        """Log one diagnostic duration."""

        self._log_end(event, failed=False)

    def on_diagnostic_failed(self, event: Event) -> None:
        """Log one diagnostic failure duration when possible."""

        self._log_end(event, failed=True)

    def _log_end(self, event: Event, *, failed: bool) -> None:
        key = self._event_key(event)
        if key not in self._starts:
            return
        _sync_cuda(self.cuda_synchronize)
        duration = self.clock() - self._starts.pop(key)
        metrics: dict[str, float | bool] = {"time_sec": duration}
        if failed:
            metrics["failed"] = True
        step, diagnostic_name = key
        namespace = f"diagnostics/{diagnostic_name}"
        event.context.log(metrics, step=step, namespace=namespace)
        _attach_event_metrics(event, namespace, metrics)

    def _event_key(self, event: Event) -> tuple[int, str]:
        name = event.payload.get("diagnostic_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("diagnostic timing events require a non-empty diagnostic_name payload")
        step = 0 if event.step is None else int(event.step)
        return step, name


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
        sampler_mcmc_state = getattr(sampler, "mcmc_state_dict", None)
        payload = {
            "step": state.step,
            "model_state_dict": state.model.state_dict(),
            "optimizer_state_dict": state.optimizer.state_dict(),
            "sampler_mcmc_state": sampler_mcmc_state() if callable(sampler_mcmc_state) else None,
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

    Reads ``state.sampler_stats`` and logs only the stats actually available. When
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
    """Schedule one or more runtime equivariance checkers.

    The callback owns *when* to run (triggers, ``every_n_steps``,
    ``probability``), assigns each checker a stable log name, logs its metrics
    under ``checks/equivariance/<name>``, persists any failure artifact under
    ``artifact_dir``, and raises in ``fail_fast`` mode. Each injected checker
    owns *how* to check (permutation selection, comparison) and returns a
    `spenn.equivariance.checks.EquivarianceCheckResult`.

    The callback ``seed`` controls probabilistic *scheduling*; each checker's own
    ``seed`` controls *which permutations* it selects — separate random streams.

    Parameters
    ----------
    triggers : iterable of str
        Event names that trigger the check (typically ``step_end``).
    checkers : sequence
        Checker objects exposing ``run(state) -> EquivarianceCheckResult``.
    fail_fast : bool, optional
        Raise when any checker fails.
    artifact_dir : str or pathlib.Path or None, optional
        Root directory for failure artifacts. When ``None``, artifacts are not
        written even if a checker returns one.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        checkers: "Sequence[Any]",
        fail_fast: bool = True,
        artifact_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.checkers = list(checkers)
        self.fail_fast = bool(fail_fast)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._checker_log_names = _assign_checker_log_names(self.checkers)

    def on_step_end(self, event: Event) -> None:
        """Run every checker against the current state and log/persist results."""

        state = event.state
        for checker, log_name in zip(self.checkers, self._checker_log_names, strict=True):
            result = checker.run(state)

            metrics = dict(result.metrics)
            metrics["passed"] = bool(result.passed)
            metrics["checker_class"] = type(checker).__name__

            if result.artifact is not None and self.artifact_dir is not None:
                artifact_path = _write_equivariance_artifact(
                    root=self.artifact_dir,
                    checker_name=log_name,
                    step=state.step,
                    artifact=result.artifact,
                )
                metrics["artifact_path"] = str(artifact_path)

            event.context.log(metrics, step=state.step, namespace=f"checks/equivariance/{log_name}")

            if self.fail_fast and not result.passed:
                raise RuntimeError(
                    f"RuntimeEquivariance checker {log_name!r} failed at step {state.step}: "
                    f"{result.failures or metrics}"
                )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sync_cuda(cuda_synchronize: bool) -> None:
    if cuda_synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()


def _attach_event_metrics(event: Event, namespace: str, metrics: Mapping[str, object]) -> None:
    by_namespace = event.payload.setdefault("metrics_by_namespace", {})
    if not isinstance(by_namespace, dict):
        return
    existing = by_namespace.setdefault(namespace, {})
    if isinstance(existing, dict):
        existing.update(metrics)


_DEFAULT_STATUS_METRICS = (
    "train/loss",
    "train/energy",
    "train/energy_stderr",
    "train/sampler/acceptance_rate",
    "train/grad_norm",
    "train/local_energy_finite_fraction",
    "train/perf/step_time_sec",
    "train/perf/step_time_sec_rolling_mean",
)

_STATUS_LABELS = {
    "train/loss": "loss",
    "train/energy": "energy",
    "train/energy_stderr": "stderr",
    "train/sampler/acceptance_rate": "acc",
    "train/grad_norm": "grad",
    "train/local_energy_finite_fraction": "finite",
    "train/perf/step_time_sec": "step_time",
    "train/perf/step_time_sec_rolling_mean": "step_avg",
}

_STATUS_COLORS = {
    "run": "\033[36m",
    "train": "\033[34m",
    "eval": "\033[35m",
    "completed": "\033[32m",
    "failed": "\033[31m",
}


def _format_run_start(event: Event) -> str:
    metadata = event.context.metadata
    parts = [
        "[run] started",
        f"id={metadata.run_id}",
        f"dir={metadata.run_dir}",
        f"device={metadata.device}",
        f"dtype={metadata.dtype}",
    ]
    if metadata.git_commit:
        parts.append(f"git={metadata.git_commit[:7]}")
    parts.append(f"dirty={str(metadata.dirty_worktree).lower()}")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        parts.append(f"slurm_job_id={slurm_job_id}")
    parts.append(f"host={socket.gethostname()}")
    return " ".join(parts)


def _format_run_end(event: Event) -> str:
    return f"[run] completed dir={event.context.metadata.run_dir}"


def _format_run_failure(event: Event, exception: object | None) -> str:
    parts = ["[run] failed", f"dir={event.context.metadata.run_dir}"]
    if exception is not None:
        parts.extend([f"exception={type(exception).__name__}", f"message={_quote_value(str(exception))}"])
    return " ".join(parts)


def _format_train_status(event: Event, include: Sequence[str]) -> str | None:
    state = event.state
    if state is None:
        return None
    values = _training_metric_values(state)
    values.update(_payload_metric_values(event))
    rendered = [
        f"{_STATUS_LABELS.get(identity, identity)}={_format_status_value(values[identity])}"
        for identity in include
        if identity in values
    ]
    if not rendered:
        return None
    step = event.step
    prefix = "[train]" if step is None else f"[train] step={step}"
    return " ".join([prefix, *rendered])


def _format_evaluate_status(event: Event) -> str | None:
    metrics = event.payload.get("metrics")
    values = {}
    if isinstance(metrics, Mapping):
        values.update({f"eval/{key}": value for key, value in metrics.items()})
    values.update(_payload_metric_values(event))
    if not values:
        return None
    include = ("eval/energy", "eval/energy_stderr", "eval/energy_error", "eval/perf/wall_time_sec")
    labels = {
        "eval/energy": "energy",
        "eval/energy_stderr": "stderr",
        "eval/energy_error": "abs_error",
        "eval/perf/wall_time_sec": "wall_time",
    }
    rendered = [
        f"{labels[identity]}={_format_status_value(values[identity])}"
        for identity in include
        if identity in values
    ]
    if not rendered:
        return None
    return " ".join(["[eval]", *rendered])


def _training_metric_values(state: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in dict(getattr(state, "metrics", {}) or {}).items():
        values[f"train/{key}"] = value
    for key, value in dict(getattr(state, "sampler_stats", {}) or {}).items():
        values[f"train/sampler/{key}"] = value
    return values


def _payload_metric_values(event: Event) -> dict[str, object]:
    values: dict[str, object] = {}
    by_namespace = event.payload.get("metrics_by_namespace")
    if not isinstance(by_namespace, Mapping):
        return values
    for namespace, metrics in by_namespace.items():
        if not isinstance(namespace, str) or not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            values[f"{namespace}/{key}"] = value
    return values


def _format_status_value(value: object) -> str:
    if isinstance(value, bool):
        return "ok" if value else "failed"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        abs_value = abs(value)
        if 0 < abs_value < 1.0e-3 or abs_value >= 1.0e4:
            return f"{value:.3e}"
        return f"{value:.6g}"
    return _quote_value(str(value)) if _needs_shell_quote(str(value)) else str(value)


def _quote_value(value: str) -> str:
    return json.dumps(value)


def _needs_shell_quote(value: str) -> bool:
    return any(character.isspace() for character in value) or value == ""


def _validate_terminal_choice(value: str, *, name: str) -> str:
    if value not in {"auto", "always", "never"}:
        raise ValueError(f"{name} must be one of 'auto', 'always', or 'never', got {value!r}")
    return value


def _logging_level(level: str) -> int:
    value = getattr(logging, str(level).upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"Unsupported logging level {level!r}")
    return value


def _color_status_line(line: str, *, kind: str, color: str) -> str:
    if not _color_enabled(color):
        return line
    prefix = _STATUS_COLORS.get(kind)
    if prefix is None:
        return line
    return f"{prefix}{line}\033[0m"


def _color_enabled(color: str) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if color == "always":
        return True
    if color == "never":
        return False
    if os.environ.get("SLURM_JOB_ID"):
        return False
    return sys.stderr.isatty()


_DEFAULT_CHECKER_NAMES = {
    "FullModelEquivarianceChecker": "full_model",
    "TraceEquivarianceChecker": "trace",
}


def _checker_base_name(checker: object) -> str:
    """Return a readable base log name for a checker."""

    class_name = type(checker).__name__
    if class_name in _DEFAULT_CHECKER_NAMES:
        return _DEFAULT_CHECKER_NAMES[class_name]
    explicit = getattr(checker, "name", None)
    if explicit:
        return str(explicit)
    snake = camel_to_snake(class_name)
    for suffix in ("_equivariance_checker", "_checker"):
        if snake.endswith(suffix):
            return snake[: -len(suffix)]
    return snake or class_name.lower()


def _assign_checker_log_names(checkers: Sequence[object]) -> list[str]:
    """Assign stable, de-duplicated log names; warn (do not fail) on duplicates.

    The first instance of a base name keeps it; later duplicates get ``_1``,
    ``_2`` suffixes.
    """

    seen: dict[str, int] = {}
    names: list[str] = []
    for checker in checkers:
        base = _checker_base_name(checker)
        count = seen.get(base, 0)
        if count == 0:
            assigned = base
        else:
            assigned = f"{base}_{count}"
            warnings.warn(
                f"RuntimeEquivariance received duplicate checker name {base!r}; "
                f"using {assigned!r} for the duplicate.",
                stacklevel=3,
            )
        seen[base] = count + 1
        names.append(assigned)
    return names


def _write_equivariance_artifact(
    *,
    root: Path,
    checker_name: str,
    step: int,
    artifact: Mapping[str, Any],
) -> Path:
    """Write a failure artifact under ``root/<checker_name>/step_<step>/failure.json``."""

    path = root / checker_name / f"step_{int(step):06d}" / "failure.json"
    write_json(path, dict(artifact))
    return path


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


__all__ = [
    "Callback",
    "Checkpoint",
    "ConfigSnapshot",
    "DataValidity",
    "DiagnosticTiming",
    "Event",
    "EvaluationTiming",
    "GradientStats",
    "Metadata",
    "ResolvedConfigSnapshot",
    "RunTiming",
    "RuntimeEquivariance",
    "SamplerHealth",
    "Status",
    "TrainStepTiming",
    "configure_terminal_logging",
]
