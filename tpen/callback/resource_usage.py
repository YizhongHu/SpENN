"""Best-effort peak-memory resource callback."""

from __future__ import annotations

from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.dependencies import OptionalDependencyError, require_torch
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.process_resources import (
    ProcessRUsageProbe,
    ProcessResourceResult,
    ResourceUnavailable,
)

from .base import Callback
from .cadence import SubscriptionGroup

_BYTES_PER_MIB = 1024 * 1024


def _default_peak_rss_mb() -> float:
    """Return process peak RSS in MiB for compatibility callers."""

    value = ProcessRUsageProbe().read().peak_rss_mb
    if isinstance(value, ResourceUnavailable):
        raise OSError(value.reason)
    return float(value)


class ResourceUsage(Callback):
    """Log run-level peak process and CUDA memory under ``runtime``.

    CUDA peak counters are reset when the run starts so the logged peaks cover
    exactly this run. Readings are best-effort runtime metadata: a failing
    reader omits its metrics instead of failing the run.

    Data-free, so a plain `tpen.callback.Callback`: it reads a process counter
    and `torch.cuda`, never any domain state.

    Notes
    -----
    On a failed run this used to log ``runtime`` TWICE. Its shipped default
    ``triggers`` answered both ``run_failed`` and ``exception``, which
    `tpen.run` emits on consecutive lines with the same payload, and
    ``on_run_failed`` was an alias calling the same method as ``on_exception``;
    ``pair_stability.yaml`` configured both names explicitly. One typed
    `tpen.run_events.RunFailed` makes the duplicate structurally impossible. No
    metric name changes and no series disappears -- one record per failed run
    replaces two identical ones.

    These metrics are logger-owned; they are not an inter-callback contract.

    Parameters
    ----------
    peak_rss_mb_reader : callable, optional
        Override returning the process peak RSS in MiB, for tests.
    **kwargs
        Forwarded to `tpen.callback.Callback`.
    """

    def __init__(
        self,
        *,
        peak_rss_mb_reader: Callable[[], float] | None = None,
        process_probe: ProcessRUsageProbe | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    )
                ),
            ),
            **kwargs,
        )
        self.process_probe = ProcessRUsageProbe() if process_probe is None else process_probe
        self.peak_rss_mb_reader = peak_rss_mb_reader
        self._process_baseline = None

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Reset the CUDA peaks at the start and report them at either end."""

        event = occurrence.event
        if isinstance(event, RunStarted):
            self._process_baseline = self.process_probe.read()
            cuda = _available_cuda()
            if cuda is not None:
                cuda.reset_peak_memory_stats()
            return
        # Both terminal boundaries report the same peaks; a failed run has no
        # different memory story to tell, which is why they share one path.
        if isinstance(event, (RunCompleted, RunFailed)):
            self._log_peaks(context)

    def _log_peaks(self, context: RunContext) -> None:
        metrics: dict[str, Any] = {}
        process_metrics: dict[str, Any] = {}
        result = None
        if self._process_baseline is not None:
            result = self.process_probe.result(self._process_baseline)
            process_metrics.update(_process_metrics(result))
            if isinstance(result.peak_rss_mb, ResourceUnavailable):
                process_metrics["process_peak_rss_unavailable"] = True
        if self.peak_rss_mb_reader is not None:
            try:
                metrics["peak_memory_mb"] = float(self.peak_rss_mb_reader())
            except OSError:
                pass
        elif result is not None:
            if not isinstance(result.peak_rss_mb, ResourceUnavailable):
                metrics["peak_memory_mb"] = float(result.peak_rss_mb)
        else:
            try:
                metrics["peak_memory_mb"] = float(_default_peak_rss_mb())
            except OSError:
                pass
        cuda = _available_cuda()
        if cuda is not None:
            metrics["cuda_max_memory_allocated_mb"] = float(cuda.max_memory_allocated()) / _BYTES_PER_MIB
            metrics["cuda_max_memory_reserved_mb"] = float(cuda.max_memory_reserved()) / _BYTES_PER_MIB
            metrics["cuda_device_count"] = int(cuda.device_count())
        if metrics:
            context.log(metrics, step=0, namespace="runtime")
        if process_metrics:
            context.log(process_metrics, step=0, namespace="process")


def _available_cuda() -> Any | None:
    """Return ``torch.cuda`` when torch is importable and CUDA is available."""

    try:
        torch = require_torch(feature="CUDA memory metrics")
    except OptionalDependencyError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


def _process_metrics(result: ProcessResourceResult) -> dict[str, Any]:
    """Project typed process results onto the logger's mapping boundary."""

    names = (
        ("process_user_cpu_seconds", result.user_cpu_seconds),
        ("process_system_cpu_seconds", result.system_cpu_seconds),
        ("process_read_block_operations", result.read_block_operations),
        ("process_write_block_operations", result.write_block_operations),
        ("process_voluntary_context_switches", result.voluntary_context_switches),
        ("process_involuntary_context_switches", result.involuntary_context_switches),
    )
    metrics: dict[str, Any] = {}
    for name, value in names:
        if isinstance(value, ResourceUnavailable):
            metrics[f"{name}_unavailable"] = True
        else:
            metrics[name] = float(value)
    return metrics


__all__ = ["ResourceUsage"]
