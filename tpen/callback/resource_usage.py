"""Best-effort peak-memory resource callback."""

from __future__ import annotations

import resource
import sys
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.dependencies import OptionalDependencyError, require_torch
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.run_events import RunCompleted, RunFailed, RunStarted

from .base import Callback
from .cadence import SubscriptionGroup

_BYTES_PER_MIB = 1024 * 1024


def _default_peak_rss_mb() -> float:
    """Return the process peak resident-set size in MiB.

    ``ru_maxrss`` is reported in kibibytes on Linux and in bytes on macOS.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak) / _BYTES_PER_MIB
    return float(peak) / 1024.0


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

    The migration also drops the ``_attach_event_metrics`` call that mirrored
    these metrics into the legacy event payload. A typed occurrence has no
    payload, and ``metrics_by_namespace`` has had ZERO readers since PR #181
    moved `tpen.callback.Status` off the legacy ``step_end`` payload, so nothing
    observed it.

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
        self.peak_rss_mb_reader = (
            _default_peak_rss_mb if peak_rss_mb_reader is None else peak_rss_mb_reader
        )

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Reset the CUDA peaks at the start and report them at either end."""

        event = occurrence.event
        if isinstance(event, RunStarted):
            cuda = _available_cuda()
            if cuda is not None:
                cuda.reset_peak_memory_stats()
            return
        # Both terminal boundaries report the same peaks; a failed run has no
        # different memory story to tell, which is why they share one path.
        if isinstance(event, (RunCompleted, RunFailed)):
            self._log_peaks(context)

    def _log_peaks(self, context: RunContext) -> None:
        metrics: dict[str, float] = {}
        try:
            metrics["peak_memory_mb"] = float(self.peak_rss_mb_reader())
        except OSError:
            pass
        cuda = _available_cuda()
        if cuda is not None:
            metrics["cuda_max_memory_allocated_mb"] = float(cuda.max_memory_allocated()) / _BYTES_PER_MIB
            metrics["cuda_max_memory_reserved_mb"] = float(cuda.max_memory_reserved()) / _BYTES_PER_MIB
            metrics["cuda_device_count"] = int(cuda.device_count())
        if metrics:
            context.log(metrics, step=0, namespace="runtime")


def _available_cuda() -> Any | None:
    """Return ``torch.cuda`` when torch is importable and CUDA is available."""

    try:
        torch = require_torch(feature="CUDA memory metrics")
    except OptionalDependencyError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


__all__ = ["ResourceUsage"]
