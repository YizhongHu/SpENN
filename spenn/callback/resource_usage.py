"""Best-effort peak-memory resource callback."""

from __future__ import annotations

import resource
import sys
from collections.abc import Iterable
from typing import Any, Callable

from spenn.dependencies import OptionalDependencyError, require_torch

from .base import Callback, Event, _attach_event_metrics

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

    CUDA peak counters are reset at ``run_start`` so the logged peaks cover
    exactly this run. Readings are best-effort runtime metadata: a failing
    reader omits its metrics instead of failing the run.

    Parameters
    ----------
    triggers : iterable of str, optional
        Event names that should trigger this callback.
    peak_rss_mb_reader : callable, optional
        Override returning the process peak RSS in MiB, for tests.
    """

    def __init__(
        self,
        triggers: Iterable[str] = ("run_start", "run_end", "exception", "run_failed"),
        *,
        peak_rss_mb_reader: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.peak_rss_mb_reader = (
            _default_peak_rss_mb if peak_rss_mb_reader is None else peak_rss_mb_reader
        )

    def on_run_start(self, event: Event) -> None:
        """Reset CUDA peak-memory counters so logged peaks are run-scoped."""

        cuda = _available_cuda()
        if cuda is not None:
            cuda.reset_peak_memory_stats()

    def on_run_end(self, event: Event) -> None:
        """Log peak memory at normal completion."""

        self._log_peaks(event)

    def on_exception(self, event: Event) -> None:
        """Log peak memory on failure without swallowing the exception."""

        self._log_peaks(event)

    def on_run_failed(self, event: Event) -> None:
        """Alias for runtimes that emit ``run_failed``."""

        self._log_peaks(event)

    def _log_peaks(self, event: Event) -> None:
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
            event.context.log(metrics, step=0, namespace="runtime")
            _attach_event_metrics(event, "runtime", metrics)


def _available_cuda() -> Any | None:
    """Return ``torch.cuda`` when torch is importable and CUDA is available."""

    try:
        torch = require_torch(feature="CUDA memory metrics")
    except OptionalDependencyError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


__all__ = ["ResourceUsage"]
