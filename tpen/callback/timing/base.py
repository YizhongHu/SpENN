"""Shared helpers for timing callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tpen.events import Occurrence

from tpen.accelerator import device_event_timer as _device_event_timer
from tpen.accelerator import synchronize as _synchronize_accelerator

from ..base import Callback


def _sync_device(accelerator_synchronize: bool) -> None:
    """Synchronize the accelerator for benchmark timing when explicitly requested.

    Parameters
    ----------
    accelerator_synchronize : bool
        Whether to synchronize. Renamed from ``cuda_synchronize`` in v0.3.0 — the
        deliberate change commit ``0b4a2cf`` deferred — with no alias for the old
        spelling. The name states a BEHAVIOUR, not which hardware produced a value,
        and the behaviour is device-neutral: the work is delegated to
        `tpen.accelerator.synchronize`, which covers CUDA, ROCm, and XPU alike, so the
        old device-specific spelling was simply wrong. Durable ``cuda_*`` METRIC and
        METADATA keys are the opposite case and deliberately keep their device name,
        because they record which hardware produced a measurement; ADR-E006 keeps such
        durable and external names as strings.
    """

    if not accelerator_synchronize:
        return
    _synchronize_accelerator(feature="device timing synchronization")


class TimingBackend(Protocol):
    """Internal contract for one timing source.

    ``start`` and ``elapsed`` are deliberately tiny so deterministic tests can
    provide a fake clock or device-event source without importing torch.
    Implementations must not synchronize an accelerator from ``start``.
    """

    metric_suffix: str

    def start(self) -> Any:
        """Capture a timing source's start marker."""

    def elapsed(self, marker: Any) -> float:
        """Resolve elapsed seconds from a previously returned marker."""


class HostTimingBackend:
    """Unsynchronized monotonic host timing backend."""

    metric_suffix = ""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock

    def start(self) -> float:
        return float(self.clock())

    def elapsed(self, marker: float) -> float:
        return float(self.clock()) - marker


class DeviceTimingBackend:
    """Optional event-based accelerator backend.

    Event completion is resolved on the event itself when supported. There is
    no device-wide synchronization at scope boundaries; a backend that cannot
    provide event timing fails loudly at construction/use time.
    """

    metric_suffix = "_device"

    def __init__(self, *, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory or (lambda: _device_event_timer(feature="callback device timing"))

    def start(self) -> Any:
        timer = self._factory()
        timer.start()
        return timer

    def elapsed(self, marker: Any) -> float:
        return float(marker.stop())


@dataclass(frozen=True)
class TimingPair:
    """Host measurement plus an optional distinct device measurement."""

    host: float
    device: float | None = None


class TimingSource:
    """Compose the mandatory host source with an optional device source."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        backend: TimingBackend | None = None,
        device_backend: TimingBackend | None = None,
    ) -> None:
        self._uses_emitter_stamp = backend is None
        self.host_backend = backend or HostTimingBackend(clock)
        self.device_backend = device_backend

    def start(self, host_marker: float | None = None) -> tuple[Any, Any | None]:
        return (host_marker if self._uses_emitter_stamp and host_marker is not None else self.host_backend.start()), (
            None if self.device_backend is None else self.device_backend.start()
        )

    def elapsed(
        self, markers: tuple[Any, Any | None], host_value: float | None = None
    ) -> TimingPair:
        start_marker, device_marker = markers
        return TimingPair(
            host=(
                float(self.host_backend.elapsed(start_marker))
                if host_value is None or not self._uses_emitter_stamp
                else float(host_value) - float(start_marker)
            ),
            device=(
                None
                if self.device_backend is None or device_marker is None
                else float(self.device_backend.elapsed(device_marker))
            ),
        )


def _occurrence_time(occurrence: Occurrence[object], clock: Callable[[], float]) -> float:
    """Return an emitter-stamped host boundary time.

    Direct callback tests may construct an occurrence without a `RunContext`;
    the injected clock preserves that narrow testing seam. Runtime delivery
    always supplies the shared stamp captured before persistence and callbacks.
    """

    return clock() if occurrence.monotonic_time is None else occurrence.monotonic_time


__all__ = [
    "Callback",
    "DeviceTimingBackend",
    "HostTimingBackend",
    "TimingBackend",
    "TimingPair",
    "TimingSource",
    "_occurrence_time",
    "_sync_device",
]
