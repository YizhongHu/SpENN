"""Shared helpers for timing callbacks."""

from __future__ import annotations

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


__all__ = ["Callback", "_sync_device"]
