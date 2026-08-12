"""Shared helpers for timing callbacks."""

from __future__ import annotations

from tpen.accelerator import synchronize as _synchronize_accelerator

from ..base import Callback


def _sync_device(cuda_synchronize: bool) -> None:
    """Synchronize the accelerator for benchmark timing when explicitly requested.

    Parameters
    ----------
    cuda_synchronize : bool
        Whether to synchronize. The parameter keeps its published name; it is a
        public callback option and renaming it is a separate, deliberate change.
    """

    if not cuda_synchronize:
        return
    _synchronize_accelerator(feature="device timing synchronization")


__all__ = ["Callback", "_sync_device"]
