"""Shared helpers for timing callbacks."""

from __future__ import annotations

from spenn.dependencies import require_torch

from ..base import Callback, Event, _attach_event_metrics


def _sync_cuda(cuda_synchronize: bool) -> None:
    """Synchronize CUDA for benchmark timing when explicitly requested."""

    if not cuda_synchronize:
        return
    torch = require_torch(feature="CUDA timing synchronization")
    if torch.cuda.is_available():
        torch.cuda.synchronize()


__all__ = ["Callback", "Event", "_attach_event_metrics", "_sync_cuda"]
