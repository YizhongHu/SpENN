"""Explicit opt-in low-rank virtual-order approximations."""

from __future__ import annotations

from torch import nn


class LowRankVirtualBrancher(nn.Module):
    """Disabled in phase 1; exists only to keep the namespace explicit."""

    def __init__(self, *_, **__) -> None:
        super().__init__()
        raise NotImplementedError("Low-rank virtual-order approximations are not available in phase 1")
