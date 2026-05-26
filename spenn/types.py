"""Shared type aliases and lightweight protocols for SpENN."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch
    from spenn.data_structures.partitions import Partition
    from torch import Tensor
else:  # pragma: no cover - keep import lightweight when torch is absent
    Partition = Any
    Tensor = Any

TensorDict = dict[str, Tensor]
FeatureKey = tuple[int, Partition]

__all__ = ["FeatureKey", "Tensor", "TensorDict"]
