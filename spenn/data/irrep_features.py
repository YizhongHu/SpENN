"""Placeholder containers for temporary irrep-space Specht features.

The real-space restructure keeps persistent state in
:mod:`spenn.data.real_features`. This module is reserved for temporary
irrep/Specht-coordinate payloads used inside Fourier, activation, and gated
update wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spenn.data.partitions import Partition


@dataclass(frozen=True)
class IrrepFeature:
    """Store a temporary irrep-space feature block.

    Parameters
    ----------
    partition : Partition
        Specht partition labeling the irrep block.
    tensor : torch.Tensor
        Tensor payload in the local irrep basis.
    """

    partition: Partition
    tensor: torch.Tensor


@dataclass(frozen=True)
class IrrepMessage:
    """Store a temporary irrep-space message block.

    Parameters
    ----------
    partition : Partition
        Specht partition labeling the irrep block.
    tensor : torch.Tensor
        Tensor payload in the local irrep basis.
    """

    partition: Partition
    tensor: torch.Tensor


__all__ = ["IrrepFeature", "IrrepMessage"]
