"""Hard-coded Specht brancher module for the phase-1 prototype."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from spenn.data_structures.feature_dict import FeatureDict


class SpechtBrancher(nn.Module):
    """Phase-1 brancher that keeps the public interface but only supports ``M <= 2``."""

    def __init__(
        self,
        M: int = 2,
        M_virtual: int = 2,
        channels: Mapping | None = None,
        branch_maps: object | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if M > 2 or M_virtual > 2:
            raise ValueError("Phase 1 SpechtBrancher only supports M <= 2 and M_virtual <= 2")
        self.M = M
        self.M_virtual = M_virtual
        self.channels = channels
        self.branch_maps = branch_maps

    def forward(self, messages: FeatureDict, residual: FeatureDict | None = None) -> FeatureDict:
        if residual is None:
            return messages.clone()
        combined = residual.clone()
        for order, block in messages.items():
            if order not in combined:
                combined[order] = {}
            for irrep, tensor in block.items():
                combined[order][irrep] = combined[order].get(irrep, 0) + tensor
        return combined
