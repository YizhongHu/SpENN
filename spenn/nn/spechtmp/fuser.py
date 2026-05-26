"""Phase-1 Specht fuser placeholder."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from spenn.data_structures.feature_dict import FeatureDict


class SpechtFuser(nn.Module):
    """Placeholder tensor-product fuser for the phase-1 SpechtMP stack.

    The full fuser will consume fixed fusion maps from `spenn.reps` and build
    messages on target subsets. For now, it preserves the public constructor
    shape and returns a cloned `FeatureDict` so downstream phase-1 modules can
    be wired and tested independently.
    """

    def __init__(
        self,
        M: int = 2,
        M_virtual: int = 2,
        channels: Mapping | None = None,
        fixed_maps: object | None = None,
        use_lowrank_virtual: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if M > 2 or M_virtual > 2:
            raise ValueError("Phase 1 SpechtFuser only supports M <= 2 and M_virtual <= 2")
        if use_lowrank_virtual:
            raise ValueError("Low-rank virtual-order approximations are disabled in phase 1")
        self.M = M
        self.M_virtual = M_virtual
        self.channels = channels
        self.fixed_maps = fixed_maps

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Return a cloned feature dictionary until fusion maps are implemented."""

        return features.clone()
