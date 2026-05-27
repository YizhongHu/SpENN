"""Channel and multiplicity mixing layers."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from spenn.data.feature_dict import FeatureDict
from spenn.data.partitions import Partition


class FeatureChannelMixer(nn.Module):
    """Apply a learned channel mixing to every stored feature tensor."""

    def __init__(self, channels: Mapping[int, Mapping[Partition, int]] | None = None) -> None:
        super().__init__()
        self.channels = channels or {}
        self.mixers = nn.ModuleDict()

    def _key(self, order: int, irrep: Partition) -> str:
        return f"{order}:{irrep}"

    def forward(self, features: FeatureDict) -> FeatureDict:
        return features.clone()
