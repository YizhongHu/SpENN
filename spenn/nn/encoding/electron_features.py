"""One-body and pair electron feature encoders."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from spenn.data_structures.batch import ElectronBatch
from spenn.data_structures.feature_dict import FeatureDict
from spenn.data_structures.partitions import PartitionLike, normalize_partition
from spenn.nn.encoding.base import BaseEncoder
from spenn.nn.encoding.distance_features import (
    one_body_raw_features,
    pair_antisymmetric_raw_features,
    pair_symmetric_raw_features,
)
from spenn.utils.tensor_utils import antisymmetrize_pair_tensor, symmetrize_pair_tensor


def _parse_channel_spec(channels: Mapping | None, order: int, irrep: PartitionLike, default: int) -> int:
    if channels is None:
        return default
    partition = normalize_partition(order, irrep)
    key = f"order{order}"
    block = channels.get(key, channels.get(order, {}))
    if not isinstance(block, Mapping):
        return int(block)
    for candidate, value in block.items():
        candidate_partition = normalize_partition(order, candidate)
        if candidate_partition == partition:
            return int(value)
    return default


class ElectronPairEncoder(BaseEncoder):
    """Phase 1 encoder with order-1 and order-2 feature blocks."""

    def __init__(
        self,
        max_order: int = 2,
        channels: Mapping | None = None,
        include_spins: bool = True,
        name: str = "basic",
        **_: object,
    ) -> None:
        super().__init__()
        if max_order > 2:
            raise ValueError("Phase 1 encoder only supports max_order <= 2")
        self.max_order = max_order
        self.channels = channels
        self.include_spins = include_spins
        self.name = name

        self.order1_channels = _parse_channel_spec(channels, 1, (1,), 32)
        self.order2_s_channels = _parse_channel_spec(channels, 2, (2,), 32)
        self.order2_a_channels = _parse_channel_spec(channels, 2, (1, 1), 32)

        self.order1_proj = nn.LazyLinear(self.order1_channels, dtype=torch.float64)
        self.order2_s_proj = nn.LazyLinear(self.order2_s_channels, dtype=torch.float64) if self.order2_s_channels > 0 else None
        self.order2_a_proj = (
            nn.LazyLinear(self.order2_a_channels, bias=False, dtype=torch.float64) if self.order2_a_channels > 0 else None
        )

    def forward(self, batch: ElectronBatch) -> FeatureDict:
        features = FeatureDict()
        order1_raw = one_body_raw_features(batch, include_spins=self.include_spins)
        features.set(1, (1), self.order1_proj(order1_raw))
        if self.max_order >= 2:
            pair_s_raw = pair_symmetric_raw_features(batch, include_spins=self.include_spins)
            pair_a_raw = pair_antisymmetric_raw_features(batch, include_spins=self.include_spins)
            if self.order2_s_proj is not None:
                pair_s = symmetrize_pair_tensor(self.order2_s_proj(pair_s_raw))
                features.set(2, (2), pair_s)
            if self.order2_a_proj is not None:
                pair_a = antisymmetrize_pair_tensor(self.order2_a_proj(pair_a_raw))
                features.set(2, (1, 1), pair_a)
        features.validate(batch_size=batch.batch_size, n_electrons=batch.n_electrons)
        return features
