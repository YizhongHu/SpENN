"""Base encoder interface."""

from __future__ import annotations

from torch import nn

from spenn.data_structures.batch import ElectronBatch
from spenn.data_structures.feature_dict import FeatureDict


class BaseEncoder(nn.Module):
    """Common encoder interface."""

    def forward(self, batch: ElectronBatch) -> FeatureDict:  # pragma: no cover - abstract
        raise NotImplementedError
