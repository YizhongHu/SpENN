"""Equivariance-preserving activation modules."""

from __future__ import annotations

from torch import nn

from spenn.data_structures.feature_dict import FeatureDict


class TensorProductActivation(nn.Module):
    """Apply an elementwise activation to every feature tensor."""

    def __init__(self, kind: str = "tanh") -> None:
        super().__init__()
        self.kind = kind
        if kind == "tanh":
            self._activation = nn.Tanh()
        elif kind == "gelu":
            self._activation = nn.GELU()
        elif kind in {"identity", "none"}:
            self._activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation kind: {kind}")

    def forward(self, features: FeatureDict) -> FeatureDict:
        return FeatureDict(
            {
                order: {irrep: self._activation(tensor) for irrep, tensor in block.items()}
                for order, block in features.items()
            }
        )
