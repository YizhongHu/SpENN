"""Reusable real-space update strategy modules."""

from __future__ import annotations

import torch

from spenn.data import RealFeature, RealUpdate
from spenn.nn.equivariant_map import EquivariantMap


class ReplaceUpdate(EquivariantMap):
    """Replace persistent real features with a real update proposal."""

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return the update proposal as the next real feature state."""

        _validate_matching_blocks(x, u)
        return RealFeature([tensor.clone() for tensor in u.blocks])


class ResidualUpdate(EquivariantMap):
    """Add a scaled real update proposal to persistent features."""

    def __init__(self, step: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = float(step)

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return ``x + step * u`` blockwise."""

        _validate_matching_blocks(x, u)
        return RealFeature([left + self.step * right for left, right in zip(x.blocks, u.blocks)])


class NormGatedUpdate(EquivariantMap):
    """Gate a residual update by an equivariant per-tuple update norm."""

    def __init__(self, step: float = 1.0, eps: float = 1.0e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = float(step)
        self.eps = float(eps)

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return a norm-gated residual update."""

        _validate_matching_blocks(x, u)
        output = []
        for feature, update in zip(x.blocks, u.blocks):
            if update.shape[1] == 0:
                output.append(feature.clone())
                continue
            norm = update.square().mean(dim=1, keepdim=True).clamp_min(self.eps).sqrt()
            gate = torch.sigmoid(norm)
            output.append(feature + self.step * gate * update)
        return RealFeature(output)


def _validate_matching_blocks(x: RealFeature, u: RealUpdate) -> None:
    if len(x.blocks) != len(u.blocks):
        raise ValueError("Real update strategies require matching body-order blocks")
    for order, (feature, update) in enumerate(zip(x.blocks, u.blocks)):
        if feature.shape != update.shape:
            raise ValueError(
                f"Order-{order} feature shape {tuple(feature.shape)} does not match "
                f"update shape {tuple(update.shape)}"
            )


__all__ = ["NormGatedUpdate", "ReplaceUpdate", "ResidualUpdate"]
