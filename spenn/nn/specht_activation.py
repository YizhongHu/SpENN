"""Irrep activation and path aggregation scaffold."""

from __future__ import annotations

from torch import nn

from spenn.data import IrrepFeature, IrrepInteraction
from spenn.nn.equivariant_map import EquivariantMap


class SpechtActivation(EquivariantMap):
    """Aggregate path-resolved irrep interactions into irrep features.

    The scaffold only aggregates over paths. It deliberately avoids arbitrary
    nonlinearities over the transforming irrep coordinate; scalar-specific and
    norm/gated activations can be added behind this API later.

    Parameters
    ----------
    scalar_activation : torch.nn.Module or None, optional
        Optional activation applied only when the transforming coordinate has
        dimension one.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(self, scalar_activation: nn.Module | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scalar_activation = scalar_activation

    def forward_impl(self, x: IrrepInteraction) -> IrrepFeature:
        """Return path-aggregated irrep features."""

        blocks = {}
        for partition, tensor in x.items():
            reduced = tensor.sum(dim=2)
            if self.scalar_activation is not None and reduced.shape[-2] == 1:
                reduced = self.scalar_activation(reduced)
            blocks[partition] = reduced
        return IrrepFeature(blocks)


__all__ = ["SpechtActivation"]
