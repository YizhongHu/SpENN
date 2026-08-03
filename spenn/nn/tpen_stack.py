"""TPEN layer stack container (MIG-TPEN-000 section 2.2).

The stack owns the ordered TPEN layers of a wavefunction:
``TPENStack = TPENLayer x T``. It is a pure feature-to-feature equivariant
map; embedding, readout, and the additive log-amplitude envelope stay owned
by the wavefunction.
"""

from __future__ import annotations

from collections.abc import Iterable

from spenn.data.real import RealFeature
from spenn.dependencies import require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.context import SpENNForwardContext
from spenn.nn.spenn_layer import SpENNLayer

nn = require_torch_nn(feature="TPEN stack modules")


class TPENStack(EquivariantMap):
    """Apply a sequence of TPEN layers to a real feature state.

    Each layer maps ``RealFeature -> RealFeature``. :class:`SpENNLayer`
    members receive the forward context so their optional per-layer feature
    and update envelopes can consume batch-derived coordinate scalars; plain
    feature-to-feature modules are called without it.

    Parameters
    ----------
    layers : iterable of torch.nn.Module, optional
        Ordered TPEN layers.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(self, layers: Iterable[nn.Module] = (), **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = nn.ModuleList(tuple(layers))

    def forward_impl(
        self,
        x: RealFeature,
        context: SpENNForwardContext | None = None,
    ) -> RealFeature:
        """Apply every layer in declaration order."""

        features = x
        for layer in self.layers:
            features = layer(features, context) if isinstance(layer, SpENNLayer) else layer(features)
        return features


__all__ = ["TPENStack"]
