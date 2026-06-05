"""Equivariant real-space mixing scaffold."""

from __future__ import annotations

from spenn.data import RealFeature, RealInteraction
from spenn.nn.equivariant_map import EquivariantMap
from spenn.reps.paths import VirtualPath, enumerate_virtual_paths


class EquivariantMixing(EquivariantMap):
    """Bilinear virtual-support real-space mixing module.

    Parameters
    ----------
    max_order : int
        Maximum virtual support order.
    paths : tuple of VirtualPath or None, optional
        Precomputed paths. If ``None``, valid paths are enumerated.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        max_order: int,
        *,
        paths: tuple[VirtualPath, ...] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_order = int(max_order)
        self.paths = enumerate_virtual_paths(self.max_order) if paths is None else tuple(paths)

    def forward_impl(self, x1: RealFeature, x2: RealFeature | None = None) -> RealInteraction:
        """Mix one or two real feature states into path-resolved interactions."""

        raise NotImplementedError("EquivariantMixing tensor contraction is not implemented yet")


__all__ = ["EquivariantMixing"]
