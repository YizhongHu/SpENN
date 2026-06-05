"""Real-feature update module."""

from __future__ import annotations

from spenn.data import RealFeature, RealUpdate
from spenn.nn.equivariant_map import EquivariantMap


class Update(EquivariantMap):
    """Apply a real-space update proposal to persistent features.

    Parameters
    ----------
    step : float, optional
        Scalar multiplier for the update proposal. The first scaffold uses an
        identity channel map, so feature and update channels must match.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(self, step: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = float(step)

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return ``x + step * u`` blockwise."""

        output = [tensor.clone() for tensor in x.blocks]
        for order, update in u.items():
            if order >= len(output):
                raise ValueError(f"Update contains order-{order} block absent from RealFeature")
            if output[order].shape != update.shape:
                raise ValueError(
                    f"Update order-{order} shape {tuple(update.shape)} does not match "
                    f"feature shape {tuple(output[order].shape)}"
                )
            output[order] = output[order] + self.step * update
        return RealFeature(output)


__all__ = ["Update"]
