"""Equivariant RMS normalization for real tuple features."""

from __future__ import annotations

from tpen.data.real import Feature
from tpen.dependencies import require_torch
from tpen.equivariance import EquivariantMap

torch = require_torch(feature="SpENN normalization modules")


class RMSNorm(EquivariantMap):
    """Root-mean-square normalize each block over its channel axis.

    For every positive-order block ``x`` with shape ``[batch, channels,
    indices...]`` the output is ``x * rsqrt(mean_c x^2 + eps)`` where the mean is
    taken over the channel axis. The zero-order block (zero channels) and any
    empty block are passed through unchanged. The norm carries no learnable
    parameters, so the same module can be reused at every normalization site.

    Parameters
    ----------
    eps : float, optional
        Positive constant added to the mean square for numerical stability.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(self, *, eps: float = 1.0e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)

    def forward_impl(self, features: Feature) -> Feature:
        """Return a per-position channel RMS normalization of every block.

        Parameters
        ----------
        features : Feature
            Real tuple features (or a :class:`tpen.data.real.Update`).

        Returns
        -------
        Feature
            A new state of the same concrete type with normalized blocks.
        """

        blocks = []
        for _order, block in features.items():
            if block.shape[1] == 0:
                # Zero-channel blocks (the order-0 block) carry no scale.
                blocks.append(block.clone())
                continue
            mean_square = block.square().mean(dim=1, keepdim=True)
            blocks.append(block * torch.rsqrt(mean_square + self.eps))
        # Preserve the concrete type (Feature vs Update).
        return type(features)(blocks)


__all__ = ["RMSNorm"]
