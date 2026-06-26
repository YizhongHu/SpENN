"""One SpENN layer scaffold."""

from __future__ import annotations

from spenn.data.real import RealFeature
from spenn.dependencies import require_torch_nn
from spenn.equivariance import EquivariantMap

nn = require_torch_nn(feature="SpENN layer modules")


class SpENNLayer(EquivariantMap):
    """Compose mixing, Fourier, activation, path aggregation, and update maps.

    Parameters
    ----------
    mixing, fourier, activation, path_aggregation, inverse_fourier, update : torch.nn.Module
        Layer components implementing the new SpENN pipeline. The activation
        keeps the path axis visible, while `path_aggregation` converts the
        activated irrep interaction to an irrep feature update.
    bilinear_mixing : bool, optional
        If ``True``, call ``mixing(x, x)``. Otherwise call ``mixing(x)``.
    update_norm : torch.nn.Module or None, optional
        Optional equivariant normalization applied to the real update proposal
        before the residual update (the ``update`` feature-normalization site,
        N3). When ``None``, the update increment is used unchanged.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        mixing: nn.Module,
        fourier: nn.Module,
        activation: nn.Module,
        path_aggregation: nn.Module,
        inverse_fourier: nn.Module,
        update: nn.Module,
        bilinear_mixing: bool = False,
        update_norm: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mixing = mixing
        self.fourier = fourier
        self.activation = activation
        self.path_aggregation = path_aggregation
        self.inverse_fourier = inverse_fourier
        self.update = update
        self.bilinear_mixing = bool(bilinear_mixing)
        self.update_norm = update_norm

    def forward_impl(self, x: RealFeature) -> RealFeature:
        """Apply one SpENN layer to a real feature state."""

        interaction = self.mixing(x, x) if self.bilinear_mixing else self.mixing(x)
        irrep_interaction = self.fourier(interaction)
        activated = self.activation(irrep_interaction)
        irrep_update = self.path_aggregation(activated)
        real_update = self.inverse_fourier(irrep_update)
        if self.update_norm is not None:
            real_update = self.update_norm(real_update)
        return self.update(x, real_update)


__all__ = ["SpENNLayer"]
