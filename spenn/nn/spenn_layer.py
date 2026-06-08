"""One SpENN layer scaffold."""

from __future__ import annotations

from torch import nn

from spenn.data.real import RealFeature
from spenn.equivariance import EquivariantMap


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

    def forward_impl(self, x: RealFeature) -> RealFeature:
        """Apply one SpENN layer to a real feature state."""

        interaction = self.mixing(x, x) if self.bilinear_mixing else self.mixing(x)
        irrep_interaction = self.fourier(interaction)
        activated = self.activation(irrep_interaction)
        irrep_update = self.path_aggregation(activated)
        real_update = self.inverse_fourier(irrep_update)
        return self.update(x, real_update)


__all__ = ["SpENNLayer"]
