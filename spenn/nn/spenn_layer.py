"""One SpENN layer scaffold."""

from __future__ import annotations

from spenn.data.real import RealFeature
from spenn.dependencies import require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.activation import GatedNormActivation
from spenn.nn.context import SpENNForwardContext

nn = require_torch_nn(feature="SpENN layer modules")


class SpENNLayer(EquivariantMap):
    """Compose mixing, Fourier, activation, path aggregation, and update maps.

    Parameters
    ----------
    mixing, fourier, irrep_activation, path_aggregation, inverse_fourier, update : torch.nn.Module
        Layer components implementing the SpENN pipeline. The irrep activation
        keeps the path axis visible, while `path_aggregation` converts the
        activated irrep interaction to an irrep feature update.
    update_normalization, feature_normalization : torch.nn.Module or None, optional
        Optional real-state normalization modules applied to the update
        increment and end-of-layer feature state, respectively.
    feature_envelope, update_envelope : torch.nn.Module or None, optional
        Optional context-dependent real-state envelopes applied after the
        matching normalization.
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
        path_aggregation: nn.Module,
        inverse_fourier: nn.Module,
        update: nn.Module,
        irrep_activation: nn.Module | None = None,
        update_envelope: nn.Module | None = None,
        update_normalization: nn.Module | None = None,
        feature_envelope: nn.Module | None = None,
        feature_normalization: nn.Module | None = None,
        bilinear_mixing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mixing = mixing
        self.fourier = fourier
        self.irrep_activation = irrep_activation or GatedNormActivation(gate=nn.SiLU())
        self.path_aggregation = path_aggregation
        self.inverse_fourier = inverse_fourier
        self.update = update
        self.update_envelope = update_envelope
        self.update_normalization = update_normalization
        self.feature_envelope = feature_envelope
        self.feature_normalization = feature_normalization
        self.bilinear_mixing = bool(bilinear_mixing)

    def forward_impl(
        self,
        x: RealFeature,
        context: SpENNForwardContext | None = None,
    ) -> RealFeature:
        """Apply one SpENN layer to a real feature state."""

        interaction = self.mixing(x, x) if self.bilinear_mixing else self.mixing(x)
        irrep_interaction = self.fourier(interaction)
        activated = self.irrep_activation(irrep_interaction)
        irrep_update = self.path_aggregation(activated)
        real_update = self.inverse_fourier(irrep_update)
        if self.update_normalization is not None:
            real_update = self.update_normalization(real_update)
        if self.update_envelope is not None:
            if context is None:
                raise ValueError("update_envelope requires a SpENNForwardContext")
            real_update = self.update_envelope(real_update, context)
        features = self.update(x, real_update)
        if self.feature_normalization is not None:
            features = self.feature_normalization(features)
        if self.feature_envelope is not None:
            if context is None:
                raise ValueError("feature_envelope requires a SpENNForwardContext")
            features = self.feature_envelope(features, context)
        return features


__all__ = ["SpENNLayer"]
