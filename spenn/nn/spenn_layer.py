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
    feature_activation, update_activation : torch.nn.Module or None, optional
        Optional real-state gates applied before mixing and before the residual
        update, respectively.
    feature_envelope, update_envelope : torch.nn.Module or None, optional
        Optional context-dependent real-state envelopes applied before mixing
        and before the residual update, respectively.
    activation : torch.nn.Module or None, optional
        Backward-compatible alias for ``irrep_activation``.
    bilinear_mixing : bool, optional
        If ``True``, call ``mixing(x, x)``. Otherwise call ``mixing(x)``.
    update_norm : torch.nn.Module or None, optional
        Backward-compatible alias for ``update_activation``.
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
        feature_activation: nn.Module | None = None,
        feature_envelope: nn.Module | None = None,
        update_activation: nn.Module | None = None,
        update_envelope: nn.Module | None = None,
        activation: nn.Module | None = None,
        bilinear_mixing: bool = False,
        update_norm: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if activation is not None and irrep_activation is not None:
            raise ValueError("Specify only one of activation or irrep_activation")
        if update_norm is not None and update_activation is not None:
            raise ValueError("Specify only one of update_norm or update_activation")
        self.mixing = mixing
        self.fourier = fourier
        self.irrep_activation = irrep_activation or activation or GatedNormActivation(gate=nn.SiLU())
        # Compatibility aliases for older configs/tests and legacy feature-normalization wiring.
        self.activation = self.irrep_activation
        self.path_aggregation = path_aggregation
        self.inverse_fourier = inverse_fourier
        self.update = update
        self.feature_activation = feature_activation
        self.feature_envelope = feature_envelope
        self.update_activation = update_activation or update_norm
        self.update_norm = self.update_activation
        self.update_envelope = update_envelope
        self.bilinear_mixing = bool(bilinear_mixing)

    def forward_impl(
        self,
        x: RealFeature,
        context: SpENNForwardContext | None = None,
    ) -> RealFeature:
        """Apply one SpENN layer to a real feature state."""

        if self.feature_activation is not None:
            x = self.feature_activation(x)
        if self.feature_envelope is not None:
            if context is None:
                raise ValueError("feature_envelope requires a SpENNForwardContext")
            x = self.feature_envelope(x, context)
        interaction = self.mixing(x, x) if self.bilinear_mixing else self.mixing(x)
        irrep_interaction = self.fourier(interaction)
        activated = self.irrep_activation(irrep_interaction)
        irrep_update = self.path_aggregation(activated)
        real_update = self.inverse_fourier(irrep_update)
        update_activation = self.update_activation if self.update_activation is not None else self.update_norm
        if update_activation is not None:
            real_update = update_activation(real_update)
        if self.update_envelope is not None:
            if context is None:
                raise ValueError("update_envelope requires a SpENNForwardContext")
            real_update = self.update_envelope(real_update, context)
        return self.update(x, real_update)


__all__ = ["SpENNLayer"]
