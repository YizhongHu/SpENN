"""One SpENN layer scaffold."""

from __future__ import annotations

from spenn.data.real import RealFeature
from spenn.dependencies import require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.context import SpENNForwardContext

nn = require_torch_nn(feature="SpENN layer modules")


class SpENNLayer(EquivariantMap):
    """Compose mixing, path aggregation, and update maps in real space.

    TPEN layer contract (MIG-TPEN-000 section 2.2):

    ``x -> mixing(W, Gamma) -> h -> aggregation(U, Gamma_c) -> u -> update``

    Both compute stages own their activations: :class:`EquivariantMixing`
    applies its pointwise ``Gamma`` to the path-resolved interaction, and
    :class:`PathAggregation` applies ``Gamma_c`` after contracting the path
    axis. There is no Fourier round-trip and no standalone activation stage.

    Parameters
    ----------
    mixing, path_aggregation, update : torch.nn.Module
        Layer components implementing the TPEN pipeline. `mixing` produces a
        path-resolved real interaction; `path_aggregation` contracts the path
        axis into a real feature update.
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
        path_aggregation: nn.Module,
        update: nn.Module,
        update_envelope: nn.Module | None = None,
        update_normalization: nn.Module | None = None,
        feature_envelope: nn.Module | None = None,
        feature_normalization: nn.Module | None = None,
        bilinear_mixing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mixing = mixing
        self.path_aggregation = path_aggregation
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
        """Apply one TPEN layer to a real feature state."""

        interaction = self.mixing(x, x) if self.bilinear_mixing else self.mixing(x)
        real_update = self.path_aggregation(interaction)
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
