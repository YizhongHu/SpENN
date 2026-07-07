"""Composed SpENN wavefunction scaffold."""

from __future__ import annotations

from collections.abc import Iterable

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.dependencies import require_torch, require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.context import SpENNForwardContext
from spenn.nn.normalization import FeatureNormalization
from spenn.nn.spenn_layer import SpENNLayer

torch = require_torch(feature="SpENN wavefunction modules")
nn = require_torch_nn(feature="SpENN wavefunction modules")


class SpENNWaveFunction(EquivariantMap):
    """Compose basis, embedding, SpENN layers, readout, and an envelope factor.

    The full pipeline is::

        ElectronBatch
          -> ElectronBasis (optional)
          -> ElectronBasisFeatures
          -> embedding
          -> SpENN feature layers
          -> readout
          -> Gaussian envelope

    The raw :class:`ElectronBatch` is still passed to the readout and envelope so
    they see true coordinates; the basis only re-represents the per-particle
    input to the embedding.

    Parameters
    ----------
    embedding : torch.nn.Module
        Module mapping the basis output (or, when ``basis`` is ``None``, an
        :class:`ElectronBatch`) to :class:`spenn.data.real.RealFeature`.
    embedding_activation : torch.nn.Module or None, optional
        Optional real-feature activation applied after embedding and before
        SpENN layers.
    layers : iterable of torch.nn.Module
        Sequence of SpENN layers.
    readout : torch.nn.Module
        Module mapping final real features to :class:`WavefunctionOutput`.
    envelope : torch.nn.Module
        Required additive log-amplitude envelope. Envelopes accept ``batch``
        and return an additive tensor matching ``output.logabs``.
    basis : torch.nn.Module or None, optional
        Optional :class:`spenn.nn.ElectronBasis` applied before the embedding.
        When ``None``, the embedding consumes the raw :class:`ElectronBatch`.
    feature_normalization : spenn.nn.FeatureNormalization or None, optional
        Optional feature-scale normalization choice. Its ``mode`` selects the
        insertion site; ``update`` mode is wired into each layer's
        ``update_norm`` while the other sites are applied here in ``forward``.
    seed : int or None, optional
        Legacy config-resolution shim. This value may be used by OmegaConf
        interpolations such as ``${model.seed}``, but the wavefunction does not
        use it to seed or initialize anything. New configs should wire explicit
        initializer objects into randomized components instead.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        embedding: nn.Module,
        layers: Iterable[nn.Module] = (),
        readout: nn.Module,
        envelope: nn.Module | None,
        basis: nn.Module | None = None,
        embedding_activation: nn.Module | None = None,
        feature_normalization: FeatureNormalization | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if envelope is None:
            raise ValueError("SpENNWaveFunction requires an envelope module")
        self.basis = basis
        self.embedding = embedding
        self.embedding_activation = embedding_activation
        self.layers = nn.ModuleList(tuple(layers))
        self.readout = readout
        self.envelope = envelope
        self.feature_normalization = feature_normalization
        self.legacy_config_seed = None if seed is None else int(seed)
        # The ``update`` site lives inside each layer, so inject the shared norm
        # into every layer's update_norm slot. Other sites are applied in forward.
        if feature_normalization is not None and feature_normalization.applies_at("update"):
            for index, layer in enumerate(self.layers):
                if not hasattr(layer, "update_norm"):
                    raise TypeError(
                        "feature normalization mode 'update' requires layers to accept "
                        f"update_norm; layers[{index}]={type(layer).__name__} does not"
                    )
                layer.update_activation = feature_normalization.norm
                layer.update_norm = feature_normalization.norm

    def forward_impl(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction for an electron batch."""

        basis_features = self.basis(batch) if self.basis is not None else None
        context = SpENNForwardContext(batch=batch, basis_features=basis_features)
        embedded_input = basis_features if basis_features is not None else batch
        features = self.embedding(embedded_input)
        if self.embedding_activation is not None:
            features = self.embedding_activation(features)
        normalization = self.feature_normalization
        if normalization is not None and normalization.applies_at("post_embedding"):
            features = normalization.apply_norm(features)
        for layer in self.layers:
            features = layer(features, context) if isinstance(layer, SpENNLayer) else layer(features)
            if normalization is not None and normalization.applies_at("post_feature_layer"):
                features = normalization.apply_norm(features)
        if normalization is not None and normalization.applies_at("pre_readout"):
            features = normalization.apply_norm(features)
        output = self.readout(features, batch)
        logabs = output.logabs
        logabs = logabs + _log_factor(self.envelope, batch, output.logabs.shape, name="Envelope")
        return WavefunctionOutput(
            logabs=logabs,
            sign=output.sign,
            phase=output.phase,
            aux=dict(output.aux),
        )


def _log_factor(module: nn.Module, batch: ElectronBatch, shape: torch.Size, *, name: str) -> torch.Tensor:
    value = module(batch)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} output must be a torch.Tensor, got {type(value)!r}")
    if value.shape != shape:
        raise ValueError(f"{name} output must have shape {tuple(shape)}, got {tuple(value.shape)}")
    return value


__all__ = ["SpENNWaveFunction"]
