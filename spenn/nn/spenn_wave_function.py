"""Composed SpENN wavefunction scaffold."""

from __future__ import annotations

from collections.abc import Iterable

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.dependencies import require_torch, require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.context import SpENNForwardContext
from spenn.nn.tpen_stack import TPENStack

torch = require_torch(feature="SpENN wavefunction modules")
nn = require_torch_nn(feature="SpENN wavefunction modules")


class SpENNWaveFunction(EquivariantMap):
    """Compose basis, embedding, a TPEN layer stack, readout, and an envelope.

    The full pipeline is::

        ElectronBatch
          -> ElectronBasis (optional)
          -> ElectronBasisFeatures
          -> embedding
          -> TPENStack (TPEN layers)
          -> readout
          -> + additive log-amplitude envelope

    The raw :class:`ElectronBatch` is still passed to the readout and envelope so
    they see true coordinates; the basis only re-represents the per-particle
    input to the embedding.

    Parameters
    ----------
    embedding : torch.nn.Module
        Module mapping the basis output (or, when ``basis`` is ``None``, an
        :class:`ElectronBatch`) to :class:`spenn.data.real.RealFeature`.
    layers : iterable of torch.nn.Module or TPENStack
        TPEN layers, or an already-constructed :class:`TPENStack`. Iterables
        are wrapped into a stack; the layers always live in ``self.stack``.
    readout : torch.nn.Module
        Module mapping final real features to :class:`WavefunctionOutput`.
    envelope : torch.nn.Module
        Required additive log-amplitude envelope. Envelopes accept ``batch``
        and return an additive tensor matching ``output.logabs``.
    basis : torch.nn.Module or None, optional
        Optional :class:`spenn.nn.ElectronBasis` applied before the embedding.
        When ``None``, the embedding consumes the raw :class:`ElectronBatch`.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        embedding: nn.Module,
        layers: Iterable[nn.Module] | TPENStack = (),
        readout: nn.Module,
        envelope: nn.Module | None,
        basis: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if envelope is None:
            raise ValueError("SpENNWaveFunction requires an envelope module")
        self.basis = basis
        self.embedding = embedding
        self.stack = layers if isinstance(layers, TPENStack) else TPENStack(layers)
        self.readout = readout
        self.envelope = envelope

    def forward_impl(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction for an electron batch."""

        basis_features = self.basis(batch) if self.basis is not None else None
        context = SpENNForwardContext(batch=batch, basis_features=basis_features)
        embedded_input = basis_features if basis_features is not None else batch
        features = self.embedding(embedded_input, context=context)
        features = self.stack(features, context)
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
