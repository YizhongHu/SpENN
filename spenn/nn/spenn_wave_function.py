"""Composed SpENN wavefunction scaffold."""

from __future__ import annotations

from collections.abc import Iterable

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.dependencies import require_torch, require_torch_nn
from spenn.equivariance import EquivariantMap

torch = require_torch(feature="SpENN wavefunction modules")
nn = require_torch_nn(feature="SpENN wavefunction modules")


class SpENNWaveFunction(EquivariantMap):
    """Compose embedding, SpENN layers, readout, and an envelope factor.

    Parameters
    ----------
    embedding : torch.nn.Module
        Module mapping :class:`ElectronBatch` to
        :class:`spenn.data.real.RealFeature`.
    layers : iterable of torch.nn.Module
        Sequence of SpENN layers.
    readout : torch.nn.Module
        Module mapping final real features to :class:`WavefunctionOutput`.
    envelope : torch.nn.Module
        Required additive log-amplitude envelope. Envelopes accept ``batch``
        and return an additive tensor matching ``output.logabs``.
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if envelope is None:
            raise ValueError("SpENNWaveFunction requires an envelope module")
        self.embedding = embedding
        self.layers = nn.ModuleList(tuple(layers))
        self.readout = readout
        self.envelope = envelope

    def forward_impl(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction for an electron batch."""

        features = self.embedding(batch)
        for layer in self.layers:
            features = layer(features)
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
