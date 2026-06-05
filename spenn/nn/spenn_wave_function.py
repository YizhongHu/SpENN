"""Composed SpENN wavefunction scaffold."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from spenn.data import ElectronBatch, WavefunctionOutput


class SpENNWaveFunction(nn.Module):
    """Compose embedding, SpENN layers, readout, and optional cusp.

    Parameters
    ----------
    embedding : torch.nn.Module
        Module mapping :class:`ElectronBatch` to :class:`spenn.data.RealFeature`.
    layers : iterable of torch.nn.Module
        Sequence of SpENN layers.
    readout : torch.nn.Module
        Module mapping final real features to :class:`WavefunctionOutput`.
    cusp : torch.nn.Module or None, optional
        Optional additive log-amplitude cusp. A cusp may either accept
        ``(batch, output)`` and return a full output, or accept ``batch`` and
        return an additive tensor matching ``output.logabs``.
    """

    def __init__(
        self,
        *,
        embedding: nn.Module,
        layers: Iterable[nn.Module] = (),
        readout: nn.Module,
        cusp: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList(tuple(layers))
        self.readout = readout
        self.cusp = cusp

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction for an electron batch."""

        features = self.embedding(batch)
        for layer in self.layers:
            features = layer(features)
        output = self.readout(features, batch)
        if self.cusp is None:
            return output
        try:
            cusp_output = self.cusp(batch, output)
        except TypeError:
            cusp_output = self.cusp(batch)
        if isinstance(cusp_output, WavefunctionOutput):
            return cusp_output
        if cusp_output.shape != output.logabs.shape:
            raise ValueError(
                f"Cusp output must have shape {tuple(output.logabs.shape)}, got {tuple(cusp_output.shape)}"
            )
        return WavefunctionOutput(
            logabs=output.logabs + cusp_output,
            sign=output.sign,
            phase=output.phase,
            aux=dict(output.aux),
        )


__all__ = ["SpENNWaveFunction"]
