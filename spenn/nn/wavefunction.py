"""Composed SpENN wavefunction module."""

from __future__ import annotations

from torch import nn

from spenn.data_structures.batch import ElectronBatch, WavefunctionOutput
from spenn.nn.encoding.cusp import ElectronElectronCusp


class SpENNWavefunction(nn.Module):
    """Compose encoder, SpechtMP, readout, and optional cusp."""

    def __init__(self, encoder, spechtmp, readout, cusp: ElectronElectronCusp | None = None, **_: object) -> None:
        super().__init__()
        self.encoder = encoder
        self.spechtmp = spechtmp
        self.readout = readout
        self.cusp = cusp

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        features = self.encoder(batch)
        features = self.spechtmp(features)
        out = self.readout(features, batch)
        if self.cusp is not None:
            out = WavefunctionOutput(
                logabs=out.logabs + self.cusp(batch),
                sign=out.sign,
                phase=out.phase,
                aux=dict(out.aux),
            )
        return out
