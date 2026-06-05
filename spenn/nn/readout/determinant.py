"""Determinant readout scaffold.

All readouts in the new SpENN core consume :class:`spenn.data.RealFeature`.
If a readout needs irrep-space carriers, it should run the required
Fourier-transform logic before evaluating the readout-specific determinant.
"""

from __future__ import annotations

from torch import nn

from spenn.data import ElectronBatch, RealFeature, WavefunctionOutput


class DeterminantReadout(nn.Module):
    """Placeholder determinant readout over real tuple features."""

    def forward(self, features: RealFeature, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate a determinant readout from real features."""

        raise NotImplementedError("DeterminantReadout.forward is not implemented yet")


__all__ = ["DeterminantReadout"]
