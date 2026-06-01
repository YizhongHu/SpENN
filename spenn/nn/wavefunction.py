"""Composed SpENN wavefunction module."""

from __future__ import annotations

from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.nn.cusp import Cusp


class SpENNWavefunction(nn.Module):
    """Compose encoder, message passing, readout, and optional cusp.

    Parameters
    ----------
    encoder : torch.nn.Module
        Module that maps an :class:`~spenn.data.batch.ElectronBatch` to a
        feature dictionary.
    spechtmp : torch.nn.Module
        Equivariant message-passing module applied to encoded features.
    readout : torch.nn.Module
        Module that maps final features and the input batch to a
        :class:`~spenn.data.batch.WavefunctionOutput`.
    cusp : spenn.nn.cusp.Cusp or None, optional
        Additive log-amplitude cusp module. When present, it is evaluated after
        the readout and must return a tensor with the same shape as
        ``readout(batch).logabs``.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    The cusp is intentionally the last model step. It changes only the
    log-amplitude and preserves the readout sign and phase, so antisymmetry
    must come from the readout/model architecture rather than the cusp.
    """

    def __init__(self, encoder, spechtmp, readout, cusp: Cusp | None = None, **_: object) -> None:
        super().__init__()
        self.encoder = encoder
        self.spechtmp = spechtmp
        self.readout = readout
        self.cusp = cusp

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction.

        Parameters
        ----------
        batch : ElectronBatch
            Electron positions and optional metadata.

        Returns
        -------
        WavefunctionOutput
            Signed-log wavefunction output with cusp applied after readout
            when configured.
        """

        features = self.encoder(batch)
        features = self.spechtmp(features)
        out = self.readout(features, batch)
        if self.cusp is not None:
            cusp = self.cusp(batch)
            if cusp.shape != out.logabs.shape:
                raise ValueError(f"Cusp output must have shape {tuple(out.logabs.shape)}, got {tuple(cusp.shape)}")
            out = WavefunctionOutput(
                logabs=out.logabs + cusp,
                sign=out.sign,
                phase=out.phase,
                aux=dict(out.aux),
            )
        return out
