"""Wavefunction readout namespace.

Readouts consume :class:`spenn.data.real.RealFeature`. If a readout needs irrep
coordinates, it should perform the required Fourier transform internally before
evaluating the readout itself.
"""

from spenn.nn.readout.determinant import DeterminantReadout
from spenn.nn.readout.pfaffian import PfaffianReadout
from spenn.nn.readout.sum_readout import SumReadout

__all__ = ["DeterminantReadout", "PfaffianReadout", "SumReadout"]
