"""Wavefunction readout namespace.

Readouts consume :class:`spenn.data.real.RealFeature`.
"""

from spenn.nn.readout.pfaffian import PfaffianReadout

__all__ = ["PfaffianReadout"]
