"""Wavefunction readout namespace.

Readouts consume :class:`spenn.data.real.Feature`.
"""

from spenn.nn.readout.pfaffian import PfaffianReadout

__all__ = ["PfaffianReadout"]
