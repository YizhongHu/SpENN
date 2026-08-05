"""Wavefunction readout namespace.

Readouts consume :class:`tpen.data.real.Feature`.
"""

from tpen.nn.readout.pfaffian import PfaffianReadout

__all__ = ["PfaffianReadout"]
