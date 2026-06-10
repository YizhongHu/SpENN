"""Composite readout scaffold.

All readouts in the new SpENN core consume :class:`spenn.data.real.RealFeature`.
Readout-specific Fourier transforms should happen inside the component readout
that needs them, before it contributes to the final signed-log sum.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.real import RealFeature


class SumReadout(nn.Module):
    """Combine multiple real-feature readouts in signed-log form.

    Parameters
    ----------
    readouts : iterable of torch.nn.Module
        Component readouts accepting ``(RealFeature, ElectronBatch)``.
    trainable : bool, optional
        Whether to learn one linear weight per component readout.
    """

    def __init__(self, readouts: Iterable[nn.Module], *, trainable: bool = False) -> None:
        super().__init__()
        readout_tuple = tuple(readouts)
        self.readouts = nn.ModuleList(readout_tuple)
        self.trainable = bool(trainable)
        if self.trainable:
            self.readout_weights = nn.Parameter(torch.ones(len(readout_tuple), dtype=torch.float64))
        else:
            self.register_parameter("readout_weights", None)

    def forward(self, features: RealFeature, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate and sum component readouts."""

        if not self.readouts:
            raise ValueError("SumReadout requires at least one component readout")
        outputs = [readout(features, batch) for readout in self.readouts]
        values = torch.stack([output.sign * torch.exp(output.logabs) for output in outputs], dim=0)
        if self.readout_weights is not None:
            values = values * self.readout_weights.view(-1, *([1] * (values.ndim - 1)))
        values = values.sum(dim=0)
        sign = torch.sign(values)
        logabs = torch.where(sign == 0, torch.full_like(values, -torch.inf), torch.log(values.abs()))
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"components": outputs})


__all__ = ["SumReadout"]
