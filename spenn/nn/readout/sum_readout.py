"""Stable signed-log readout summation."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.feature_dict import FeatureDict
from spenn.nn.readout.pfaffian import signed_logsumexp_outputs


class SumReadout(nn.Module):
    """Combine several readouts via a stable signed-log sum."""

    def __init__(self, readouts: list[nn.Module], learn_weights: bool = True, **_: object) -> None:
        super().__init__()
        self.readouts = nn.ModuleList(readouts)
        self.learn_weights = learn_weights
        if learn_weights and len(readouts) > 0:
            self.weight_logits = nn.Parameter(torch.zeros(len(readouts)))
        else:
            self.register_buffer("weight_logits", torch.zeros(len(readouts)), persistent=False)

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        outputs = [readout(features, batch) for readout in self.readouts]
        weights = torch.softmax(self.weight_logits, dim=0) if self.learn_weights and len(outputs) > 0 else None
        return signed_logsumexp_outputs(outputs, weights=weights)
