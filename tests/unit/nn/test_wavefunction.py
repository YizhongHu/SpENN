"""Composed wavefunction module tests."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data import FeatureDict, Par
from spenn.data.batch import ElectronBatch
from spenn.nn.readout.pfaffian import PfaffianReadout
from spenn.nn.wavefunction import SpENNWavefunction


class PairDifferenceEncoder(nn.Module):
    """Build simple one- and two-body features from 1D positions."""

    def forward(self, batch: ElectronBatch) -> FeatureDict:
        x = batch.positions[..., 0]
        carrier = (x.unsqueeze(2) - x.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        gate = torch.ones_like(carrier)
        one_body = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return FeatureDict({Par("H"): one_body, Par("S"): gate, Par("A"): carrier})


def _set_unit_readout_weights(readout: PfaffianReadout) -> None:
    with torch.no_grad():
        for carrier, gate, border in zip(
            readout.carrier_projections,
            readout.gate_projections,
            readout.border_projections,
            strict=True,
        ):
            carrier.weight.fill_(1.0)
            gate.weight.zero_()
            gate.bias.fill_(1.0)
            border.weight.fill_(1.0)


def test_spenn_wavefunction_returns_signed_log_output_and_kernel_aux() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    encoder = PairDifferenceEncoder()
    readout = PfaffianReadout()
    readout.build_skew_kernel(encoder(batch), batch)
    _set_unit_readout_weights(readout)
    model = SpENNWavefunction(encoder=encoder, spechtmp=nn.Identity(), readout=readout)

    output = model(batch)

    assert output.logabs.shape == (2,)
    assert output.sign.shape == (2,)
    assert torch.all(torch.isfinite(output.logabs))
    assert torch.all(torch.isin(output.sign, torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)))
    assert "K" in output.aux
    assert torch.allclose(output.aux["K"] + output.aux["K"].transpose(-1, -2), torch.zeros_like(output.aux["K"]))
