"""End-to-end antisymmetry tests for Pfaffian wavefunctions."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data import FeatureDict, Par
from spenn.data.batch import ElectronBatch
from spenn.nn.cusp import ElectronElectronCusp
from spenn.nn.readout.pfaffian import PfaffianReadout
from spenn.nn.wavefunction import SpENNWavefunction


class PairDifferenceEncoder(nn.Module):
    def forward(self, batch: ElectronBatch) -> FeatureDict:
        x = batch.positions[..., 0]
        carrier = (x.unsqueeze(2) - x.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        gate = torch.ones_like(carrier)
        one_body = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return FeatureDict({Par("H"): one_body, Par("S"): gate, Par("A"): carrier})


def _set_unit_readout_weights(readout: PfaffianReadout) -> None:
    with torch.no_grad():
        carrier = readout.carrier_projections[0]
        gate = readout.gate_projections[0]
        carrier.weight.fill_(1.0)
        gate.weight.zero_()
        gate.bias.fill_(1.0)


def test_pfaffian_wavefunction_sign_flips_under_transposition_with_symmetric_cusp() -> None:
    positions = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    encoder = PairDifferenceEncoder()
    readout = PfaffianReadout()
    readout.build_skew_kernel(encoder(batch), batch)
    _set_unit_readout_weights(readout)
    model = SpENNWavefunction(
        encoder=encoder,
        spechtmp=nn.Identity(),
        readout=readout,
        cusp=ElectronElectronCusp(coefficient=0.25, range_parameter=0.5, eps=0.0),
    )

    original = model(batch)
    swapped_positions = positions[:, [1, 0]]
    swapped = model(ElectronBatch(positions=swapped_positions))

    assert torch.allclose(original.logabs, swapped.logabs)
    assert torch.equal(original.sign, -swapped.sign)
