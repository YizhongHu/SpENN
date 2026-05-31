"""Unit tests for neural-network shape assertions."""

from __future__ import annotations

import pytest
import torch

from spenn.data import FeatureDict, Par
from spenn.data.batch import ElectronBatch
from spenn.nn.cusp import ElectronElectronCusp
from spenn.nn.encoding import ElectronPairEncoder
from spenn.nn.readout.node import TwoElectronSingletSymmetricReadout, TwoElectronTripletNodeReadout
from spenn.nn.spechtmp.message_head import _project_product
from spenn.nn.spechtmp.update_head import _project_branch


def test_encoder_shape_asserts_catch_bad_tuple_mlp_output() -> None:
    class BadTupleMLP(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.new_zeros(*inputs.shape[:-1], 2, 1)

    encoder = ElectronPairEncoder(channels=[0, 2, 0])
    positions = torch.zeros(1, 3, 2)

    with pytest.raises(AssertionError):
        encoder.apply_tuple_mlp(BadTupleMLP(), positions)


def test_spechtmp_projection_asserts_catch_bad_linear_output_shape() -> None:
    class BadLinear(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.new_zeros(*inputs.shape[:-1], 2, 1)

    product = torch.zeros(1, 1, 1, 2, 2, 2, 2, 1, 1)
    branch = torch.zeros(1, 1, 1, 2, 2, 2, 2, 1, 1)

    with pytest.raises(AssertionError):
        _project_product(BadLinear(), Par("S"), Par("H"), Par("H"), product)
    with pytest.raises(AssertionError):
        _project_branch(BadLinear(), Par("S"), Par("S"), branch)


def test_cusp_shape_asserts_preserve_batch_shape() -> None:
    positions = torch.zeros(4, 3, 2, dtype=torch.float64)
    output = ElectronElectronCusp()(ElectronBatch(positions))

    assert output.shape == (4,)


def test_two_electron_triplet_node_readout_requires_pair_symmetric_features() -> None:
    readout = TwoElectronTripletNodeReadout(node_axis=2)
    batch = ElectronBatch(positions=torch.zeros(2, 2, 3, dtype=torch.float64))
    features = ElectronPairEncoder(channels=[0, 2, 3])(batch)

    output = readout(features, batch)

    assert output.logabs.shape == (2,)
    assert output.sign.shape == (2,)

    with pytest.raises(KeyError):
        readout(FeatureDict({Par("A"): features.get(Par("A"))}), batch)


def test_two_electron_singlet_readout_is_positive_and_exchange_symmetric() -> None:
    readout = TwoElectronSingletSymmetricReadout()
    spins = torch.tensor([[1.0, -1.0]], dtype=torch.float64).expand(3, -1)
    batch = ElectronBatch(positions=torch.randn(3, 2, 3, dtype=torch.float64), spins=spins)
    encoder = ElectronPairEncoder(channels=[0, 2, 3], include_spins=False)
    features = encoder(batch)
    swapped = ElectronBatch(positions=batch.positions[:, [1, 0]], spins=spins)

    output = readout(features, batch)
    swapped_output = readout(encoder(swapped), swapped)

    assert output.logabs.shape == (3,)
    assert output.sign.shape == (3,)
    assert torch.equal(output.sign, torch.ones_like(output.sign))
    assert torch.allclose(output.logabs, swapped_output.logabs)


def test_two_electron_singlet_readout_can_add_harmonic_envelope() -> None:
    coefficient = 0.25
    readout = TwoElectronSingletSymmetricReadout(envelope_coefficient=coefficient, zero_init_residual=True)
    positions = torch.randn(3, 2, 3, dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    features = ElectronPairEncoder(channels=[0, 2, 3], include_spins=False)(batch)

    output = readout(features, batch)

    expected = -coefficient * positions.square().sum(dim=(1, 2))
    assert torch.allclose(output.logabs, expected)
