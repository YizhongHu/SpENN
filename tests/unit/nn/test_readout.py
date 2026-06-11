"""Tests for readout scaffold trainability."""

from __future__ import annotations

import pytest
import torch
from typeguard import TypeCheckError

from spenn.data.batch import ElectronBatch
from spenn.data.permutation import all_permutations
from spenn.data.real import RealFeature, zero_block
from spenn.nn.readout import PfaffianReadout
from spenn.nn.readout.pfaffian import _ODD_PADDING_IRREP, pfaffian


class BlockContainer:
    def __init__(self, blocks: list[torch.Tensor]) -> None:
        self.blocks = blocks

    def __contains__(self, order: int) -> bool:
        return 0 <= order < len(self.blocks)


def _batch(n_electrons: int = 2) -> ElectronBatch:
    return ElectronBatch(positions=torch.zeros(1, n_electrons, 1, dtype=torch.float64))


def _pfaffian_features(n_electrons: int = 2) -> RealFeature:
    pair = torch.zeros(1, 2, n_electrons, n_electrons, dtype=torch.float64)
    pair[:, :, 0, 1] = torch.tensor([2.0, 4.0], dtype=torch.float64)
    pair[:, :, 1, 0] = -pair[:, :, 0, 1]
    one_body = torch.zeros(1, 1, n_electrons, dtype=torch.float64)
    return RealFeature([zero_block(dtype=torch.float64), one_body, pair])


def test_pfaffian_matches_known_four_by_four_formula() -> None:
    matrix = torch.tensor(
        [
            [0.0, 2.0, 3.0, 7.0],
            [-2.0, 0.0, 5.0, 11.0],
            [-3.0, -5.0, 0.0, 13.0],
            [-7.0, -11.0, -13.0, 0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        pfaffian(matrix),
        torch.tensor(2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0, dtype=torch.float64),
    )


def test_pfaffian_readout_weights_are_fixed_by_default() -> None:
    readout = PfaffianReadout(pair_channels=2, border_channels=1)

    output = readout(_pfaffian_features(), _batch())

    assert output.logabs.shape == (1,)
    assert "channel_weights" not in dict(readout.named_parameters())
    assert "channel_weight_buffer" in dict(readout.named_buffers())
    assert "border_weight_buffer" in dict(readout.named_buffers())


def test_pfaffian_readout_trainable_flag_registers_weights() -> None:
    readout = PfaffianReadout(pair_channels=2, border_channels=1, trainable=True)

    readout(_pfaffian_features(), _batch())

    parameters = dict(readout.named_parameters())
    assert "channel_weights" in parameters
    assert "border_weights" in parameters
    assert parameters["channel_weights"].requires_grad
    assert parameters["border_weights"].requires_grad


def test_pfaffian_readout_is_antisymmetric_under_even_particle_permutations() -> None:
    pair = torch.tensor(
        [
            [
                [
                    [0.0, 2.0, 3.0, 7.0],
                    [-2.0, 0.0, 5.0, 11.0],
                    [-3.0, -5.0, 0.0, 13.0],
                    [-7.0, -11.0, -13.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float64,
    )
    features = RealFeature([zero_block(dtype=torch.float64), torch.empty(1, 0, 4, dtype=torch.float64), pair])
    batch = _batch(n_electrons=4)
    readout = PfaffianReadout(channels=1)

    output = readout(features, batch)
    for permutation in all_permutations(4):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_uses_one_irrep_padding_block_for_odd_electrons() -> None:
    one_body = torch.tensor([[[7.0, 11.0, 13.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    pair[:, :, 0, 1] = 2.0
    pair[:, :, 0, 2] = 3.0
    pair[:, :, 1, 2] = 5.0
    pair = pair - pair.transpose(-1, -2)
    features = RealFeature([zero_block(dtype=torch.float64), one_body, pair])

    assert _ODD_PADDING_IRREP.parts == (1,)
    output = PfaffianReadout(channels=1)(features, _batch(n_electrons=3))

    expected = torch.tensor([2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0], dtype=torch.float64)
    torch.testing.assert_close(output.aux["pfaffian"], expected)
    torch.testing.assert_close(output.logabs, expected.log())
    torch.testing.assert_close(output.sign, torch.ones_like(expected))
    torch.testing.assert_close(output.aux["K"] + output.aux["K"].transpose(-1, -2), torch.zeros_like(output.aux["K"]))


def test_pfaffian_readout_is_antisymmetric_under_odd_particle_permutations() -> None:
    one_body = torch.tensor([[[7.0, 11.0, 13.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    pair[:, :, 0, 1] = 2.0
    pair[:, :, 0, 2] = 3.0
    pair[:, :, 1, 2] = 5.0
    pair = pair - pair.transpose(-1, -2)
    features = RealFeature([zero_block(dtype=torch.float64), one_body, pair])
    batch = _batch(n_electrons=3)
    readout = PfaffianReadout(channels=1)

    output = readout(features, batch)
    for permutation in all_permutations(3):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_builds_weighted_bordered_kernel() -> None:
    one_body = torch.tensor([[[2.0, 4.0, 6.0], [10.0, 14.0, 18.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    pair[:, 0, 0, 1] = 2.0
    pair[:, 0, 0, 2] = 3.0
    pair[:, 0, 1, 2] = 5.0
    pair[:, 1, 0, 1] = 4.0
    pair[:, 1, 0, 2] = 9.0
    pair[:, 1, 1, 2] = 11.0
    pair = pair - pair.transpose(-1, -2)
    features = RealFeature([zero_block(dtype=torch.float64), one_body, pair])

    kernel = PfaffianReadout(channels=2).build_skew_kernel(features, _batch(n_electrons=3))

    expected_pair = 0.5 * (pair[:, 0] + pair[:, 1])
    expected_border = torch.tensor([[6.0, 9.0, 12.0]], dtype=torch.float64)
    expected = torch.zeros(1, 4, 4, dtype=torch.float64)
    expected[:, :-1, :-1] = expected_pair
    expected[:, :-1, -1] = expected_border
    expected[:, -1, :-1] = -expected_border
    torch.testing.assert_close(kernel, expected)


def test_pfaffian_readout_requires_one_irrep_padding_for_odd_electron_systems() -> None:
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    features_without_border = RealFeature([zero_block(dtype=torch.float64), torch.empty(1, 0, 3, dtype=torch.float64), pair])

    with pytest.raises(KeyError, match=r"irrep \(1\)"):
        PfaffianReadout(channels=1)(features_without_border, _batch(n_electrons=3))


def test_pfaffian_readout_does_not_expose_odd_padding_toggle() -> None:
    with pytest.raises(TypeError, match="allow_odd_electron_bordered"):
        PfaffianReadout(allow_odd_electron_bordered=False, channels=1)  # type: ignore[call-arg]


def test_pfaffian_readout_rejects_malformed_kernel_inputs() -> None:
    readout = PfaffianReadout(pair_channels=2, border_channels=1)
    malformed_pair = BlockContainer([zero_block(dtype=torch.float64), torch.empty(1, 0, 3), torch.zeros(1, 1, 3)])
    feature = _pfaffian_features()
    border_mismatch = BlockContainer(
        [
            zero_block(dtype=torch.float64),
            torch.ones(1, 1, 4, dtype=torch.float64),
            torch.zeros(1, 1, 3, 3, dtype=torch.float64),
        ]
    )

    with pytest.raises(TypeCheckError, match="RealFeature"):
        readout.build_skew_kernel(malformed_pair)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Feature batch size"):
        readout.build_skew_kernel(feature, ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64)))
    with pytest.raises(TypeCheckError, match="RealFeature"):
        readout.build_skew_kernel(border_mismatch)  # type: ignore[arg-type]


def test_pfaffian_readout_returns_empty_pfaffian_for_zero_electrons() -> None:
    features = RealFeature(
        [
            zero_block(batch_size=1, dtype=torch.float64),
            torch.empty(1, 1, 0, dtype=torch.float64),
            torch.empty(1, 2, 0, 0, dtype=torch.float64),
        ]
    )
    batch = _batch(n_electrons=0)

    output = PfaffianReadout(pair_channels=2, border_channels=1)(features, batch)

    torch.testing.assert_close(output.logabs, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(output.sign, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(output.aux["pfaffian"], torch.ones(1, dtype=torch.float64))
