"""Tests for real-space tensor container permutation actions."""

from __future__ import annotations

import torch

from spenn.data.base import ConcatenatedState
from spenn.data.batch import ElectronBatch
from spenn.data.permutation import Permutation
from spenn.data.real_features import RealConcatenatedState, RealFeature, RealMessage
from spenn.testing import assert_tree_allclose


def _feature() -> RealFeature:
    return RealFeature(
        [
            torch.tensor([[1.0, 2.0]]),
            torch.arange(2 * 2 * 3, dtype=torch.float64).reshape(2, 2, 3),
            torch.arange(2 * 2 * 3 * 3, dtype=torch.float64).reshape(2, 2, 3, 3),
        ]
    )


def _message() -> RealMessage:
    return RealMessage(
        [
            torch.tensor([[3.0, 4.0]]),
            torch.arange(2 * 1 * 3, dtype=torch.float64).reshape(2, 1, 3),
        ]
    )


def test_real_feature_identity_returns_equal_new_state() -> None:
    feature = _feature()

    permuted = feature.permute(Permutation.identity(3))

    assert permuted is not feature
    assert permuted.data is not feature.data
    assert_tree_allclose(permuted, feature)


def test_real_feature_permute_matches_axis_indexing() -> None:
    feature = _feature()
    permutation = Permutation((2, 0, 1))

    permuted = feature.permute(permutation)

    index = torch.tensor(permutation.inverse().image)
    assert torch.equal(permuted[1], feature[1].index_select(2, index))
    assert torch.equal(permuted[2], feature[2].index_select(2, index).index_select(3, index))
    assert torch.equal(permuted[1][:, :, permutation.apply_index(0)], feature[1][:, :, 0])


def test_real_message_and_real_concatenated_state_permute() -> None:
    state = RealConcatenatedState(features=_feature(), messages=_message())
    permutation = Permutation((1, 2, 0))

    permuted = state.permute(permutation)

    assert isinstance(permuted.features, RealFeature)
    assert isinstance(permuted.messages, RealMessage)
    assert_tree_allclose(permuted.features, state.features.permute(permutation))
    assert_tree_allclose(permuted.messages, state.messages.permute(permutation))


def test_generic_concatenated_state_permute() -> None:
    state = ConcatenatedState((_feature(), _message()))
    permutation = Permutation((1, 2, 0))

    permuted = state.permute(permutation)

    assert isinstance(permuted, ConcatenatedState)
    assert_tree_allclose(permuted[0], state[0].permute(permutation))
    assert_tree_allclose(permuted[1], state[1].permute(permutation))


def test_electron_batch_permute_moves_electron_axis() -> None:
    batch = ElectronBatch(
        positions=torch.arange(2 * 3 * 2, dtype=torch.float64).reshape(2, 3, 2),
        spins=torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]),
    )
    permutation = Permutation((2, 0, 1))

    permuted = batch.permute(permutation)

    index = torch.tensor(permutation.inverse().image)
    assert torch.equal(permuted.positions, batch.positions.index_select(-2, index))
    assert torch.equal(permuted.spins, batch.spins.index_select(-1, index))


def test_real_state_permutation_composition() -> None:
    feature = _feature()
    first = Permutation((1, 0, 2))
    second = Permutation((2, 1, 0))

    sequential = feature.permute(first).permute(second)
    composed = feature.permute(second.compose(first))

    assert_tree_allclose(sequential, composed)


def test_permutation_does_not_mutate_original_tensors() -> None:
    feature = _feature()
    original_blocks = [tensor.clone() for tensor in feature.data]

    _ = feature.permute(Permutation((2, 0, 1)))

    for actual, expected in zip(feature.data, original_blocks):
        assert torch.equal(actual, expected)
