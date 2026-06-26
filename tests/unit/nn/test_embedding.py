"""Tests for trainable tuple particle-vector embedding."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data.batch import ElectronBatch
from spenn.nn import Embedding


class SliceTupleInputs(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[..., : self.channels]


def test_embedding_feeds_raw_coordinate_tuples_to_order_mlps() -> None:
    batch = ElectronBatch(positions=torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float64))
    embedding = Embedding(
        max_order=2,
        mlps={
            1: SliceTupleInputs(2),
            2: SliceTupleInputs(4),
        },
    )

    feature = embedding(batch)

    assert feature.blocks[0].shape == (1, 0)
    assert feature.blocks[1].shape == (1, 2, 2)
    assert feature.blocks[2].shape == (1, 4, 2, 2)
    torch.testing.assert_close(feature.blocks[1], batch.positions.movedim(-1, 1))
    distinct = torch.tensor([[False, True], [True, False]]).reshape(1, 1, 2, 2)
    expected_slot0 = batch.positions.movedim(-1, 1).unsqueeze(-1).expand(-1, -1, -1, 2) * distinct
    expected_slot1 = batch.positions.movedim(-1, 1).unsqueeze(-2).expand(-1, -1, 2, -1) * distinct
    torch.testing.assert_close(
        feature.blocks[2][:, 0:2],
        expected_slot0,
    )
    torch.testing.assert_close(
        feature.blocks[2][:, 2:4],
        expected_slot1,
    )
    torch.testing.assert_close(feature.blocks[2][:, :, 0, 0], torch.zeros(1, 4, dtype=torch.float64))
    torch.testing.assert_close(feature.blocks[2][:, :, 1, 1], torch.zeros(1, 4, dtype=torch.float64))


def test_embedding_particle_vectors_include_spins_and_aux_features() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        aux={"types": torch.tensor([[[0.5], [1.5]]], dtype=torch.float64)},
    )
    embedding = Embedding(max_order=1, mlps={1: SliceTupleInputs(4)}, aux_feature_keys=("types",))

    feature = embedding(batch)

    expected = torch.cat(
        [batch.positions, batch.spins.unsqueeze(-1), batch.aux["types"]],
        dim=-1,
    ).movedim(-1, 1)
    torch.testing.assert_close(feature.blocks[1], expected)


def test_embedding_builds_mlp_blocks_for_arbitrary_orders() -> None:
    batch = ElectronBatch(positions=torch.arange(1 * 4 * 2, dtype=torch.float64).reshape(1, 4, 2))
    embedding = Embedding(
        max_order=4,
        out_channels={1: 2, 2: 3, 3: 4, 4: 5},
        hidden_channels=7,
        num_hidden_layers=1,
    )

    feature = embedding(batch)

    assert feature.blocks[1].shape == (1, 2, 4)
    assert feature.blocks[2].shape == (1, 3, 4, 4)
    assert feature.blocks[3].shape == (1, 4, 4, 4, 4)
    assert feature.blocks[4].shape == (1, 5, 4, 4, 4, 4)


def test_embedding_rejects_orders_above_particle_count() -> None:
    batch = ElectronBatch(positions=torch.zeros(1, 3, 2, dtype=torch.float64))

    with pytest.raises(ValueError, match="max_order=4 exceeds n_electrons=3"):
        Embedding(max_order=4, out_channels={1: 2, 2: 3, 3: 4, 4: 5})(batch)


def test_embedding_validates_configuration() -> None:
    with pytest.raises(ValueError, match="max_order"):
        Embedding(max_order=0)
    with pytest.raises(ValueError, match="out_channels"):
        Embedding(out_channels=0)
    with pytest.raises(KeyError, match="order 2"):
        Embedding(max_order=2, out_channels={1: 2})
    with pytest.raises(ValueError, match="outside"):
        Embedding(max_order=1, mlps={2: nn.Identity()})


def test_embedding_keeps_dtype_and_allows_gradients() -> None:
    positions = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
    batch = ElectronBatch(positions=positions)
    embedding = Embedding(max_order=2, out_channels=3, hidden_channels=5, num_hidden_layers=1)

    feature = embedding(batch)
    loss = sum(block.sum() for block in feature.blocks)
    loss.backward()

    assert feature.blocks[1].dtype == torch.float64
    assert feature.blocks[2].dtype == torch.float64
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_embedding_flattens_multi_sample_batches() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 3, 4, 5, dtype=torch.float64))

    feature = Embedding(max_order=1, out_channels=6)(batch)

    assert feature.blocks[1].shape == (6, 6, 4)
