"""Tests for modular SpENN scale-control gates and envelopes."""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch
from spenn.data.permutation import Permutation
from spenn.data.real import RealFeature, zero_block
from spenn.nn import (
    GaussianCoordinateEnvelope,
    GaussianDecayGate,
    RealCoordinateEnvelope,
    SpENNForwardContext,
)


def _feature() -> RealFeature:
    return RealFeature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 3.0], [4.0, 5.0]],
                ],
                dtype=torch.float64,
            ),
        ]
    )


def _batch() -> ElectronBatch:
    return ElectronBatch(
        positions=torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        ),
        spins=torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64),
    )


def test_gaussian_decay_gate_matches_formula() -> None:
    x = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)

    torch.testing.assert_close(GaussianDecayGate(sigma=2.0)(x), torch.exp(-x / 8.0))


def test_coordinate_envelope_broadcasts_and_reuses_context_cache() -> None:
    batch = _batch()
    feature = _feature()
    context = SpENNForwardContext(batch=batch)
    module = RealCoordinateEnvelope(GaussianCoordinateEnvelope(sigma=2.0))

    first = module(feature, context)
    cached = context.coordinate_envelopes["gaussian"]
    second = module(feature, context)

    radius_squared = batch.positions.square().sum(dim=(1, 2))
    expected_gate = torch.exp(-radius_squared / 8.0)
    expected = feature.blocks[1] * expected_gate.reshape(2, 1, 1)
    torch.testing.assert_close(cached, expected_gate)
    torch.testing.assert_close(first.blocks[1], expected)
    torch.testing.assert_close(second.blocks[1], expected)


def test_coordinate_envelope_is_particle_equivariant() -> None:
    batch = _batch()
    feature = _feature()
    permutation = Permutation((1, 0))
    module = RealCoordinateEnvelope(GaussianCoordinateEnvelope(sigma=1.0))

    output = module(feature, SpENNForwardContext(batch=batch))
    lhs = module(feature.permute(permutation), SpENNForwardContext(batch=batch.permute(permutation)))
    rhs = output.permute(permutation)
    close, comparison = lhs.compare(rhs)
    assert close, dict(comparison)
