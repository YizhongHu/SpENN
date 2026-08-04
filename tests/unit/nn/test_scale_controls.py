"""Tests for modular SpENN scale-control gates and envelopes."""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch
from spenn.data.permutation import Permutation
from spenn.data.real import Feature, zero_block
from spenn.nn import (
    GaussianCoordinateEnvelope,
    GaussianDecayGate,
    TPENForwardContext,
)


def _feature() -> Feature:
    return Feature(
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


def test_coordinate_envelope_broadcasts_and_is_repeatable() -> None:
    # D14 collapse: the envelope owns its multiply; the former context cache
    # is gone (it was keyed by class-level cache_key, not by sigma), so two
    # calls must simply recompute the same gate.
    batch = _batch()
    feature = _feature()
    context = TPENForwardContext(batch=batch)
    module = GaussianCoordinateEnvelope(sigma=2.0)

    first = module(feature, context)
    second = module(feature, context)

    radius_squared = batch.positions.square().sum(dim=(1, 2))
    expected_gate = torch.exp(-radius_squared / 8.0)
    expected = feature.blocks[1] * expected_gate.reshape(2, 1, 1)
    torch.testing.assert_close(first.blocks[1], expected)
    torch.testing.assert_close(second.blocks[1], expected)


def test_distinct_sigma_envelopes_do_not_share_state() -> None:
    # Regression pin for the pre-D14 footgun: two envelopes with different
    # widths must produce different gates on the same context.
    batch = _batch()
    feature = _feature()
    context = TPENForwardContext(batch=batch)

    wide = GaussianCoordinateEnvelope(sigma=2.0)(feature, context)
    narrow = GaussianCoordinateEnvelope(sigma=0.5)(feature, context)
    assert not torch.allclose(wide.blocks[1], narrow.blocks[1])


def test_coordinate_envelope_is_particle_equivariant() -> None:
    batch = _batch()
    feature = _feature()
    permutation = Permutation((1, 0))
    module = GaussianCoordinateEnvelope(sigma=1.0)

    output = module(feature, TPENForwardContext(batch=batch))
    lhs = module(feature.permute(permutation), TPENForwardContext(batch=batch.permute(permutation)))
    rhs = output.permute(permutation)
    close, comparison = lhs.compare(rhs)
    assert close, dict(comparison)
