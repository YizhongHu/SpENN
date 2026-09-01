"""Tests for the slow unary linear support-path reference."""

from __future__ import annotations

import pytest
import torch

from tpen.data.paths import LinearPathMetadata, LinearPathPolicy, SupportPath
from tpen.data.real import Feature, zero_block
from tpen.nn import LinearEquivariantMixing
from tpen.nn.mixing_kernel import Aggregation


def _feature(values: torch.Tensor, order: int = 1) -> Feature:
    blocks = [zero_block(batch_size=values.shape[0], dtype=values.dtype)]
    blocks.extend(torch.zeros(values.shape[0], 1, *(values.shape[-1:] * current_order), dtype=values.dtype) for current_order in range(1, order))
    blocks.append(values.unsqueeze(1))
    return Feature(blocks)


def test_coordinate_neighbor_metadata_has_static_paths() -> None:
    metadata = LinearPathMetadata.generate(max_order=2)

    assert metadata.policy is LinearPathPolicy.COORDINATE_NEIGHBOR
    assert [len(metadata.paths_for_output_order(order)) for order in (1, 2)] == [2, 3]
    assert metadata.paths_for_output_order(2)[1].tau_in == (2, 1)
    assert metadata.paths_for_output_order(2)[2].tau_in == (0, 2)


@pytest.mark.parametrize("aggregation", [Aggregation.SUM, Aggregation.COMPLETION_MEAN])
def test_linear_one_body_formula_and_parameter_static(aggregation: Aggregation) -> None:
    values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    mixing = LinearEquivariantMixing(max_order=1, channels=1, aggregation=aggregation).to(dtype=torch.float64)
    before = tuple(mixing.state_dict())
    output = mixing(_feature(values))
    paths = mixing.metadata.paths_for_output_order(1)
    identity_index = paths.index(SupportPath(1, 1, (0,), (0,)))
    disjoint_index = paths.index(SupportPath(1, 1, (0,), (1,)))

    assert output.blocks[1].shape == (1, 1, 2, 3)
    torch.testing.assert_close(output.blocks[1][0, 0, identity_index], values[0])
    expected = torch.tensor([6.0, 5.0, 3.0], dtype=values.dtype)
    if aggregation is Aggregation.COMPLETION_MEAN:
        expected = torch.tensor([3.0, 2.5, 1.5], dtype=values.dtype)
    torch.testing.assert_close(output.blocks[1][0, 0, disjoint_index], expected)
    assert tuple(mixing.state_dict()) == before


def test_linear_empty_orbit_and_zero_particles_are_zero_safe() -> None:
    metadata = LinearPathMetadata.generate(max_order=1, policy=LinearPathPolicy.ORBIT_COMPLETE)
    mixing = LinearEquivariantMixing(max_order=1, channels=1, metadata=metadata, aggregation="completion_mean")
    paths = metadata.paths_for_output_order(1)
    disjoint_index = paths.index(SupportPath(1, 1, (0,), (1,)))
    identity_index = paths.index(SupportPath(1, 1, (0,), (0,)))
    one_particle = mixing(Feature([zero_block(batch_size=1), torch.ones(1, 1, 1)]))
    assert one_particle.blocks[1].shape == (1, 1, 2, 1)
    assert torch.isfinite(one_particle.blocks[1]).all()
    torch.testing.assert_close(one_particle.blocks[1][0, 0, identity_index], torch.ones(1))
    torch.testing.assert_close(one_particle.blocks[1][0, 0, disjoint_index], torch.zeros(1))

    output = mixing(Feature([zero_block(batch_size=1), torch.empty(1, 1, 0)]))

    assert output.blocks[1].shape == (1, 1, 2, 0)
    assert torch.isfinite(output.blocks[1]).all()
    torch.testing.assert_close(output.blocks[1], torch.zeros_like(output.blocks[1]))


def test_linear_explicit_metadata_preserves_declared_order() -> None:
    paths = (
        SupportPath(1, 1, (0,), (1,), "sum"),
        SupportPath(1, 1, (0,), (0,), "sum"),
    )
    metadata = LinearPathMetadata.generate(max_order=1, policy="explicit", explicit=paths)
    assert metadata.all_paths() == paths
