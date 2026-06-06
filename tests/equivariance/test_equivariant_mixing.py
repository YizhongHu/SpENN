"""Equivariance tests for slow real-space mixing."""

from __future__ import annotations

import torch

from spenn.data.real import RealFeature, zero_block
from spenn.nn import EquivariantMixing
from spenn.reps.paths import load_default_path_metadata


def _one_channel_feature(values: torch.Tensor) -> RealFeature:
    batch, n_particles = values.shape
    return RealFeature(
        [
            zero_block(batch_size=batch, device=values.device, dtype=values.dtype),
            values.unsqueeze(1),
        ]
    )


def test_slow_mixing_matches_one_body_product_formula() -> None:
    values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    feature = _one_channel_feature(values)
    mixing = EquivariantMixing(max_order=1, max_virtual_order=1, initial_weight=1.0)

    output = mixing(feature)

    assert output.blocks[1].shape == (1, 1, 1, 3)
    torch.testing.assert_close(output.blocks[1][:, 0, 0], values.square())


def test_completion_mean_averages_compatible_virtual_tuples() -> None:
    values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    feature = _one_channel_feature(values)
    mixing = EquivariantMixing(
        max_order=1,
        max_virtual_order=2,
        aggregation="completion_mean",
        initial_weight=1.0,
    )

    output = mixing(feature)

    torch.testing.assert_close(output.blocks[1][:, 0, 0], values.square())
    expected_order_two_completion = torch.tensor([[3.0, 5.0, 6.0]], dtype=torch.float64)
    torch.testing.assert_close(output.blocks[1][:, 0, 1], expected_order_two_completion)
    torch.testing.assert_close(output.blocks[1][:, 0, 2], expected_order_two_completion)


def test_mixing_default_paths_come_from_saved_metadata() -> None:
    mixing = EquivariantMixing(max_order=2, max_virtual_order=2, output_embedding="full")
    metadata = load_default_path_metadata("full")
    expected = [
        path
        for path in metadata.all_paths()
        if path.s <= 2 and path.m <= 2 and path.m1 <= 2 and path.m2 <= 2
    ]

    assert [path.as_tuple() for path in mixing.paths] == [path.as_tuple() for path in expected]


def test_slow_mixing_passes_forced_runtime_equivariance_check() -> None:
    generator = torch.Generator().manual_seed(4321)
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 3, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 3, 3, generator=generator, dtype=torch.float64),
        ]
    )
    mixing = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        equivariance_check=True,
        check_probability=1.0,
        tensor_validation_check=True,
    )

    output = mixing(feature)

    assert output.validate() is output


def test_vectorized_mixing_matches_slow_reference_for_all_aggregations() -> None:
    generator = torch.Generator().manual_seed(2026)
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 4, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 4, 4, generator=generator, dtype=torch.float64),
        ]
    )
    other = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 3, 4, generator=generator, dtype=torch.float64),
            torch.randn(1, 3, 4, 4, generator=generator, dtype=torch.float64),
        ]
    )
    out_channels = {1: 4, 2: 5}

    for output_embedding in ("canonical", "full"):
        for aggregation in ("sum", "completion_mean"):
            slow = EquivariantMixing(
                max_order=2,
                max_virtual_order=2,
                output_embedding=output_embedding,
                aggregation=aggregation,
                out_channels=out_channels,
                implementation="slow",
                initial_weight=0.5,
            )
            vectorized = EquivariantMixing(
                max_order=2,
                max_virtual_order=2,
                output_embedding=output_embedding,
                aggregation=aggregation,
                out_channels=out_channels,
                implementation="vectorized",
                initial_weight=0.5,
            )

            slow_output = slow(feature, other)
            vectorized_output = vectorized(feature, other)

            torch.testing.assert_close(vectorized_output.blocks, slow_output.blocks)


def test_vectorized_mixing_passes_forced_runtime_equivariance_check() -> None:
    generator = torch.Generator().manual_seed(31415)
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 3, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 3, 3, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 3, 3, 3, generator=generator, dtype=torch.float64),
        ]
    )
    mixing = EquivariantMixing(
        max_order=3,
        max_virtual_order=3,
        aggregation="completion_mean",
        implementation="vectorized",
        equivariance_check=True,
        check_probability=1.0,
        tensor_validation_check=True,
    )

    output = mixing(feature)

    assert output.validate() is output
