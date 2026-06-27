"""Tests for explicit SpENN neural initializers."""

from __future__ import annotations

import torch

from spenn.data.partition import Partition
from spenn.nn import Embedding, MLP, PathAggregation, SeededLinear, TorchInitializer


def _state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _assert_state_dicts_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state = _state_dict(left)
    right_state = _state_dict(right)
    assert left_state.keys() == right_state.keys()
    for key, left_value in left_state.items():
        torch.testing.assert_close(left_value, right_state[key])


def test_seeded_linear_is_deterministic_without_global_rng_mutation() -> None:
    initializer = TorchInitializer(seed=17)
    torch.manual_seed(999)
    before = torch.get_rng_state()

    first = SeededLinear(3, 4, initializer=initializer)

    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.manual_seed(123)
    second = SeededLinear(3, 4, initializer=TorchInitializer(seed=17))

    _assert_state_dicts_equal(first, second)


def test_initializer_seed_changes_initialized_parameters() -> None:
    first = SeededLinear(3, 4, initializer=TorchInitializer(seed=17))
    second = SeededLinear(3, 4, initializer=TorchInitializer(seed=18))

    with torch.no_grad():
        assert not torch.equal(first.weight, second.weight)


def test_mlp_uses_explicit_initializer_without_global_rng_mutation() -> None:
    torch.manual_seed(999)
    before = torch.get_rng_state()

    first = MLP(3, 2, hidden_channels=5, num_hidden_layers=2, initializer=TorchInitializer(seed=23))

    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.manual_seed(123)
    second = MLP(3, 2, hidden_channels=5, num_hidden_layers=2, initializer=TorchInitializer(seed=23))

    _assert_state_dicts_equal(first, second)


def test_embedding_passes_initializer_to_generated_order_mlps() -> None:
    first = Embedding(
        max_order=2,
        spatial_dim=3,
        out_channels=4,
        hidden_channels=5,
        num_hidden_layers=1,
        include_spins=False,
        initializer=TorchInitializer(seed=29),
    )
    second = Embedding(
        max_order=2,
        spatial_dim=3,
        out_channels=4,
        hidden_channels=5,
        num_hidden_layers=1,
        include_spins=False,
        initializer=TorchInitializer(seed=29),
    )

    _assert_state_dicts_equal(first, second)


def test_path_aggregation_uses_explicit_initializer_without_global_rng_mutation() -> None:
    partition = Partition((1,))
    torch.manual_seed(999)
    before = torch.get_rng_state()

    first = PathAggregation(
        max_order=1,
        channels=2,
        channel_out_by_order=3,
        path_counts_by_order={1: 2},
        partitions=(partition,),
        initializer=TorchInitializer(seed=31),
    )

    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.manual_seed(123)
    second = PathAggregation(
        max_order=1,
        channels=2,
        channel_out_by_order=3,
        path_counts_by_order={1: 2},
        partitions=(partition,),
        initializer=TorchInitializer(seed=31),
    )

    _assert_state_dicts_equal(first, second)
