"""Tests for real-space learned path aggregation (MIG-TPEN-000 §2.2, slice b1).

Covers the TPEN aggregation contract: per-order weight shapes and keys,
channel/path-count mismatch rejection, explicit versus default-metadata path
counts, the owned elementwise activation ``Gamma_c``, and the fast-vs-slow
oracle gate T1 against ``tests.helpers.tpen_reference.slow_tpen_aggregation``.

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from spenn.data.real import RealInteraction, RealUpdate, zero_block
from spenn.nn import PathAggregation, TorchInitializer
from spenn.reps.paths import load_default_path_metadata
from tests.helpers.tpen_reference import slow_tpen_aggregation

_DTYPE = torch.float64


def _random_interaction(
    n_particles: int,
    channels: int,
    paths_by_order: Mapping[int, int],
    *,
    seed: int,
    batch: int = 2,
) -> RealInteraction:
    # Same construction as the oracle suite: fixed torch.Generator seed,
    # float64 blocks [batch, channels, paths, indices...].
    generator = torch.Generator().manual_seed(seed)
    max_paths = max(paths_by_order.values())
    blocks: list[torch.Tensor] = [zero_block(batch_size=batch, paths=max_paths, dtype=_DTYPE)]
    for order, paths in sorted(paths_by_order.items()):
        shape = (batch, channels, paths, *((n_particles,) * order))
        blocks.append(torch.randn(shape, generator=generator, dtype=_DTYPE))
    return RealInteraction(blocks)


def _module(
    channels: int | Mapping[int, int],
    paths_by_order: Mapping[int, int],
    *,
    activation: torch.nn.Module | None = None,
    seed: int = 0,
) -> PathAggregation:
    # Explicit initializer keeps construction deterministic and side-effect
    # free (no global RNG mutation) for every test in this suite.
    return PathAggregation(
        max_order=max(paths_by_order),
        channels=channels,
        path_counts_by_order=paths_by_order,
        activation=activation,
        initializer=TorchInitializer(seed=seed),
    ).to(dtype=_DTYPE)


def test_path_aggregation_registers_one_weight_per_order_with_stable_keys() -> None:
    module = _module({1: 2, 2: 3}, {1: 4, 2: 5})

    assert module.key(1) == "o1"
    assert module.key(2) == "o2"
    assert set(module.weights.keys()) == {"o1", "o2"}
    # Weight U^{(m)} has shape [channels, paths]: shared over batch and tuple
    # positions, never mixing channels or particle indices.
    assert tuple(module.weights["o1"].shape) == (2, 4)
    assert tuple(module.weights["o2"].shape) == (3, 5)


def test_path_aggregation_removes_path_axis_and_selects_learned_path() -> None:
    # One-hot path weights must select exactly one path slice: the contraction
    # is u[c, I] = sum_p U[c, p] h[c, p, I] with no channel or index mixing.
    tensor = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=_DTYPE).reshape(1, 1, 2, 3)
    interaction = RealInteraction([zero_block(batch_size=1, paths=2, dtype=_DTYPE), tensor])
    module = _module(1, {1: 2})

    with torch.no_grad():
        module.weights[module.key(1)].zero_()
        module.weights[module.key(1)][0, 0] = 1.0
    path_zero = module(interaction)[1]

    with torch.no_grad():
        module.weights[module.key(1)].zero_()
        module.weights[module.key(1)][0, 1] = 1.0
    path_one = module(interaction)[1]

    assert path_zero.shape == (1, 1, 3)
    torch.testing.assert_close(path_zero, tensor[:, :, 0])
    torch.testing.assert_close(path_one, tensor[:, :, 1])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(path_zero, path_one)


def test_path_aggregation_returns_real_update_with_channels_preserved() -> None:
    # Contract: RealInteraction [batch, channels, paths, indices...] in,
    # RealUpdate [batch, channels, indices...] out, with C_out == C_in.
    paths_by_order = {1: 3, 2: 4}
    interaction = _random_interaction(3, channels=2, paths_by_order=paths_by_order, seed=3)
    module = _module(2, paths_by_order)

    output = module(interaction)

    assert isinstance(output, RealUpdate)
    assert output.validate() is output
    assert output[1].shape == (2, 2, 3)
    assert output[2].shape == (2, 2, 3, 3)


def test_path_aggregation_rejects_input_channels_that_disagree_with_config() -> None:
    interaction = _random_interaction(2, channels=3, paths_by_order={1: 2}, seed=5)
    module = _module(2, {1: 2})

    with pytest.raises(ValueError, match="channels are 3, expected 2"):
        module(interaction)


def test_path_aggregation_rejects_path_count_mismatch() -> None:
    # T5-style negative case on the fast module: a path-count disagreement
    # must raise, never broadcast or truncate.
    interaction = _random_interaction(2, channels=2, paths_by_order={1: 3}, seed=7)
    module = _module(2, {1: 4})

    with pytest.raises(ValueError, match="path count is 3, expected 4"):
        module(interaction)


def test_path_aggregation_requires_explicit_path_counts_for_all_orders() -> None:
    with pytest.raises(ValueError, match="missing orders"):
        PathAggregation(max_order=2, channels=1, path_counts_by_order={2: 1})


def test_path_aggregation_derives_path_counts_from_default_metadata() -> None:
    max_order = max_virtual_order = 2
    module = PathAggregation(
        max_order=max_order,
        channels=2,
        max_virtual_order=max_virtual_order,
        initializer=TorchInitializer(seed=0),
    )

    # Independent restatement of the documented filter over the checked-in
    # canonical metadata: order-m paths with s and input orders in range.
    metadata = load_default_path_metadata("canonical")
    expected = {
        order: sum(
            1
            for path in metadata.all_paths()
            if path.m == order
            and path.s <= max_virtual_order
            and path.m1 <= max_order
            and path.m2 <= max_order
        )
        for order in range(1, max_order + 1)
    }

    assert all(count > 0 for count in expected.values())
    assert module.path_counts_by_order == expected
    for order, paths in expected.items():
        assert tuple(module.weights[module.key(order)].shape) == (2, paths)


def test_path_aggregation_zero_path_orders_yield_zero_updates() -> None:
    # Zero-path orders keep an eager empty weight and produce a zero update
    # block instead of failing lazily at forward time.
    module = _module(2, {1: 0})
    tensor = torch.empty(1, 2, 0, 3, dtype=_DTYPE)
    interaction = RealInteraction([zero_block(batch_size=1, paths=0, dtype=_DTYPE), tensor])

    output = module(interaction)

    assert tuple(module.weights[module.key(1)].shape) == (2, 0)
    torch.testing.assert_close(output[1], torch.zeros(1, 2, 3, dtype=_DTYPE))


def test_path_aggregation_applies_owned_activation_elementwise() -> None:
    # Gamma_c ownership: with identical weights, the activated module output
    # must equal the plain module output passed through SiLU elementwise.
    paths_by_order = {1: 3, 2: 4}
    interaction = _random_interaction(3, channels=2, paths_by_order=paths_by_order, seed=11)
    plain = _module(2, paths_by_order, activation=None, seed=41)
    activated = _module(2, paths_by_order, activation=torch.nn.SiLU(), seed=43)
    with torch.no_grad():
        for order in paths_by_order:
            activated.weights[activated.key(order)].copy_(plain.weights[plain.key(order)])

    plain_output = plain(interaction)
    activated_output = activated(interaction)

    for order in paths_by_order:
        torch.testing.assert_close(
            activated_output[order],
            torch.nn.functional.silu(plain_output[order]),
        )


@pytest.mark.parametrize("activation_name", ["identity", "silu"])
def test_path_aggregation_matches_slow_tpen_reference(activation_name: str) -> None:
    # Oracle gate T1: the fast module must agree with the literal-loop
    # reference when both use the same weights, on random float64 input.
    module_activation = None if activation_name == "identity" else torch.nn.SiLU()
    reference_activation = (lambda t: t) if activation_name == "identity" else torch.nn.functional.silu
    paths_by_order = {1: 3, 2: 4}
    channels = 2
    interaction = _random_interaction(3, channels=channels, paths_by_order=paths_by_order, seed=13)
    module = _module(channels, paths_by_order, activation=module_activation, seed=47)
    # Copy the module weights into the reference's order-indexed list.
    weights: list[torch.Tensor | None] = [None]
    for order in sorted(paths_by_order):
        weights.append(module.weights[module.key(order)].detach().clone())

    fast = module(interaction)
    slow = slow_tpen_aggregation(interaction, weights, reference_activation)

    expected = RealUpdate(list(slow.blocks))
    matches, stats = fast.compare(expected, atol=1e-12, rtol=1e-12)
    assert matches, f"fast PathAggregation disagrees with slow_tpen_aggregation: {stats}"
