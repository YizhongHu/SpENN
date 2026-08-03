"""TPEN oracle-contract tests over the slow reference (MIG-TPEN-000 §5).

Covers the gates that are executable before any model rewiring:

- T2: exhaustive small-n permutation equivariance of the reference
  aggregation and the composed reference layer.
- T3: activation isolation — equivariance holds for identity and nonlinear
  Gamma / Gamma_c alike.
- T5 (negative): the aggregation rejects a path-count/weight-shape mismatch.
- T6: the B1 per-channel Pfaffian readout is pinned by a hand-computed
  two-channel case and shown to differ from the rejected mixed-kernel form.
- T12: gradients flow to every mixing weight and aggregation weight.

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import pytest
import torch

from spenn.data.permutation import all_permutations
from spenn.data.real import RealFeature, RealInteraction, zero_block
from spenn.nn import EquivariantMixing
from tests.helpers.tpen_reference import (
    mixed_kernel_pfaffian_readout,
    per_channel_pfaffian_readout,
    slow_tpen_aggregation,
    slow_tpen_layer,
)

_DTYPE = torch.float64


def _random_feature(n_particles: int, channels: int, max_order: int, *, seed: int, batch: int = 2) -> RealFeature:
    generator = torch.Generator().manual_seed(seed)
    blocks: list[torch.Tensor] = [zero_block(batch_size=batch, dtype=_DTYPE)]
    for order in range(1, max_order + 1):
        shape = (batch, channels, *((n_particles,) * order))
        blocks.append(torch.randn(shape, generator=generator, dtype=_DTYPE))
    return RealFeature(blocks)


def _random_interaction(
    n_particles: int, channels: int, paths_by_order: dict[int, int], *, seed: int, batch: int = 2
) -> RealInteraction:
    generator = torch.Generator().manual_seed(seed)
    max_paths = max(paths_by_order.values())
    blocks: list[torch.Tensor] = [zero_block(batch_size=batch, paths=max_paths, dtype=_DTYPE)]
    for order, paths in sorted(paths_by_order.items()):
        shape = (batch, channels, paths, *((n_particles,) * order))
        blocks.append(torch.randn(shape, generator=generator, dtype=_DTYPE))
    return RealInteraction(blocks)


def _random_path_weights(channels: int, paths_by_order: dict[int, int], *, seed: int) -> list[torch.Tensor | None]:
    generator = torch.Generator().manual_seed(seed)
    weights: list[torch.Tensor | None] = [None]
    for order in sorted(paths_by_order):
        weights.append(torch.randn((channels, paths_by_order[order]), generator=generator, dtype=_DTYPE))
    return weights


_ACTIVATIONS = {
    "identity": lambda t: t,
    "silu": torch.nn.functional.silu,
    "tanh": torch.tanh,
}


@pytest.mark.parametrize("n_particles", [2, 3])
@pytest.mark.parametrize("activation_name", sorted(_ACTIVATIONS))
def test_reference_aggregation_is_equivariant_for_all_permutations(
    n_particles: int, activation_name: str
) -> None:
    # T2/T3: aggregation contracts only inert axes, so equivariance must hold
    # for identity and nonlinear Gamma_c alike, over every permutation.
    activation = _ACTIVATIONS[activation_name]
    paths_by_order = {1: 3, 2: 4}
    interaction = _random_interaction(n_particles, channels=2, paths_by_order=paths_by_order, seed=11)
    weights = _random_path_weights(2, paths_by_order, seed=13)

    for permutation in all_permutations(n_particles):
        permuted_first = slow_tpen_aggregation(interaction.permute(permutation), weights, activation)
        permuted_last = slow_tpen_aggregation(interaction, weights, activation).permute(permutation)
        matches, stats = permuted_first.compare(permuted_last, atol=1e-12, rtol=1e-12)
        assert matches, f"aggregation equivariance failed for {permutation}: {stats}"


@pytest.mark.parametrize("activation_name", sorted(_ACTIVATIONS))
def test_reference_layer_is_equivariant_for_all_permutations(activation_name: str) -> None:
    # T2/T3 on the composed layer: mixing (slow) -> Gamma -> aggregation ->
    # residual update. Exhaustive over S_3.
    activation = _ACTIVATIONS[activation_name]
    n_particles, channels = 3, 2
    mixing = EquivariantMixing(max_order=2, channels=channels, implementation="slow").to(dtype=_DTYPE)
    feature = _random_feature(n_particles, channels, max_order=2, seed=17)
    paths_by_order = {
        order: sum(1 for path in mixing.paths if path.m == order) for order in (1, 2)
    }
    weights = _random_path_weights(channels, paths_by_order, seed=19)

    def layer(x: RealFeature) -> RealFeature:
        return slow_tpen_layer(
            x,
            mixing=mixing,
            mixing_activation=activation,
            path_weights=weights,
            aggregation_activation=activation,
        )

    for permutation in all_permutations(n_particles):
        matches, stats = layer(feature.permute(permutation)).compare(
            layer(feature).permute(permutation), atol=1e-10, rtol=1e-10
        )
        assert matches, f"layer equivariance failed for {permutation}: {stats}"


def test_aggregation_rejects_path_count_mismatch() -> None:
    # T5 negative: a weight whose path axis disagrees with the interaction
    # must raise, never broadcast or truncate.
    interaction = _random_interaction(2, channels=2, paths_by_order={1: 3}, seed=23)
    bad_weights: list[torch.Tensor | None] = [None, torch.ones((2, 4), dtype=_DTYPE)]
    with pytest.raises(ValueError, match="does not match"):
        slow_tpen_aggregation(interaction, bad_weights, lambda t: t)


def test_aggregation_rejects_missing_order_weight() -> None:
    interaction = _random_interaction(2, channels=1, paths_by_order={1: 2}, seed=29)
    with pytest.raises(ValueError, match="missing the order-1 path weight"):
        slow_tpen_aggregation(interaction, [None, None], lambda t: t)


def test_per_channel_pfaffian_matches_hand_computed_case_and_differs_from_mixed_kernel() -> None:
    # T6 (B1): Pf of a 4x4 skew matrix is a12*a34 - a13*a24 + a14*a23.
    # Two channels with different kernels distinguish
    # sum_c w_c Pf[skew_c]  (adopted)   from   Pf[sum_c w_c skew_c] (rejected).
    def skew_from_upper(a12, a13, a14, a23, a24, a34):
        matrix = torch.zeros((4, 4), dtype=_DTYPE)
        upper = {(0, 1): a12, (0, 2): a13, (0, 3): a14, (1, 2): a23, (1, 3): a24, (2, 3): a34}
        for (row, col), value in upper.items():
            matrix[row, col] = value
            matrix[col, row] = -value
        return matrix

    kernel_a = skew_from_upper(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    kernel_b = skew_from_upper(2.0, -1.0, 0.5, 1.5, -2.0, 3.0)
    pf_a = 1.0 * 6.0 - 2.0 * 5.0 + 3.0 * 4.0  # = 8
    pf_b = 2.0 * 3.0 - (-1.0) * (-2.0) + 0.5 * 1.5  # = 4.75
    weights = torch.tensor([0.75, -1.25], dtype=_DTYPE)

    # The reference consumes raw (not yet skew) order-2 blocks; feeding the
    # skew kernels directly is fine because skew(skew(A)) == skew(A).
    pair = torch.stack([kernel_a, kernel_b]).unsqueeze(0)
    features = RealFeature(
        [
            zero_block(batch_size=1, dtype=_DTYPE),
            torch.zeros((1, 2, 4), dtype=_DTYPE),
            pair,
        ]
    )

    adopted = per_channel_pfaffian_readout(features, weights)
    expected = weights[0] * pf_a + weights[1] * pf_b
    torch.testing.assert_close(adopted, expected.reshape(1))

    rejected = mixed_kernel_pfaffian_readout(features, weights)
    assert not torch.allclose(adopted, rejected), (
        "per-channel and mixed-kernel readouts must differ on this case; "
        "if they agree the T6 pin is not distinguishing the function classes"
    )


def test_gradients_flow_to_all_mixing_and_aggregation_weights() -> None:
    # T12: backward through the layer output must reach every registered
    # mixing weight and every aggregation path weight with a finite,
    # not-identically-zero gradient.
    n_particles, channels = 3, 2
    mixing = EquivariantMixing(max_order=2, channels=channels, implementation="slow").to(dtype=_DTYPE)
    feature = _random_feature(n_particles, channels, max_order=2, seed=31)
    paths_by_order = {
        order: sum(1 for path in mixing.paths if path.m == order) for order in (1, 2)
    }
    weights = _random_path_weights(channels, paths_by_order, seed=37)
    for weight in weights[1:]:
        assert weight is not None
        weight.requires_grad_(True)

    output = slow_tpen_layer(
        feature,
        mixing=mixing,
        mixing_activation=torch.nn.functional.silu,
        path_weights=weights,
        aggregation_activation=torch.tanh,
    )
    loss = sum(block.square().sum() for block in output.blocks[1:])
    loss.backward()

    for name, parameter in mixing.named_parameters():
        assert parameter.grad is not None, f"mixing parameter {name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"mixing parameter {name} gradient not finite"
        assert parameter.grad.abs().sum() > 0, f"mixing parameter {name} gradient identically zero"
    for order, weight in enumerate(weights):
        if weight is None:
            continue
        assert weight.grad is not None, f"aggregation order-{order} weight received no gradient"
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
