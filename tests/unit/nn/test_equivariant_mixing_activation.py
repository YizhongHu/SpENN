"""Owned-activation tests for EquivariantMixing (MIG-TPEN-000 slice a).

Pins the TPEN mixing contract: Gamma applied pointwise to full output blocks
after the bilinear contraction, default identity preserving pre-TPEN behavior
exactly (T3 activation isolation, T12 gradient flow at module level).

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import pytest
import torch

from tpen.data.permutation import all_permutations
from tpen.data.real import Feature, zero_block
from tpen.nn import EquivariantMixing

_DTYPE = torch.float64


def _random_feature(n_particles: int, channels: int, max_order: int, *, seed: int, batch: int = 2) -> Feature:
    generator = torch.Generator().manual_seed(seed)
    blocks: list[torch.Tensor] = [zero_block(batch_size=batch, dtype=_DTYPE)]
    for order in range(1, max_order + 1):
        shape = (batch, channels, *((n_particles,) * order))
        blocks.append(torch.randn(shape, generator=generator, dtype=_DTYPE))
    return Feature(blocks)


def _mixing(activation=None, implementation: str = "slow") -> EquivariantMixing:
    # initial_weight is a shared constant, so two instances with the same
    # arguments are parameter-identical and outputs are directly comparable.
    return EquivariantMixing(
        max_order=2,
        channels=2,
        implementation=implementation,
        activation=activation,
    ).to(dtype=_DTYPE)


def test_default_activation_is_identity() -> None:
    # activation=None must preserve the pre-TPEN output bit-for-bit.
    feature = _random_feature(3, channels=2, max_order=2, seed=41)
    baseline = _mixing(activation=None)(feature)
    explicit = _mixing(activation=torch.nn.Identity())(feature)
    for order in range(1, len(baseline.blocks)):
        torch.testing.assert_close(explicit.blocks[order], baseline.blocks[order])


def test_activation_applies_pointwise_to_full_blocks() -> None:
    # Gamma(mixing(x)) elementwise: the activated module must equal applying
    # the same function to the unactivated output, on every entry.
    feature = _random_feature(3, channels=2, max_order=2, seed=43)
    plain = _mixing(activation=None)(feature)
    activated = _mixing(activation=torch.nn.SiLU())(feature)
    for order in range(1, len(plain.blocks)):
        torch.testing.assert_close(
            activated.blocks[order], torch.nn.functional.silu(plain.blocks[order])
        )


def test_gamma_of_zero_lands_on_non_distinct_entries() -> None:
    # Mixing never writes non-distinct tuples, so with sigmoid (Gamma(0)=0.5)
    # every order-2 diagonal entry must be exactly Gamma(0) — the pinned
    # full-block contract, not an accident.
    feature = _random_feature(3, channels=2, max_order=2, seed=47)
    activated = _mixing(activation=torch.nn.Sigmoid())(feature)
    diagonal = torch.diagonal(activated.blocks[2], dim1=-2, dim2=-1)
    torch.testing.assert_close(diagonal, torch.full_like(diagonal, 0.5))


@pytest.mark.parametrize("activation", [None, torch.nn.SiLU(), torch.nn.Tanh()])
def test_mixing_with_activation_is_equivariant(activation) -> None:
    # T3: equivariance must hold with and without nonlinear Gamma.
    n_particles = 3
    mixing = _mixing(activation=activation)
    feature = _random_feature(n_particles, channels=2, max_order=2, seed=53)
    for permutation in all_permutations(n_particles):
        matches, stats = mixing(feature.permute(permutation)).compare(
            mixing(feature).permute(permutation), atol=1e-12, rtol=1e-12
        )
        assert matches, f"mixing equivariance failed for {permutation}: {stats}"


def test_slow_and_vectorized_agree_with_activation() -> None:
    feature = _random_feature(3, channels=2, max_order=2, seed=59)
    slow = _mixing(activation=torch.nn.SiLU(), implementation="slow")(feature)
    fast = _mixing(activation=torch.nn.SiLU(), implementation="vectorized")(feature)
    matches, stats = slow.compare(fast, atol=1e-12, rtol=1e-12)
    assert matches, f"slow/vectorized divergence with activation: {stats}"


def test_gradients_flow_through_activation_to_all_weights() -> None:
    # T12 at module level: nonlinear Gamma must not detach or zero the path
    # weights' gradients.
    mixing = _mixing(activation=torch.nn.SiLU())
    feature = _random_feature(3, channels=2, max_order=2, seed=61)
    output = mixing(feature)
    loss = sum(block.square().sum() for block in output.blocks[1:])
    loss.backward()
    for name, parameter in mixing.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient not finite"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient identically zero"
