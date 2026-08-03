"""Exhaustive permutation-equivariance tests for real-space PathAggregation.

T2/T3 analogue on the fast module (MIG-TPEN-000 §2.2, slice b1): the path
contraction touches only inert axes, so ``F(pi x) == pi F(x)`` must hold
exhaustively for n in {2, 3, 4}, with identity and nonlinear ``Gamma_c``
alike.
Follows the typed ``.permute``/``.compare`` oracle pattern established in
``tests/unit/nn/test_tpen_reference_oracle.py``.

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from spenn.data.permutation import all_permutations
from spenn.data.real import RealInteraction, zero_block
from spenn.nn import PathAggregation, TorchInitializer
from tests.helpers.equivariance import assert_equivariant_all

_DTYPE = torch.float64

_PATHS_BY_ORDER = {1: 3, 2: 4}

# None exercises the identity Gamma_c; SiLU exercises a nonlinear owned
# activation (activation isolation must not affect equivariance).
_ACTIVATIONS: dict[str, torch.nn.Module | None] = {
    "identity": None,
    "silu": torch.nn.SiLU(),
}


def _random_interaction(
    n_particles: int,
    channels: int,
    paths_by_order: Mapping[int, int],
    *,
    seed: int,
    batch: int = 2,
) -> RealInteraction:
    generator = torch.Generator().manual_seed(seed)
    max_paths = max(paths_by_order.values())
    blocks: list[torch.Tensor] = [zero_block(batch_size=batch, paths=max_paths, dtype=_DTYPE)]
    for order, paths in sorted(paths_by_order.items()):
        shape = (batch, channels, paths, *((n_particles,) * order))
        blocks.append(torch.randn(shape, generator=generator, dtype=_DTYPE))
    return RealInteraction(blocks)


def _module(activation: torch.nn.Module | None, *, seed: int) -> PathAggregation:
    return PathAggregation(
        max_order=2,
        channels=2,
        path_counts_by_order=_PATHS_BY_ORDER,
        activation=activation,
        initializer=TorchInitializer(seed=seed),
    ).to(dtype=_DTYPE)


@pytest.mark.parametrize("n_particles", [2, 3, 4])
@pytest.mark.parametrize("activation_name", sorted(_ACTIVATIONS))
def test_path_aggregation_is_equivariant_for_all_permutations(
    n_particles: int, activation_name: str
) -> None:
    module = _module(_ACTIVATIONS[activation_name], seed=53)
    interaction = _random_interaction(
        n_particles, channels=2, paths_by_order=_PATHS_BY_ORDER, seed=11
    )

    for permutation in all_permutations(n_particles):
        permuted_first = module(interaction.permute(permutation))
        permuted_last = module(interaction).permute(permutation)
        matches, stats = permuted_first.compare(permuted_last, atol=1e-12, rtol=1e-12)
        assert matches, (
            f"PathAggregation equivariance failed for {permutation} "
            f"with {activation_name} activation: {stats}"
        )


def test_path_aggregation_passes_forced_runtime_equivariance_check() -> None:
    # Preserved intent from the irrep-era suite: the output validates fluently
    # and the shared pytest equivariance helper agrees with the typed loop.
    module = _module(torch.nn.SiLU(), seed=59)
    interaction = _random_interaction(3, channels=2, paths_by_order=_PATHS_BY_ORDER, seed=17)

    output = module(interaction)

    assert output.validate() is output
    assert_equivariant_all(module, interaction, atol=1e-12, rtol=1e-12)
