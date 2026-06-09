"""Pytest-only equivariance assertions built on typed semantic contracts.

Lives under ``tests/`` (never ``spenn/``). Uses only the allowed typed
contracts -- ``apply_particle_permutation``, ``select_nonidentity_permutations``,
and each value's own ``.compare(...)``. There is no generic tree walking here:
multi-input modules pass an explicit tuple of typed args, each permuted
individually, and particle count comes from explicit typed-value metadata.
"""

from __future__ import annotations

from typing import Any

from spenn.data.equivariant_state import apply_particle_permutation
from spenn.data.permutation import Permutation, select_nonidentity_permutations


def _as_args(inputs: Any) -> tuple[Any, ...]:
    return inputs if isinstance(inputs, tuple) else (inputs,)


def _particle_count_from_args(args: tuple[Any, ...]) -> int | None:
    """Particle count from explicit typed args via ``n_particles``/``n_electrons``.

    Scans only the explicit argument tuple (no recursive container probing) and
    reads each typed value's own particle-count metadata.
    """

    counts: list[int] = []
    for arg in args:
        for attr in ("n_particles", "n_electrons"):
            value = getattr(arg, attr, None)
            if value is not None:
                counts.append(int(value))
                break
    if not counts:
        return None
    first = counts[0]
    for count in counts[1:]:
        if count != first:
            raise ValueError(f"equivariant inputs disagree on particle count: {counts}")
    return first


def assert_equivariant(
    module: Any,
    inputs: Any,
    permutation: Permutation,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> None:
    """Assert ``F(pi x) == pi F(x)`` for one permutation via the normal forward."""

    args = _as_args(inputs)
    output = module(*args)
    permuted_args = tuple(apply_particle_permutation(arg, permutation) for arg in args)
    lhs = module(*permuted_args)
    rhs = apply_particle_permutation(output, permutation)
    close, comparison = lhs.compare(rhs, atol=atol, rtol=rtol)
    assert close, f"equivariance violated for {permutation.image}: {dict(comparison)}"


def assert_equivariant_all(
    module: Any,
    inputs: Any,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> None:
    """Assert equivariance over every non-identity particle permutation."""

    args = _as_args(inputs)
    n_particles = _particle_count_from_args(args)
    if n_particles is None or n_particles < 2:
        return
    permutations = select_nonidentity_permutations(
        n_particles=n_particles, fraction=1.0, max_count=10**9, seed=0
    )
    for permutation in permutations:
        assert_equivariant(module, inputs, permutation, atol=atol, rtol=rtol)
