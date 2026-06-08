"""Runtime checker protocol and equivariance checker for training callbacks.

A `RuntimeChecker` decides *how* to check a `TrainerState`; the scheduling of
*when* to run lives in `spenn.callback.RuntimeEquivariance`. Checkers return a
`CheckResult` carrying a name, pass/fail, and scalar metrics for logging.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

import torch

from spenn.data.equivariant_state import infer_particle_count, permute_tree
from spenn.data.permutation import Permutation
from spenn.testing.equivariance import assert_tree_allclose


@dataclass
class CheckResult:
    """Result of one runtime check.

    Parameters
    ----------
    name : str
        Check name, used as the ``checks/<name>`` logging namespace.
    passed : bool
        Whether the check passed.
    metrics : dict
        Scalar metrics to log.
    """

    name: str
    passed: bool
    metrics: dict[str, float | int | bool | str] = field(default_factory=dict)


@runtime_checkable
class RuntimeChecker(Protocol):
    """Protocol for runtime checks driven by training callbacks."""

    name: str

    def run(self, state: Any) -> CheckResult:
        """Inspect a `TrainerState` and return a `CheckResult`."""

        ...


class EquivarianceChecker:
    """Check that a model commutes with sampled particle permutations.

    Reads ``state.model`` (a module exposing ``forward_impl``) and
    ``state.batch`` (a permutable input tree), and verifies
    ``F(pi x) ~= pi F(x)`` for ``n_permutations`` sampled non-identity
    permutations. Uses ``forward_impl`` directly so no recursive runtime checks
    are triggered. Owns permutation selection; the callback stays oblivious.

    Parameters
    ----------
    atol, rtol : float, optional
        Tensor comparison tolerances.
    n_permutations : int, optional
        Number of distinct non-identity permutations to sample.
    seed : int or None, optional
        Seed for the checker-local permutation RNG.
    name : str, optional
        Result/namespace name.
    """

    def __init__(
        self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
        n_permutations: int = 2,
        seed: int | None = None,
        name: str = "equivariance",
    ) -> None:
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.n_permutations = int(n_permutations)
        self.seed = seed
        self.name = name

    def run(self, state: Any) -> CheckResult:
        """Run the equivariance check against ``state.model`` and ``state.batch``."""

        module = getattr(state, "model", None)
        inputs = getattr(state, "batch", None)
        n_particles = infer_particle_count(inputs)
        metrics: dict[str, float | int | bool | str] = {
            "n_particles": int(n_particles) if n_particles is not None else 0,
            "n_permutations": 0,
            "max_abs_error": 0.0,
        }
        if module is None or inputs is None or n_particles is None or n_particles < 2:
            metrics["passed"] = True
            return CheckResult(name=self.name, passed=True, metrics=metrics)

        forward_impl = getattr(module, "forward_impl", None) or getattr(module, "forward")
        original_output = forward_impl(inputs)
        permutations = self._sample_permutations(n_particles)

        passed = True
        max_abs_error = 0.0
        for permutation in permutations:
            lhs = forward_impl(permute_tree(inputs, permutation))
            rhs = permute_tree(original_output, permutation)
            max_abs_error = max(max_abs_error, _max_abs_tree_diff(lhs, rhs))
            try:
                assert_tree_allclose(lhs, rhs, atol=self.atol, rtol=self.rtol)
            except AssertionError:
                passed = False

        metrics["n_permutations"] = len(permutations)
        metrics["max_abs_error"] = float(max_abs_error)
        metrics["passed"] = passed
        return CheckResult(name=self.name, passed=passed, metrics=metrics)

    def _sample_permutations(self, n_particles: int) -> list[Permutation]:
        """Sample distinct non-identity permutations with the checker-local RNG."""

        rng = random.Random(self.seed)
        identity = tuple(range(n_particles))
        seen: set[tuple[int, ...]] = set()
        permutations: list[Permutation] = []
        max_attempts = max(1, self.n_permutations) * 20
        attempts = 0
        while len(permutations) < self.n_permutations and attempts < max_attempts:
            attempts += 1
            image = list(range(n_particles))
            rng.shuffle(image)
            candidate = tuple(image)
            if candidate == identity or candidate in seen:
                continue
            seen.add(candidate)
            permutations.append(Permutation(candidate))
        return permutations


def _tree_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Yield tensor leaves from a dataclass/mapping/sequence/tensor tree."""

    from collections.abc import Mapping, Sequence
    from dataclasses import fields, is_dataclass

    if isinstance(obj, torch.Tensor):
        yield obj
        return
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            yield from _tree_tensors(getattr(obj, f.name))
        return
    if isinstance(obj, Mapping):
        for value in obj.values():
            yield from _tree_tensors(value)
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for value in obj:
            yield from _tree_tensors(value)


def _max_abs_tree_diff(a: Any, b: Any) -> float:
    """Return the max absolute difference across paired tensor leaves."""

    diffs = [
        float((left - right).abs().max().item())
        for left, right in zip(_tree_tensors(a), _tree_tensors(b))
        if left.numel() and right.numel()
    ]
    return max(diffs) if diffs else 0.0


__all__ = ["CheckResult", "EquivarianceChecker", "RuntimeChecker"]
