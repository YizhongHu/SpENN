"""Typed local-energy evaluator seam (LOCAL-ENERGY-ARCHITECTURE.md active slice).

Pins the acceptance criteria of the typed-interface slice: `local_energy` is
an explicit delegate to `NaiveLocalEnergyEvaluator`; the evaluator statically
and dynamically consumes a `NaiveLocalEnergyContext` (never an arbitrary
mapping); term outputs, names, and validation behavior are unchanged; and a
protocol-only custom term unknown to SpENN core evaluates successfully.
"""

from __future__ import annotations

import pytest
import torch
from typeguard import TypeCheckError, suppress_type_checks

from spenn.data.batch import ElectronBatch
from spenn.physics.potential import HarmonicTrap
from spenn.physics.hamiltonian import (
    LocalEnergyResult,
    NaiveLocalEnergyContext,
    NaiveLocalEnergyEvaluator,
    local_energy,
)

_DTYPE = torch.float64


def _batch(n_walkers: int = 3) -> ElectronBatch:
    generator = torch.Generator().manual_seed(11)
    return ElectronBatch(positions=torch.randn(n_walkers, 2, 3, generator=generator, dtype=_DTYPE))


class ConstantTerm:
    """Protocol-only term unknown to SpENN core."""

    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        flat = batch.flatten_samples()
        total = torch.full((flat.batch_size,), self.value, dtype=flat.dtype, device=flat.device)
        return LocalEnergyResult(total=total)


class BrokenShapeTerm:
    name = "broken"

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        return LocalEnergyResult(total=torch.zeros((2, 2), dtype=_DTYPE))


def test_local_energy_delegates_to_naive_evaluator() -> None:
    # A position-dependent physics term (HarmonicTrap) makes the equality a
    # real pin rather than a constant-term tautology.
    batch = _batch()
    terms = {"trap": HarmonicTrap(omega=0.5), "shift": ConstantTerm(-0.25)}

    via_entry_point = local_energy(terms, None, batch, return_terms=True)
    via_evaluator = NaiveLocalEnergyEvaluator().evaluate(
        terms, NaiveLocalEnergyContext(wavefunction=None, batch=batch), return_terms=True
    )

    torch.testing.assert_close(via_entry_point.total, via_evaluator.total)
    torch.testing.assert_close(via_entry_point.terms["trap"], via_evaluator.terms["trap"])
    assert list(via_entry_point.terms) == list(via_evaluator.terms) == ["trap", "shift"]
    expected_trap = 0.5 * 0.5**2 * batch.positions.square().sum(dim=(1, 2))
    torch.testing.assert_close(via_entry_point.terms["trap"], expected_trap)


def test_naive_evaluator_rejects_untyped_context() -> None:
    # Acceptance pin: the evaluator does not accept arbitrary mappings or
    # positional (wavefunction, batch) shims — only the typed context. Under
    # the instrumented suite typeguard raises first (TypeCheckError); the
    # evaluator's own TypeError guard covers uninstrumented production runs.
    with pytest.raises((TypeError, TypeCheckError), match="NaiveLocalEnergyContext"):
        NaiveLocalEnergyEvaluator().evaluate(
            {"a": ConstantTerm(1.0)},
            {"wavefunction": None, "batch": _batch()},
        )
    # Pin the evaluator's own guard too — production runs are uninstrumented,
    # so the TypeError branch must stay live without typeguard.
    with suppress_type_checks():
        with pytest.raises(TypeError, match="NaiveLocalEnergyContext"):
            NaiveLocalEnergyEvaluator().evaluate(
                {"a": ConstantTerm(1.0)},
                {"wavefunction": None, "batch": _batch()},
            )


def test_empty_hamiltonian_returns_zero_energy() -> None:
    batch = _batch()
    total = NaiveLocalEnergyEvaluator().evaluate(
        {}, NaiveLocalEnergyContext(wavefunction=None, batch=batch)
    )
    torch.testing.assert_close(total, torch.zeros(3, dtype=_DTYPE))


def test_validation_failures_surface_through_evaluator() -> None:
    context = NaiveLocalEnergyContext(wavefunction=None, batch=_batch())
    with pytest.raises(ValueError, match="must have shape"):
        NaiveLocalEnergyEvaluator().evaluate({"broken": BrokenShapeTerm()}, context)


def test_sequence_terms_keep_snake_case_naming_through_delegation() -> None:
    batch = _batch()
    result = local_energy([ConstantTerm(2.0)], None, batch, return_terms=True)
    assert list(result.terms) == ["constant_term"]
    torch.testing.assert_close(result.terms["constant_term"], torch.full((3,), 2.0, dtype=_DTYPE))
