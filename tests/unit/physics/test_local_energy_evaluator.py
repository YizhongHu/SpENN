"""Typed local-energy evaluator seam (LOCAL-ENERGY-ARCHITECTURE.md active slice).

Pins the acceptance criteria of the typed-interface slice: `local_energy` is
an explicit delegate to `NaiveLocalEnergyEvaluator`; the evaluator statically
and dynamically consumes a `NaiveLocalEnergyContext` (never an arbitrary
mapping); term outputs, names, and validation behavior are unchanged; and a
protocol-only custom term unknown to TPEN core evaluates successfully.
"""

from __future__ import annotations

import pytest
import torch
from typeguard import TypeCheckError, suppress_type_checks

from tpen.data import AtomicConfiguration
from tpen.data.batch import ElectronBatch, FactorizedLocalEnergyInput, WavefunctionOutput
from tpen.nn.cusp import ElectronNucleusCusp
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronNucleusPotential, HarmonicTrap
from tpen.physics.hamiltonian import (
    AnalyticCuspContext,
    AnalyticCuspEvaluator,
    LocalEnergyResult,
    NaiveLocalEnergyContext,
    NaiveLocalEnergyEvaluator,
    local_energy,
)

_DTYPE = torch.float64


class _AnalyticWavefunction:
    def __init__(
        self, provider, curvature: float = 0.0, linear=None, sign: float = 1.0, node: bool = False
    ):
        self.analytic_cusp_provider = provider
        self.curvature = curvature
        self.linear = linear
        self.sign = sign
        self.node = node
        self.factorized_calls = 0

    def factorized_local_energy_input(self, batch):
        self.factorized_calls += 1
        flat = batch.flatten_samples()
        logabs = -self.curvature * flat.positions.square().sum(dim=(1, 2))
        if self.linear is not None:
            linear = torch.as_tensor(self.linear, dtype=flat.dtype, device=flat.device)
            logabs = logabs + (flat.positions * linear).sum(dim=(1, 2))
        if self.node:
            logabs = torch.full_like(logabs, -torch.inf)
        output = WavefunctionOutput(
            logabs=logabs,
            sign=torch.full_like(logabs, self.sign),
            aux={"source": "regular"},
        )
        return FactorizedLocalEnergyInput(output, self.analytic_cusp_provider.analytic_evaluation(flat))


class _BombKinetic(KineticEnergy):
    def local_energy(self, wavefunction, batch):
        raise AssertionError("kinetic participant was called")


class _BombPotential(ElectronNucleusPotential):
    def local_energy(self, wavefunction, batch):
        raise AssertionError("potential participant was called")


class _CountingTerm:
    name = "counting"

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def local_energy(self, wavefunction, batch):
        self.calls += 1
        return LocalEnergyResult(total=torch.full((batch.flatten_samples().batch_size,), self.value, dtype=batch.positions.dtype))


def _analytic_setup(*, z=1.0, nuclear_position=None, eps=0.0):
    position = torch.zeros((1, 3), dtype=_DTYPE) if nuclear_position is None else torch.tensor([nuclear_position], dtype=_DTYPE)
    atoms = AtomicConfiguration(position, torch.tensor([z], dtype=_DTYPE))
    provider = ElectronNucleusCusp(atoms)
    wavefunction = _AnalyticWavefunction(provider)
    batch = ElectronBatch(positions=torch.tensor([[[0.4, 0.0, 0.0]]], dtype=_DTYPE))
    terms = {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms, eps=eps)}
    return atoms, wavefunction, batch, terms


def _batch(n_walkers: int = 3) -> ElectronBatch:
    generator = torch.Generator().manual_seed(11)
    return ElectronBatch(positions=torch.randn(n_walkers, 2, 3, generator=generator, dtype=_DTYPE))


class ConstantTerm:
    """Protocol-only term unknown to TPEN core."""

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


def test_analytic_evaluator_fuses_hydrogen_and_returns_full_output() -> None:
    _, wavefunction, batch, terms = _analytic_setup()
    result = AnalyticCuspEvaluator().evaluate(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )

    # A production edit that differentiates the cusp-free output incorrectly,
    # or adds Coulomb separately, changes this exact hydrogenic value.
    torch.testing.assert_close(result.total, torch.full((1,), -0.5, dtype=_DTYPE))
    assert list(result.terms) == ["kinetic_plus_electron_nucleus"]
    assert result.wavefunction_output is not None
    torch.testing.assert_close(result.wavefunction_output.logabs, torch.tensor([-0.4], dtype=_DTYPE))
    assert result.wavefunction_output.aux == {"source": "regular"}
    assert result.per_electron_kinetic is None
    assert result.term_provenance == {
        "kinetic_plus_electron_nucleus": ("kinetic", "electron_nucleus")
    }


def test_analytic_fast_kernel_matches_independent_reference_and_parameter_gradient() -> None:
    _, wavefunction, batch, terms = _analytic_setup()
    # Deliberately use unequal batch/electron axes: equal axes can hide a
    # missing electron reduction through accidental broadcasting.
    batch = ElectronBatch(
        positions=torch.tensor(
            [
                [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0]],
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.6, 0.0, 0.0], [0.0, 0.6, 0.0]],
            ],
            dtype=_DTYPE,
        )
    )
    curvature = torch.nn.Parameter(torch.tensor(0.23, dtype=_DTYPE))
    wavefunction.curvature = curvature
    fast = AnalyticCuspEvaluator().evaluate(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )
    fast_gradient = torch.autograd.grad(fast.total.sum(), curvature, retain_graph=True)[0]
    reference = AnalyticCuspEvaluator().evaluate_reference(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )
    reference_gradient = torch.autograd.grad(reference.total.sum(), curvature)[0]
    torch.testing.assert_close(fast.total, reference.total, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(fast_gradient, reference_gradient, rtol=1.0e-12, atol=1.0e-12)
    assert tuple(fast.total.shape) == (batch.flatten_samples().batch_size,)
    assert tuple(reference.total.shape) == (batch.flatten_samples().batch_size,)


def test_analytic_evaluator_hydrogenic_near_cusp_ladder_has_derived_limit() -> None:
    z = 2.0
    _, wavefunction, _, terms = _analytic_setup(z=z)
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=_DTYPE,
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    radii = torch.tensor([1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10], dtype=_DTYPE)
    energies = []
    for radius in radii:
        batch = ElectronBatch(positions=(radius * directions).reshape(-1, 1, 3))
        result = AnalyticCuspEvaluator().evaluate(
            terms, AnalyticCuspContext(wavefunction, batch)
        )
        # Changing the cusp cancellation, kinetic contraction, or Coulomb
        # fusion changes these finite values or their derived -Z**2 / 2 limit.
        assert torch.isfinite(result).all()
        expected_limit = -(z**2) / 2.0
        torch.testing.assert_close(
            result,
            torch.full_like(result, expected_limit),
            atol=2.0 * radius.item() + 1.0e-12,
            rtol=1.0e-12,
        )
        energies.append(result.detach())

    # A production edit that reintroduces a radius-dependent singular term or
    # drops the analytic cusp contribution prevents convergence at the last
    # ladder point, even if a single radius happens to look finite.
    torch.testing.assert_close(energies[-1], energies[-2], atol=3.0e-8, rtol=1.0e-12)


def test_analytic_evaluator_near_cusp_with_regular_gradient_stays_finite() -> None:
    z = 1.5
    regular_gradient = torch.tensor([0.7, -0.2, 0.4], dtype=_DTYPE)
    _, wavefunction, _, terms = _analytic_setup(z=z)
    wavefunction.linear = regular_gradient
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, 0.5]],
        dtype=_DTYPE,
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    radii = torch.tensor([1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9], dtype=_DTYPE)
    energies = []
    for radius in radii:
        batch = ElectronBatch(positions=(radius * directions).reshape(-1, 1, 3))
        result = AnalyticCuspEvaluator().evaluate(
            terms, AnalyticCuspContext(wavefunction, batch)
        )
        # The nonzero regular gradient makes direction-independence an
        # invalid assertion; boundedness is the contract for this fixture.
        assert torch.isfinite(result).all()
        energies.append(result.detach())

    spread = (energies[-1].max() - energies[-1].min()).item()
    print(f"near-cusp directional spread (z={z}, eps=0): {spread:.17g}")


@pytest.mark.parametrize("kernel", ["evaluate", "evaluate_reference"])
def test_analytic_evaluator_rejects_exact_electron_nucleus_coalescence(kernel) -> None:
    _, wavefunction, _, terms = _analytic_setup(z=1.0)
    batch = ElectronBatch(positions=torch.zeros((1, 1, 3), dtype=_DTYPE))

    # Removing the explicit distance gate exposes displacement / distance =
    # 0 / 0 instead of rejecting the undefined unit direction.
    with pytest.raises(ValueError, match="does not support electron-nucleus coalescence"):
        getattr(AnalyticCuspEvaluator(), kernel)(terms, AnalyticCuspContext(wavefunction, batch))


@pytest.mark.parametrize("kernel", ["evaluate", "evaluate_reference"])
def test_analytic_evaluator_rejects_zero_sign_center_node(kernel) -> None:
    _, wavefunction, _, terms = _analytic_setup(z=1.0)
    wavefunction.sign = 0.0
    wavefunction.node = True
    batch = ElectronBatch(positions=torch.tensor([[[0.2, 0.0, 0.0]]], dtype=_DTYPE))

    # Exact zeros require the typed signed-log representation sign == 0 and
    # logabs == -inf. Removing the off-node gate would allow that legitimate
    # node into the real-wavefunction local-energy path instead of failing for
    # its true cause.
    with pytest.raises(ValueError, match="requires an off-node real wavefunction"):
        getattr(AnalyticCuspEvaluator(), kernel)(terms, AnalyticCuspContext(wavefunction, batch))


def test_analytic_evaluator_skips_participants_and_runs_custom_terms_once() -> None:
    before = _CountingTerm(2.25)
    after = _CountingTerm(-0.75)
    atoms, wavefunction, batch, _ = _analytic_setup()
    terms = {
        "before": before,
        "kinetic": _BombKinetic(),
        "electron_nucleus": _BombPotential(atoms, eps=0.0),
        "after": after,
    }
    result = AnalyticCuspEvaluator().evaluate(terms, AnalyticCuspContext(wavefunction, batch), return_terms=True)
    assert before.calls == 1
    assert after.calls == 1
    assert list(result.terms) == ["before", "kinetic_plus_electron_nucleus", "after"]
    expected_total = sum(result.terms.values())
    torch.testing.assert_close(result.total, expected_total)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda terms: terms.update({"extra": KineticEnergy()}), "exactly one KineticEnergy"),
        (lambda terms: terms.update({"electron_nucleus": ElectronNucleusPotential(_analytic_setup()[0])}), "eps == 0"),
    ],
)
def test_analytic_evaluator_rejects_invalid_participant_domain(mutator, message) -> None:
    atoms, wavefunction, batch, terms = _analytic_setup()
    mutator(terms)
    with pytest.raises(ValueError, match=message):
        AnalyticCuspEvaluator().evaluate(terms, AnalyticCuspContext(wavefunction, batch))
