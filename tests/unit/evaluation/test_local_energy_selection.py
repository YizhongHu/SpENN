"""Explicit local-energy evaluator selection and preflight tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tpen.data import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.evaluation.calculators.local_energy import (
    LocalEnergyCalculator,
    evaluate_local_energy_in_chunks,
)
from tpen.evaluation import Evaluator
from tpen.runner.evaluate import Evaluate
from tpen.nn.cusp import ElectronNucleusCusp
from tpen.physics.hamiltonian import (
    AnalyticCuspEvaluator,
    LocalEnergyResult,
    NaiveLocalEnergyEvaluator,
)
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronNucleusPotential


def _terms(*, eps: float = 0.0):
    atoms = AtomicConfiguration(
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    return atoms, {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms, eps=eps)}


class _EligibleModel:
    def __init__(self, atoms):
        self.analytic_cusp_provider = ElectronNucleusCusp(atoms)

    def factorized_local_energy_input(self, batch):  # pragma: no cover - preflight only
        raise AssertionError("preflight must not evaluate a batch")


class _Sampler:
    n_electrons = 2
    spatial_dim = 3


class _Generator:
    sampler = _Sampler()


class _Backend:
    def __init__(self, evaluator_id):
        self.evaluator_id = evaluator_id

    def make_context(self, wavefunction, batch):
        return (wavefunction, batch)

    def validate_for_generator(self, terms, wavefunction, generator):
        del terms, wavefunction, generator

    def evaluate(self, terms, context, *, return_terms=False):
        """Conform to the evaluator protocol; the gate runs before execution."""

        raise AssertionError("mismatch fixture evaluator must not be executed")


class _Trajectory:
    name = "trajectory_mcmc"

    def __init__(self, evaluator_id):
        self.evaluator = _Backend(evaluator_id)


class _LocalEnergy:
    name = "local_energy"

    def __init__(self, evaluator_id):
        self.evaluator = _Backend(evaluator_id)


@dataclass(frozen=True)
class _ThirdContext:
    """A test-only context proving the chunker has no closed evaluator list."""

    wavefunction: object
    batch: ElectronBatch


class _ThirdEvaluator:
    """Structural LocalEnergyEvaluator implementation unknown to TPEN core."""

    evaluator_id = "test_third/v1"

    def make_context(self, wavefunction: object, batch: ElectronBatch) -> _ThirdContext:
        return _ThirdContext(wavefunction=wavefunction, batch=batch)

    def validate_for_generator(self, terms, wavefunction: object, generator: object) -> None:
        del terms, wavefunction, generator

    def evaluate(
        self, terms, context: _ThirdContext, *, return_terms: bool = False
    ) -> torch.Tensor | LocalEnergyResult:
        if not isinstance(context, _ThirdContext):
            raise TypeError("_ThirdEvaluator requires _ThirdContext")
        del terms
        total = torch.full(
            (context.batch.flatten_samples().batch_size,),
            7.0,
            dtype=context.batch.positions.dtype,
        )
        if return_terms:
            return LocalEnergyResult(total=total, terms={"third": total})
        return total


def test_trajectory_rejects_generator_calculator_backend_mismatch() -> None:
    """Selection disagreement fails before a trajectory can start sampling."""

    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            {
                "name": "trajectory",
                "namespace": "eval/trajectory",
                "output_dir": ".",
                "generator": _Trajectory("analytic_cusp/v1"),
                "calculators": [_LocalEnergy("naive/v1")],
            }
        ],
    )

    with pytest.raises(ValueError, match="trajectory local-energy backend mismatch") as error:
        Evaluate(model=object(), evaluator=evaluator)._validate_evaluator_configuration()
    # The IDs in the message prove this was the intended cross-component gate,
    # rather than a later evaluator capability or model validation failure.
    assert "generator='analytic_cusp/v1'" in str(error.value)
    assert "calculator='naive/v1'" in str(error.value)


def test_calculator_defaults_to_naive_and_opt_in_is_explicit() -> None:
    atoms, terms = _terms()
    default = LocalEnergyCalculator(hamiltonian_terms=terms)
    analytic = LocalEnergyCalculator(
        hamiltonian_terms=terms,
        evaluator=AnalyticCuspEvaluator(),
    )

    assert isinstance(default.evaluator, NaiveLocalEnergyEvaluator)
    assert isinstance(analytic.evaluator, AnalyticCuspEvaluator)
    # A production edit that silently inferred analytic mode from the model or
    # terms would make this explicit-construction pin fail.
    assert default.evaluator is not analytic.evaluator
    assert atoms.n_nuclei == terms["electron_nucleus"].atoms.n_nuclei


def test_chunking_delegates_context_creation_to_protocol_only_evaluator() -> None:
    batch = torch.zeros((5, 2, 3), dtype=torch.float64)
    evaluator = _ThirdEvaluator()

    result = evaluate_local_energy_in_chunks(
        {},
        wavefunction=object(),
        batch=ElectronBatch(positions=batch),
        chunk_size=2,
        evaluator=evaluator,
    )

    # A production edit that restores concrete evaluator dispatch or bypasses
    # make_context makes this structural third backend fail before this value.
    torch.testing.assert_close(result, torch.full((5,), 7.0, dtype=torch.float64))


def test_analytic_calculator_preflight_rejects_missing_capability() -> None:
    _, terms = _terms()
    calculator = LocalEnergyCalculator(hamiltonian_terms=terms, evaluator=AnalyticCuspEvaluator())

    with pytest.raises(ValueError, match="explicitly bound ElectronNucleusCusp provider"):
        calculator.validate(model=object(), generator=_Generator())


@pytest.mark.parametrize(
    ("terms_kwargs", "message"),
    [
        ({"eps": 1.0e-12}, "eps == 0"),
    ],
)
def test_analytic_calculator_preflight_rejects_domain_configuration(terms_kwargs, message) -> None:
    atoms, terms = _terms(**terms_kwargs)
    calculator = LocalEnergyCalculator(hamiltonian_terms=terms, evaluator=AnalyticCuspEvaluator())

    with pytest.raises(ValueError, match=message):
        calculator.validate(model=_EligibleModel(atoms), generator=_Generator())


def test_analytic_calculator_preflight_rejects_provider_geometry() -> None:
    atoms, terms = _terms()
    provider_atoms = AtomicConfiguration(
        torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    calculator = LocalEnergyCalculator(hamiltonian_terms=terms, evaluator=AnalyticCuspEvaluator())

    with pytest.raises(ValueError, match="provider must share matching atoms"):
        calculator.validate(model=_EligibleModel(provider_atoms), generator=_Generator())


def test_analytic_calculator_preflight_rejects_duplicate_kinetic_participant() -> None:
    atoms, terms = _terms()
    terms["kinetic_extra"] = KineticEnergy()
    calculator = LocalEnergyCalculator(hamiltonian_terms=terms, evaluator=AnalyticCuspEvaluator())

    with pytest.raises(ValueError, match="exactly one KineticEnergy"):
        calculator.validate(model=_EligibleModel(atoms), generator=_Generator())


def test_analytic_calculator_preflight_rejects_sampler_dimension() -> None:
    atoms, terms = _terms()
    calculator = LocalEnergyCalculator(hamiltonian_terms=terms, evaluator=AnalyticCuspEvaluator())

    class WrongDimensionGenerator:
        class sampler:
            n_electrons = 2
            spatial_dim = 2

    with pytest.raises(ValueError, match="spatial dimension 3"):
        calculator.validate(model=_EligibleModel(atoms), generator=WrongDimensionGenerator())


def test_analytic_calculator_preflight_rejects_sampler_geometry() -> None:
    atoms, terms = _terms()
    other_atoms = AtomicConfiguration(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    calculator = LocalEnergyCalculator(hamiltonian_terms=terms, evaluator=AnalyticCuspEvaluator())

    class WrongGeometrySampler:
        n_electrons = 2
        spatial_dim = 3
        atomic_configuration = other_atoms

    class WrongGeometryGenerator:
        sampler = WrongGeometrySampler()

    with pytest.raises(ValueError, match="sampler must share matching atoms"):
        calculator.validate(model=_EligibleModel(atoms), generator=WrongGeometryGenerator())
