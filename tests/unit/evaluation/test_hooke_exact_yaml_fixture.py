"""Tests verifying the exact Hooke evaluation stack against analytic results.

These tests run the real evaluation pipeline (generators → calculators → summaries)
on the exact Hooke wavefunction and check that the diagnostics recover known values.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from spenn.evaluation import Evaluator, EvaluationTask
from spenn.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from spenn.evaluation.calculators import (
    LocalEnergyCalculator,
    RadialLogAbsDerivativeCalculator,
    WavefunctionCalculator,
)
from spenn.evaluation.generators import CuspGridGenerator
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import SummaryResult
from spenn.evaluation.summaries import (
    CoalescenceDivergenceSummary,
    LocalEnergySummary,
    OppositeSpinCuspSummary,
)
from spenn.physics.hooke import HookeSingletExact
from spenn.physics.kinetic import KineticEnergy
from spenn.physics.potential import ElectronElectronInteraction, HarmonicTrap

FIXTURES = Path(__file__).resolve().parents[3] / "integration" / "artifacts" / "hooke"
SINGLET_FIXTURE = FIXTURES / "exact_singlet_eval.yaml"


def _context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        namespace="eval",
        artifact_level="metrics_only",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        output_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _hooke_hamiltonian_terms():
    return {
        "kinetic": KineticEnergy(),
        "harmonic_trap": HarmonicTrap(omega=0.5),
        "electron_electron": ElectronElectronInteraction(),
    }


def test_hooke_exact_yaml_fixture_has_no_phase_key() -> None:
    cfg = OmegaConf.load(SINGLET_FIXTURE)
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert "phase" not in raw.get("evaluation", {})
    assert "phase" not in raw.get("evaluator", {})


def test_hooke_exact_yaml_fixture_has_no_required_key() -> None:
    cfg = OmegaConf.load(SINGLET_FIXTURE)
    raw = OmegaConf.to_container(cfg, resolve=False)
    for task in raw.get("evaluation_tasks", {}).values():
        assert "required" not in task, f"found 'required' key in task: {task}"


def test_hooke_exact_local_energy_is_constant_under_evaluation_stack(tmp_path: Path) -> None:
    """LocalEnergyCalculator on exact Hooke singlet should return E_L ≈ 2.0 everywhere."""

    generated = CuspGridGenerator(
        n_points=5,
        r12_min=0.05,
        r12_max=2.0,
        n_directions=3,
        center_of_mass_radii=[0.0, 0.5],
        paired_directions=True,
        seed=0,
    ).generate(model=None, context=_context(tmp_path))

    model = HookeSingletExact()
    bundle = EvaluationBundle(generated=generated)
    bundle = WavefunctionCalculator().calculate(model=model, bundle=bundle, context=_context(tmp_path))
    bundle = LocalEnergyCalculator(hamiltonian_terms=_hooke_hamiltonian_terms()).calculate(
        model=model, bundle=bundle, context=_context(tmp_path)
    )

    assert bundle.local_energy is not None
    finite = bundle.local_energy.local_energy[bundle.local_energy.finite_mask]
    assert finite.numel() > 0, "all local energies were non-finite"
    assert float(finite.mean().item()) == pytest.approx(2.0, abs=1.0e-4)
    assert float(finite.var().item()) == pytest.approx(0.0, abs=1.0e-8)


def test_hooke_exact_cusp_even_slope_matches_half(tmp_path: Path) -> None:
    """OppositeSpinCuspSummary on exact Hooke should give cusp_even_slope ≈ 0.5."""

    generated = CuspGridGenerator(
        n_points=8,
        r12_min=0.01,
        r12_max=0.5,
        n_directions=4,
        center_of_mass_radii=[0.0, 0.3],
        paired_directions=True,
        seed=0,
    ).generate(model=None, context=_context(tmp_path))

    model = HookeSingletExact()
    bundle = EvaluationBundle(generated=generated)
    bundle = WavefunctionCalculator().calculate(model=model, bundle=bundle, context=_context(tmp_path))
    bundle = RadialLogAbsDerivativeCalculator(coordinate="r12").calculate(
        model=model, bundle=bundle, context=_context(tmp_path)
    )

    metrics = OppositeSpinCuspSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/cusp",
    ).metrics

    assert metrics["cusp_even_slope_mean"] == pytest.approx(0.5, abs=1.0e-4)
    assert metrics["cusp_even_slope_abs_error"] == pytest.approx(0.0, abs=1.0e-4)


def test_hooke_exact_coalescence_c_minus_one_is_zero(tmp_path: Path) -> None:
    """CoalescenceDivergenceSummary on exact Hooke should give C_{-1} ≈ 0."""

    generated = CuspGridGenerator(
        n_points=10,
        r12_min=0.01,
        r12_max=0.3,
        n_directions=2,
        center_of_mass_radii=[0.0],
        paired_directions=False,
        seed=0,
    ).generate(model=None, context=_context(tmp_path))

    model = HookeSingletExact()
    bundle = EvaluationBundle(generated=generated)
    bundle = LocalEnergyCalculator(hamiltonian_terms=_hooke_hamiltonian_terms()).calculate(
        model=model, bundle=bundle, context=_context(tmp_path)
    )

    metrics = CoalescenceDivergenceSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/cusp",
    ).metrics

    assert metrics["c_minus_1_abs_max"] == pytest.approx(0.0, abs=1.0e-3)
    assert metrics["coalescence_fit_failure_count"] == 0


def test_hooke_exact_task_outputs_use_task_directories(tmp_path: Path) -> None:
    """Evaluator resolves task_output_dir to run_dir / task.name for each task."""

    recorded: list[Path] = []

    class _DirRecordSummary:
        name = "dir_record"
        required_fields: frozenset[str] = frozenset()

        def summarize(self, *, bundle, context: EvaluationContext, namespace: str) -> SummaryResult:
            recorded.append(context.task_output_dir)
            return SummaryResult(metrics={})

    class _NullGenerator:
        name = "null"

        def generate(self, *, model, context: EvaluationContext) -> GeneratedConfigurations:
            from spenn.data.batch import ElectronBatch

            batch = ElectronBatch(
                positions=torch.zeros(1, 2, 3, dtype=torch.float64),
                spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
            )
            return GeneratedConfigurations(batch=batch, metadata={})

    ctx = SimpleNamespace()
    ctx.run_dir = tmp_path
    ctx.metadata = SimpleNamespace(device=None, dtype=None)
    ctx.log = lambda *a, **kw: None

    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="cusp",
                namespace="eval/cusp",
                generator=_NullGenerator(),
                calculators=[],
                summaries=[_DirRecordSummary()],
            ),
            EvaluationTask(
                name="tail",
                namespace="eval/tail",
                generator=_NullGenerator(),
                calculators=[],
                summaries=[_DirRecordSummary()],
            ),
        ],
    )
    evaluator.evaluate(model=None, context=ctx, emit=lambda *a, **kw: None)

    assert recorded[0] == tmp_path / "cusp"
    assert recorded[1] == tmp_path / "tail"
