"""Analytic and numerical-boundary contracts for the helium P3 atlases."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tpen.data import AtomicConfiguration
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.geometry import two_electron_atomic_geometry
from tpen.data.indices import permute_particle_axis
from tpen.data.permutation import Permutation
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations, HeliumAtlasValues
from tpen.evaluation.calculators import HeliumAtlasCalculator
from tpen.evaluation.calculators.helium_atlas import (
    _validate_boundary_order,
    directional_log_derivatives,
    directional_log_derivatives_reference,
)
from tpen.evaluation.generators import (
    HeliumAngularShellGenerator,
    HeliumCenterOfMassEscapeGenerator,
    HeliumElectronElectronApproachGenerator,
    HeliumElectronNucleusApproachGenerator,
    HeliumOneElectronEscapeGenerator,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries import (
    HeliumAtlasWriter,
    HeliumCurvatureSummary,
    HeliumNumericalLimitSummary,
    HeliumTailSummary,
)
from tpen.nn import ElectronElectronCusp
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.physics.kinetic import (
    KineticEnergy,
    per_electron_kinetic_from_logabs,
    per_electron_kinetic_from_logabs_reference,
)


class _QuadraticFactor(torch.nn.Module):
    """Analytic additive factor ``coefficient * sum_i |r_i|^2``."""

    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = float(coefficient)

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        positions = batch.flatten_samples().positions
        return self.coefficient * positions.square().sum(dim=(1, 2))


class _AnalyticAtlasModel(torch.nn.Module):
    """Restored-model-shaped analytic wavefunction with two factors."""

    def __init__(self) -> None:
        super().__init__()
        self.factors = torch.nn.ModuleList(
            [_QuadraticFactor(-0.25), _QuadraticFactor(-0.5)]
        )

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = sum((factor(batch) for factor in self.factors), start=torch.zeros(
            batch.flatten_samples().batch_size,
            device=batch.device,
            dtype=batch.dtype,
        ))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _CountingAnalyticAtlasModel(_AnalyticAtlasModel):
    """Analytic atlas model that exposes its exact forward count."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_count = 0

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        self.forward_count += 1
        return super().forward(batch)


class _ConstantTerm:
    """One registry term with an analytic constant local value."""

    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        del wavefunction
        flat = batch.flatten_samples()
        value = torch.full(
            (flat.batch_size,), self.value, device=flat.device, dtype=flat.dtype
        )
        return LocalEnergyResult(total=value, terms={self.name: value})


class _ExecutedCuspModel(torch.nn.Module):
    """Minimal model exposing the real executed analytic e-e factor."""

    def __init__(self) -> None:
        super().__init__()
        self.factors = torch.nn.ModuleList([ElectronElectronCusp()])

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = self.factors[0](batch)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _atoms() -> AtomicConfiguration:
    return AtomicConfiguration(
        positions=torch.zeros((1, 3), dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )


def _context(tmp_path: Path, *, dtype: torch.dtype = torch.float64) -> EvaluationContext:
    return EvaluationContext(
        namespace="eval/helium_atlas",
        artifact_level="records",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=dtype,
        seed=17,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _calculator(*, electron_electron_name: str = "constant") -> HeliumAtlasCalculator:
    return HeliumAtlasCalculator(
        hamiltonian_terms={
            "kinetic": KineticEnergy(),
            electron_electron_name: _ConstantTerm(1.25),
        },
        factor_indices={
            "executed_smoothed_ee_factor": 0,
            "executed_electron_nucleus_factor": 1,
        },
        chunk_size=3,
    )


def _one_electron_generated(tmp_path: Path) -> GeneratedConfigurations:
    return HeliumOneElectronEscapeGenerator(
        atoms=_atoms(),
        directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        radii=[1.0, 2.0, 3.0, 4.0],
        spectator_position=[0.0, 0.0, -1.0],
        probe_electron=0,
    ).generate(model=None, context=_context(tmp_path))


def _calculated_one_electron(tmp_path: Path) -> EvaluationBundle:
    generated = _one_electron_generated(tmp_path)
    return _calculator().calculate(
        model=_AnalyticAtlasModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
    )


def test_geometric_refinement_records_properties_and_provenance_not_magic_epsilon(
    tmp_path: Path,
) -> None:
    directions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    max_refinement_steps = 128
    generated = HeliumElectronElectronApproachGenerator(
        atoms=_atoms(),
        directions=directions,
        center_of_mass=[0.0, 0.0, 0.0],
        start_radius=1.0e-2,
        refinement_ratio=1.0e-8,
        max_refinement_steps=max_refinement_steps,
    ).generate(model=None, context=_context(tmp_path))
    bundle = _calculator().calculate(
        model=_AnalyticAtlasModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
    )
    atlas = bundle.helium_atlas
    assert isinstance(atlas, HeliumAtlasValues)
    geometry = two_electron_atomic_geometry(generated.batch)
    assert torch.equal(
        atlas.realized_coordinate,
        geometry.electron_electron_distance,
    )
    assert atlas.provenance.dtype == "float64"
    assert atlas.provenance.device == "cpu"
    assert atlas.provenance.evaluation_dtype == "float64"
    assert atlas.provenance.evaluation_device == "cpu"
    assert atlas.provenance.seed == 17
    positive_domain = atlas.ideal_unfloored_ee_positive_separation_domain_mask
    evaluation_defined = (
        atlas.ideal_unfloored_ee_reciprocal_evaluation_defined_mask
    )
    assert positive_domain is not None
    assert evaluation_defined is not None
    ideal = atlas.ideal_unfloored_ee_inverse_distance
    assert ideal is not None
    assert torch.equal(evaluation_defined, torch.isfinite(ideal))
    assert torch.equal(positive_domain, atlas.realized_coordinate > 0)

    ray_ids = generated.metadata["ray_id"]
    for ray_id in sorted(set(ray_ids.tolist())):
        selection = ray_ids == ray_id
        sentinel = atlas.is_exact_zero_sentinel[selection]
        realized = atlas.realized_coordinate[selection]
        realized_approach = realized[~sentinel]
        assert 2 <= realized_approach.numel() <= max_refinement_steps
        assert torch.all(realized_approach[1:] < realized_approach[:-1])
        assert torch.all(realized_approach >= 0)
        coordinate_boundary = atlas.is_coordinate_representability_boundary[selection]
        reciprocal_boundary_mask = (
            atlas.is_ideal_unfloored_ee_reciprocal_failure_boundary
        )
        assert reciprocal_boundary_mask is not None
        reciprocal_boundary = reciprocal_boundary_mask[selection]
        assert int(coordinate_boundary.sum().item()) == 1
        assert int(reciprocal_boundary.sum().item()) == 1
        assert int(sentinel.sum().item()) == 1
        assert bool(sentinel[-1].item())
        assert not torch.any(coordinate_boundary & sentinel)
        assert not torch.any(reciprocal_boundary & sentinel)
        assert atlas.domain_status[torch.nonzero(selection, as_tuple=False)[-1].item()] == (
            "exact_zero_sentinel"
        )
        ray_defined = evaluation_defined[selection]
        transitions = ray_defined[:-1] & ~ray_defined[1:]
        assert int(transitions.sum().item()) == 1
        transition = int(torch.nonzero(transitions, as_tuple=False).item())
        assert bool(reciprocal_boundary[transition + 1].item())
        assert torch.all(ray_defined[: transition + 1])
        assert not torch.any(ray_defined[transition + 1 :])
        assert bool(positive_domain[selection][transition + 1].item()) == bool(
            (realized[transition + 1] > 0).item()
        )
        assert not bool(sentinel[transition + 1].item())
        reciprocal_radius = atlas.ideal_unfloored_ee_reciprocal_failure_radius
        assert reciprocal_radius is not None
        evaluation_failure_radius = reciprocal_radius[selection][reciprocal_boundary].item()
        coordinate_underflow_radius = atlas.coordinate_representability_boundary_radius[
            selection
        ][coordinate_boundary].item()
        assert evaluation_failure_radius == realized[reciprocal_boundary].item()
        assert coordinate_underflow_radius == realized[coordinate_boundary].item()
        assert evaluation_failure_radius >= 0
        assert coordinate_underflow_radius >= 0
        assert evaluation_failure_radius >= coordinate_underflow_radius
        assert torch.nonzero(reciprocal_boundary, as_tuple=False).item() <= torch.nonzero(
            coordinate_boundary, as_tuple=False
        ).item()
        assert not bool(positive_domain[selection][-1].item())
        assert not bool(evaluation_defined[selection][-1].item())
    assert generated.batch.batch_size <= len(directions) * (max_refinement_steps + 1)


def test_reciprocal_failure_must_precede_coordinate_underflow() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        _validate_boundary_order(
            realized_coordinate=torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64),
            reciprocal_evaluation_defined_mask=torch.tensor([True, False, False]),
            reciprocal_failure_radius=torch.tensor(
                [float("nan"), 0.5, float("nan")], dtype=torch.float64
            ),
            reciprocal_boundary=torch.tensor([False, True, False]),
            coordinate_boundary_radius=torch.tensor(
                [float("nan"), 1.0, float("nan")], dtype=torch.float64
            ),
            coordinate_boundary=torch.tensor([False, True, False]),
            sentinel=torch.tensor([False, False, True]),
            ray=torch.tensor([0, 0, 0], dtype=torch.long),
        )


def test_refinement_fails_closed_when_bound_prevents_reaching_numerical_limit(
    tmp_path: Path,
) -> None:
    generator = HeliumElectronNucleusApproachGenerator(
        atoms=_atoms(),
        directions=[[1.0, 0.0, 0.0]],
        spectator_position=[0.0, 0.0, -1.0],
        start_radius=1.0e-2,
        refinement_ratio=0.5,
        max_refinement_steps=2,
    )
    with pytest.raises(ValueError, match="before reaching a numerical boundary"):
        generator.generate(model=None, context=_context(tmp_path))


def test_electron_nucleus_calculator_rejects_missing_coordinate_boundary(
    tmp_path: Path,
) -> None:
    generated = HeliumElectronNucleusApproachGenerator(
        atoms=_atoms(),
        directions=[[1.0, 0.0, 0.0]],
        spectator_position=[0.0, 0.0, -1.0],
        start_radius=1.0e-2,
        refinement_ratio=1.0e-64,
        probe_electrons=[0],
        max_refinement_steps=16,
    ).generate(model=None, context=_context(tmp_path))
    metadata = dict(generated.metadata)
    metadata["is_coordinate_representability_boundary"] = torch.zeros_like(
        metadata["is_coordinate_representability_boundary"]
    )
    malformed = GeneratedConfigurations(batch=generated.batch, metadata=metadata)

    with pytest.raises(ValueError, match="one coordinate-representability boundary"):
        _calculator().calculate(
            model=_AnalyticAtlasModel(),
            bundle=EvaluationBundle(generated=malformed),
            context=_context(tmp_path),
        )


def test_escape_and_shell_generators_preserve_explicit_spectator_com_and_directions(
    tmp_path: Path,
) -> None:
    directions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    spectator = HeliumOneElectronEscapeGenerator(
        atoms=_atoms(),
        directions=directions,
        radii=[2.0, 4.0],
        spectator_position=[0.0, 0.0, -1.0],
    ).generate(model=None, context=_context(tmp_path))
    center = HeliumCenterOfMassEscapeGenerator(
        atoms=_atoms(),
        directions=directions,
        radii=[2.0, 4.0],
        relative_positions=[[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]],
    ).generate(model=None, context=_context(tmp_path))
    shell = HeliumAngularShellGenerator(
        atoms=_atoms(),
        directions=directions,
        radii=[2.0, 4.0],
        spectator_position=[0.0, 0.0, -1.0],
    ).generate(model=None, context=_context(tmp_path))

    assert set(spectator.metadata["atlas_geometry_kind"]) == {"configured_spectator"}
    assert torch.all(spectator.batch.positions[:, 1] == torch.tensor([0.0, 0.0, -1.0]))
    assert set(center.metadata["atlas_geometry_kind"]) == {"configured_center_of_mass"}
    assert torch.allclose(
        center.batch.positions[:, 0] - center.batch.positions[:, 1],
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(4, -1),
    )
    assert set(shell.metadata["atlas_coordinate_kind"]) == {"angular_shell_radius"}
    assert set(spectator.metadata["direction_id"].tolist()) == {0, 1}
    assert set(center.metadata["direction_id"].tolist()) == {0, 1}
    assert set(shell.metadata["direction_id"].tolist()) == {0, 1}


def test_second_derivative_keeps_graph_and_matches_slow_reference_and_closed_form(
    tmp_path: Path,
) -> None:
    generated = _one_electron_generated(tmp_path)
    tangent = generated.metadata["coordinate_tangent"]
    model = _AnalyticAtlasModel()

    fast = directional_log_derivatives(model, generated.batch, tangent)
    slow = directional_log_derivatives_reference(model, generated.batch, tangent)
    for actual, expected in zip(fast, slow):
        assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(fast[2], torch.full_like(fast[2], -1.5))


def test_per_electron_kinetic_matches_slow_reference_and_closed_form(tmp_path: Path) -> None:
    batch = _one_electron_generated(tmp_path).batch
    model = _AnalyticAtlasModel()
    fast = per_electron_kinetic_from_logabs(model, batch)
    slow = per_electron_kinetic_from_logabs_reference(model, batch)
    expected = 2.25 - 1.125 * batch.positions.square().sum(dim=-1)

    assert torch.allclose(fast, slow, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(fast, expected, atol=1.0e-12, rtol=1.0e-12)

    counting_model = _CountingAnalyticAtlasModel()
    result = KineticEnergy().local_energy(counting_model, batch)
    assert counting_model.forward_count == 1
    assert result.wavefunction_output is not None
    result.wavefunction_output.validate(batch_size=batch.batch_size)
    assert result.per_electron_kinetic is not None
    assert torch.allclose(
        result.per_electron_kinetic.sum(dim=1),
        result.total,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_atlas_reuses_the_records_kinetic_pass_per_chunk(tmp_path: Path) -> None:
    generated = _one_electron_generated(tmp_path)
    model = _CountingAnalyticAtlasModel()

    _calculator().calculate(
        model=model,
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
    )

    chunk_count = (generated.batch.batch_size + 2) // 3
    # One complete-model directional derivative plus one kinetic/local-energy
    # pass per chunk. Per-electron attribution shares the latter.
    assert model.forward_count == 2 * chunk_count


def test_calculator_emits_typed_named_terms_float64_cancellation_and_permutation(
    tmp_path: Path,
) -> None:
    bundle = _calculated_one_electron(tmp_path)
    atlas = bundle.helium_atlas
    assert isinstance(atlas, HeliumAtlasValues)
    atlas.validate(bundle.generated.batch)
    assert tuple(atlas.derivatives) == (
        "executed_full_logabs",
        "executed_smoothed_ee_factor",
        "executed_electron_nucleus_factor",
    )
    assert tuple(atlas.hamiltonian_terms) == ("kinetic", "constant")
    assert atlas.cancellation_abs_sum.dtype == torch.float64
    assert atlas.cancellation_residual.dtype == torch.float64
    assert atlas.cancellation_ratio.dtype == torch.float64
    assert torch.allclose(
        atlas.per_electron_kinetic.sum(dim=1),
        atlas.hamiltonian_terms["kinetic"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert set(status for row in atlas.per_electron_kinetic_status for status in row) == {
        "defined"
    }
    invalid_status = replace(
        atlas,
        domain_status=("exact_zero_sentinel",) * len(atlas.domain_status),
    )
    with pytest.raises(ValueError, match="domain_status must exactly encode"):
        invalid_status.validate(bundle.generated.batch)

    permutation = Permutation((1, 0))
    metadata = dict(bundle.generated.metadata)
    metadata["coordinate_tangent"] = permute_particle_axis(
        metadata["coordinate_tangent"], permutation, axis=1
    )
    metadata["probe_electron"] = 1 - metadata["probe_electron"]
    permuted_generated = GeneratedConfigurations(
        batch=bundle.generated.batch.permute(permutation),
        metadata=metadata,
    )
    recalculated = _calculator().calculate(
        model=_AnalyticAtlasModel(),
        bundle=EvaluationBundle(generated=permuted_generated),
        context=_context(tmp_path),
    ).helium_atlas
    assert isinstance(recalculated, HeliumAtlasValues)
    close, metrics = atlas.permute(permutation).compare(
        recalculated, atol=1.0e-12, rtol=1.0e-12
    )
    assert close
    assert metrics["status_mismatch_count"] == 0


def test_undefined_per_electron_kinetic_is_nan_with_explicit_domain_status(
    tmp_path: Path,
) -> None:
    generated = _one_electron_generated(tmp_path)
    calculator = HeliumAtlasCalculator(
        hamiltonian_terms={"constant": _ConstantTerm(1.0)},
        factor_indices={
            "executed_smoothed_ee_factor": 0,
            "executed_electron_nucleus_factor": 1,
        },
        chunk_size=3,
    )
    atlas = calculator.calculate(
        model=_AnalyticAtlasModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
    ).helium_atlas

    assert isinstance(atlas, HeliumAtlasValues)
    assert torch.isnan(atlas.per_electron_kinetic).all()
    assert not atlas.per_electron_kinetic_domain_mask.any()
    assert set(status for row in atlas.per_electron_kinetic_status for status in row) == {
        "undefined_no_kinetic_registry_term"
    }


def test_curvature_and_all_five_tail_quantities_are_named_and_analytic(tmp_path: Path) -> None:
    bundle = _calculated_one_electron(tmp_path)
    curvature = HeliumCurvatureSummary(
        series_name="executed_full_logabs",
        metric_prefix="executed_full_logabs_curvature",
        windows={"inner": 2.0, "outer": 4.0},
    ).summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/helium_atlas",
    ).metrics
    tail = HeliumTailSummary(
        series_name="executed_full_logabs",
        metric_prefix="executed_full_logabs_tail",
    ).summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/helium_atlas",
    ).metrics

    assert curvature["executed_full_logabs_curvature_inner_second_derivative_mean"] == pytest.approx(-1.5)
    assert curvature["executed_full_logabs_curvature_outer_second_derivative_mean"] == pytest.approx(-1.5)
    assert tail["executed_full_logabs_tail_slope"] == pytest.approx(-6.0)
    assert tail["executed_full_logabs_tail_extrema_min"] == pytest.approx(-6.0)
    assert tail["executed_full_logabs_tail_extrema_max"] == pytest.approx(-6.0)
    assert tail["executed_full_logabs_tail_sign_fraction"] == pytest.approx(1.0)
    assert tail["executed_full_logabs_tail_outer_radius"] == pytest.approx(4.0)
    assert tail["executed_full_logabs_tail_directional_spread"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="strictly nested"):
        HeliumCurvatureSummary(
            series_name="executed_full_logabs",
            metric_prefix="executed_full_logabs_curvature",
            windows={"outer": 4.0, "inner": 2.0},
        )


def test_writer_retains_nonfinite_boundary_and_distinguishes_zero_sentinel(
    tmp_path: Path,
) -> None:
    generated = HeliumElectronElectronApproachGenerator(
        atoms=_atoms(),
        directions=[[1.0, 0.0, 0.0]],
        center_of_mass=[0.0, 0.0, 0.0],
        start_radius=1.0e-2,
        refinement_ratio=1.0e-64,
        max_refinement_steps=16,
    ).generate(model=None, context=_context(tmp_path))
    calculator = _calculator(electron_electron_name="electron_electron")
    bundle = calculator.calculate(
        model=_AnalyticAtlasModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
    )
    numerical = HeliumNumericalLimitSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/helium_atlas",
    )
    result = HeliumAtlasWriter().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/helium_atlas",
    )

    assert numerical.metrics["atlas_coordinate_representability_boundary_count"] == 1
    assert numerical.metrics[
        "ideal_unfloored_ee_reciprocal_failure_boundary_count"
    ] == 1
    assert numerical.metrics["atlas_exact_zero_sentinel_count"] == 1
    assert numerical.metrics[
        "ideal_unfloored_ee_reciprocal_evaluation_undefined_count"
    ] >= 1
    rows = list(csv.DictReader(result.artifacts[0].path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == generated.batch.batch_size
    sentinel = next(row for row in rows if row["is_exact_zero_sentinel"] == "True")
    coordinate_boundary = next(
        row
        for row in rows
        if row["is_coordinate_representability_boundary"] == "True"
    )
    reciprocal_boundary = next(
        row
        for row in rows
        if row["is_ideal_unfloored_ee_reciprocal_failure_boundary"] == "True"
    )
    coordinate_boundary_radius = float(
        coordinate_boundary["coordinate_representability_boundary_radius"]
    )
    reciprocal_boundary_radius = float(
        reciprocal_boundary["ideal_unfloored_ee_reciprocal_failure_radius"]
    )
    assert coordinate_boundary_radius == float(
        coordinate_boundary["realized_physical_coordinate"]
    )
    assert reciprocal_boundary_radius == float(
        reciprocal_boundary["realized_physical_coordinate"]
    )
    assert coordinate_boundary_radius >= 0
    assert reciprocal_boundary_radius >= 0
    assert coordinate_boundary["sample_index"] != sentinel["sample_index"]
    assert reciprocal_boundary["sample_index"] != sentinel["sample_index"]
    assert sentinel["atlas_sample_kind"] == "exact_zero_sentinel"
    assert sentinel["ideal_unfloored_ee_inverse_distance"] == "inf"
    assert sentinel["ideal_unfloored_ee_positive_separation_domain"] == "False"
    assert sentinel["ideal_unfloored_ee_reciprocal_evaluation_defined"] == "False"
    assert reciprocal_boundary[
        "ideal_unfloored_ee_positive_separation_domain"
    ] == str(float(reciprocal_boundary["realized_physical_coordinate"]) > 0)
    assert reciprocal_boundary[
        "ideal_unfloored_ee_reciprocal_evaluation_defined"
    ] == "False"
    assert sentinel["domain_status"] == "exact_zero_sentinel"
    assert sentinel["executed_smoothed_ee_factor_value_finite"] == "True"
    assert sentinel["executed_hamiltonian_cancellation_abs_sum_finite"] in {
        "True",
        "False",
    }
    assert sentinel["executed_hamiltonian_cancellation_residual_finite"] in {
        "True",
        "False",
    }
    assert sentinel["executed_hamiltonian_cancellation_ratio_finite"] in {
        "True",
        "False",
    }
    assert (
        "executed_smoothed_physical_separation_hamiltonian_term/electron_electron"
        in sentinel
    )

    ideal = bundle.helium_atlas.ideal_unfloored_ee_inverse_distance
    assert ideal is not None
    changed_ideal = ideal.clone()
    changed_ideal[0] += 1.0
    changed = replace(
        bundle.helium_atlas,
        ideal_unfloored_ee_inverse_distance=changed_ideal,
    )
    close, _ = bundle.helium_atlas.compare(changed, atol=1.0e-12, rtol=1.0e-12)
    assert not close


def test_real_executed_ee_factor_value_remains_finite_where_ideal_is_undefined(
    tmp_path: Path,
) -> None:
    """The rational cusp value is finite at coalescence without smoothing.

    Its value ``c*r/(1+d*r)`` is zero at ``r=0`` on its own merits. This test
    intentionally does not assert first- or second-derivative finiteness at
    exact coalescence: those Cartesian derivatives are not supplied by the
    analytic ``eps=0`` contract and the finite-eps e-e regime is UNMEASURED.
    """
    generated = HeliumElectronElectronApproachGenerator(
        atoms=_atoms(),
        directions=[[1.0, 0.0, 0.0]],
        center_of_mass=[0.0, 0.0, 0.0],
        start_radius=1.0e-2,
        refinement_ratio=1.0e-64,
        max_refinement_steps=16,
    ).generate(model=None, context=_context(tmp_path))
    atlas = HeliumAtlasCalculator(
        hamiltonian_terms={"constant": _ConstantTerm(1.0)},
        factor_indices={"executed_smoothed_ee_factor": 0},
        chunk_size=3,
    ).calculate(
        model=_ExecutedCuspModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
    ).helium_atlas

    assert isinstance(atlas, HeliumAtlasValues)
    sentinel = atlas.is_exact_zero_sentinel
    ideal = atlas.ideal_unfloored_ee_inverse_distance
    assert ideal is not None
    assert torch.isinf(ideal[sentinel]).all()
    reciprocal_boundary = atlas.is_ideal_unfloored_ee_reciprocal_failure_boundary
    assert reciprocal_boundary is not None
    assert int(reciprocal_boundary.sum().item()) == 1
    positive_domain = atlas.ideal_unfloored_ee_positive_separation_domain_mask
    evaluation_defined = (
        atlas.ideal_unfloored_ee_reciprocal_evaluation_defined_mask
    )
    assert positive_domain is not None
    assert evaluation_defined is not None
    assert torch.equal(
        positive_domain[reciprocal_boundary],
        atlas.realized_coordinate[reciprocal_boundary] > 0,
    )
    assert not evaluation_defined[reciprocal_boundary].any()
    # The registry label remains ``executed_smoothed_ee_factor`` for the
    # established atlas schema; the factor itself is no longer distance-
    # smoothed by default.
    executed = atlas.derivatives["executed_smoothed_ee_factor"]
    assert executed.value_finite_mask[reciprocal_boundary].all()
    assert executed.value_finite_mask[sentinel].all()


def test_ambiguous_label_and_float32_are_fail_closed(tmp_path: Path) -> None:
    generated = HeliumElectronElectronApproachGenerator(
        atoms=_atoms(),
        directions=[[1.0, 0.0, 0.0]],
        center_of_mass=[0.0, 0.0, 0.0],
        start_radius=1.0e-2,
        refinement_ratio=1.0e-64,
        max_refinement_steps=16,
    ).generate(model=None, context=_context(tmp_path))
    ambiguous = HeliumAtlasCalculator(
        hamiltonian_terms={"constant": _ConstantTerm(1.0)},
        factor_indices={"electron_electron_factor": 0},
    )
    with pytest.raises(ValueError, match="executed_smoothed_ee"):
        ambiguous.calculate(
            model=_ExecutedCuspModel(),
            bundle=EvaluationBundle(generated=generated),
            context=_context(tmp_path),
        )

    with pytest.raises(ValueError, match="chunk_size"):
        HeliumAtlasCalculator(
            hamiltonian_terms={"constant": _ConstantTerm(1.0)},
            factor_indices={"executed_smoothed_ee_factor": 0},
            chunk_size=0,
        )

    float32_context = _context(tmp_path, dtype=torch.float32)
    float32_generated = HeliumOneElectronEscapeGenerator(
        atoms=_atoms(),
        directions=[[1.0, 0.0, 0.0]],
        radii=[1.0, 2.0],
        spectator_position=[0.0, 0.0, -1.0],
    ).generate(model=None, context=float32_context)
    with pytest.raises(ValueError, match="float64"):
        _calculator().calculate(
            model=_AnalyticAtlasModel().to(dtype=torch.float32),
            bundle=EvaluationBundle(generated=float32_generated),
            context=float32_context,
        )
