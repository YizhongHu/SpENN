"""Contracts for atom-owned helium radial evaluation support."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.geometry import electron_nuclear_displacements
from tpen.data.permutation import Permutation
from tpen.evaluation.bundle import (
    ElectronNucleusRadialValues,
    EvaluationBundle,
    GeneratedConfigurations,
)
from tpen.evaluation.calculators import ElectronNucleusRadialCalculator
from tpen.evaluation.generators import HeliumRadialGridGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries import (
    ElectronNucleusCuspSummary,
    ElectronNucleusRadialProfileWriter,
    ElectronNucleusTailSummary,
)


class _ExactHeliumRadialModel(torch.nn.Module):
    """Simple ``log|psi| = -2 sum_i r_i`` radial contract model."""

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        distance = electron_nuclear_displacements(batch).norm(dim=-1)
        logabs = -2.0 * distance.sum(dim=(-2, -1))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _context(tmp_path: Path, *, artifact_level: str = "summaries") -> EvaluationContext:
    return EvaluationContext(
        namespace="eval/he_radial_profiles",
        artifact_level=artifact_level,  # type: ignore[arg-type]
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _generated(tmp_path: Path) -> GeneratedConfigurations:
    return HeliumRadialGridGenerator(
        cusp_radii=[1.0e-5, 1.0e-4, 1.0e-3],
        tail_radii=[4.0, 8.0],
        spectator_radius=1.0,
        nuclear_positions=[[0.0, 0.0, 0.0]],
        nuclear_charges=[2.0],
        n_directions=2,
    ).generate(model=None, context=_context(tmp_path))


def _calculated(tmp_path: Path) -> EvaluationBundle:
    generated = _generated(tmp_path)
    return ElectronNucleusRadialCalculator(chunk_size=5).calculate(
        model=_ExactHeliumRadialModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
    )


def test_helium_radial_grid_uses_positive_distinct_regions_and_nuclear_context(tmp_path: Path) -> None:
    generated = _generated(tmp_path)
    batch = generated.batch
    regions = generated.metadata["profile_region"]
    radii = generated.metadata["radius"]

    assert batch.positions.shape == (40, 2, 3)
    assert torch.equal(batch.nuclear_positions, torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))
    assert torch.equal(batch.nuclear_charges, torch.tensor([2.0], dtype=torch.float64))
    assert torch.all(radii > 0)
    assert set(generated.metadata["direction_sign"].tolist()) == {-1, 1}
    assert max(radii[index].item() for index, value in enumerate(regions) if value == "cusp") < min(
        radii[index].item() for index, value in enumerate(regions) if value == "tail"
    )

    permuted = batch.permute(Permutation((1, 0)))
    assert torch.equal(permuted.nuclear_positions, batch.nuclear_positions)
    assert torch.equal(permuted.nuclear_charges, batch.nuclear_charges)
    assert torch.equal(permuted.positions, batch.positions[:, [1, 0], :])


def test_helium_radial_grid_rejects_zero_cusp_and_overlapping_tail() -> None:
    kwargs = {
        "spectator_radius": 1.0,
        "nuclear_positions": [[0.0, 0.0, 0.0]],
        "nuclear_charges": [2.0],
    }
    with pytest.raises(ValueError, match="strictly positive"):
        HeliumRadialGridGenerator(cusp_radii=[0.0, 0.1], tail_radii=[4.0], **kwargs)
    with pytest.raises(ValueError, match="below tail"):
        HeliumRadialGridGenerator(cusp_radii=[0.1, 1.0], tail_radii=[0.5, 2.0], **kwargs)


def test_radial_result_validates_and_permute_compare_is_semantic(tmp_path: Path) -> None:
    bundle = _calculated(tmp_path)
    values = bundle.electron_nucleus_radial
    assert isinstance(values, ElectronNucleusRadialValues)
    values.validate(bundle.generated.batch)
    assert values.distance.shape == (40, 2, 1)
    assert bool(values.finite_mask.all())

    permutation = Permutation((1, 0))
    permuted_batch = bundle.generated.batch.permute(permutation)
    semantic_permutation = values.permute(permutation).validate(permuted_batch)
    recalculated = ElectronNucleusRadialCalculator().calculate(
        model=_ExactHeliumRadialModel(),
        bundle=EvaluationBundle(
            generated=GeneratedConfigurations(
                batch=permuted_batch,
                metadata=bundle.generated.metadata,
            )
        ),
        context=_context(tmp_path),
    ).electron_nucleus_radial
    assert isinstance(recalculated, ElectronNucleusRadialValues)
    close, metrics = semantic_permutation.compare(recalculated, atol=1.0e-12, rtol=1.0e-12)
    assert close
    assert metrics["max_abs_error"] == pytest.approx(0.0)
    assert metrics["finite_mask_mismatch_count"] == 0


def test_cusp_tail_and_profile_outputs_report_availability_and_finite_counts(tmp_path: Path) -> None:
    bundle = _calculated(tmp_path)
    cusp = ElectronNucleusCuspSummary(max_fit_points=3).summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/he_radial_profiles",
    )
    tail = ElectronNucleusTailSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/he_radial_profiles",
    )
    profile = ElectronNucleusRadialProfileWriter().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="eval/he_radial_profiles",
    )

    assert cusp.metrics["cusp_available"] is True
    assert cusp.metrics["cusp_finite_measurement_count"] == 24
    assert cusp.metrics["cusp_finite_fit_count"] == 4
    assert cusp.metrics["cusp_one_sided_slope_mean"] == pytest.approx(-2.0)
    assert cusp.metrics["cusp_one_sided_slope_abs_error_max"] == pytest.approx(0.0, abs=1.0e-10)
    assert tail.metrics["tail_available"] is True
    assert tail.metrics["tail_finite_measurement_count"] == 16
    assert tail.metrics["tail_outer_measurement_count"] == 4
    assert tail.metrics["tail_outer_slope_mean"] == pytest.approx(-2.0)
    assert tail.metrics["tail_negative_slope_fraction"] == pytest.approx(1.0)
    assert profile.metrics == {
        "profile_available": True,
        "profile_finite_measurement_count": 40,
        "profile_total_measurement_count": 40,
    }
    assert len(profile.artifacts) == 1
    artifact = profile.artifacts[0]
    assert artifact.metadata == {
        "available": True,
        "finite_measurement_count": 40,
        "total_measurement_count": 40,
    }
    rows = list(csv.DictReader(artifact.path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 40
    assert {row["profile_region"] for row in rows} == {"cusp", "tail"}
    assert {row["available"] for row in rows} == {"True"}
    assert {row["finite_measurement_count"] for row in rows} == {"40"}
    assert {row["total_measurement_count"] for row in rows} == {"40"}
    assert "logabs" not in rows[0]


def test_nonfinite_radial_measurements_are_unavailable_not_zero(tmp_path: Path) -> None:
    bundle = _calculated(tmp_path)
    values = bundle.electron_nucleus_radial
    assert isinstance(values, ElectronNucleusRadialValues)
    unavailable = ElectronNucleusRadialValues(
        distance=values.distance,
        radial_dlogabs=torch.full_like(values.radial_dlogabs, float("nan")),
        finite_mask=torch.zeros_like(values.finite_mask),
    ).validate(bundle.generated.batch)
    unavailable_bundle = EvaluationBundle(
        generated=bundle.generated,
        electron_nucleus_radial=unavailable,
    )

    cusp = ElectronNucleusCuspSummary().summarize(
        bundle=unavailable_bundle,
        context=_context(tmp_path),
        namespace="eval/he_radial_profiles",
    ).metrics
    tail = ElectronNucleusTailSummary().summarize(
        bundle=unavailable_bundle,
        context=_context(tmp_path),
        namespace="eval/he_radial_profiles",
    ).metrics
    profile = ElectronNucleusRadialProfileWriter().summarize(
        bundle=unavailable_bundle,
        context=_context(tmp_path, artifact_level="metrics_only"),
        namespace="eval/he_radial_profiles",
    )

    assert cusp["cusp_available"] is False
    assert cusp["cusp_finite_measurement_count"] == 0
    assert "cusp_one_sided_slope_mean" not in cusp
    assert tail["tail_available"] is False
    assert tail["tail_finite_measurement_count"] == 0
    assert "tail_outer_slope_mean" not in tail
    assert profile.metrics["profile_available"] is False
    assert profile.metrics["profile_finite_measurement_count"] == 0
    assert profile.artifacts == ()
