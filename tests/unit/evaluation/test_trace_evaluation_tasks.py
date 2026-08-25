"""Tests for orbit and trace evaluation tasks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.bundle import GeneratedConfigurations, TransformKind, TransformName
from tpen.evaluation.calculators import (
    FeatureTraceCalculator,
    FullModelAntisymmetryCalculator,
    ReadoutTraceCalculator,
    RotationConsistencyCalculator,
    SpatialExchangeSymmetryCalculator,
    TraceEquivarianceCalculator,
)
from tpen.evaluation.generators import (
    ExchangeOrbitGenerator,
    PermutationOrbitGenerator,
    RotationOrbitGenerator,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries import (
    FeatureTraceSummary,
    ReadoutTraceSummary,
    TraceEquivarianceSummary,
    TransformConsistencySummary,
    TransformRecordWriter,
)
from tpen.trace import ParticleTensor, trace_value


def _context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        namespace="validation/full_model_antisymmetry",
        artifact_level="metrics_only",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _base_batch() -> ElectronBatch:
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    spins = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)


class _StaticGenerator:
    name = "static"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        return GeneratedConfigurations(batch=_base_batch(), metadata={"sample_index": torch.arange(2)})


class _FermionicModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        logabs = flat.positions.square().sum(dim=(1, 2))
        sign = torch.sign(flat.positions[:, 0, 0] - flat.positions[:, 1, 0])
        return WavefunctionOutput(logabs=logabs, sign=sign)


class _SymmetricTraceModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        trace_value(
            ParticleTensor(flat.positions, particle_axis=1),
            key="positions",
            slot="features",
            semantic_type="features",
        )
        matrix = flat.positions[:, :, :2] @ flat.positions[:, :, :2].transpose(-1, -2)
        logabs = flat.positions.square().sum(dim=(1, 2))
        sign = torch.ones_like(logabs)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": matrix})


class _IdentityModel(nn.Module):
    """Return geometry-sensitive finite values for transform identity tests."""

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        weights = torch.arange(
            1,
            flat.n_electrons * flat.spatial_dim + 1,
            device=flat.device,
            dtype=flat.dtype,
        ).reshape(flat.n_electrons, flat.spatial_dim)
        logabs = (flat.positions * weights).sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _NonfiniteIdentityModel(_IdentityModel):
    """Make transformed negative-leading geometries nonfinite."""

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        output = super().forward(batch)
        flat = batch.flatten_samples()
        logabs = output.logabs.clone()
        logabs[flat.positions[:, 0, 0] < 0.0] = torch.nan
        return WavefunctionOutput(logabs=logabs, sign=output.sign)


class _FiveSampleGenerator:
    name = "five_sample"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        del model, context
        positions = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 5,
            dtype=torch.float64,
        )
        return GeneratedConfigurations(
            batch=ElectronBatch(
                positions=positions,
                spins=torch.tensor([[1.0, -1.0]] * 5, dtype=torch.float64),
            ),
            metadata={"sample_index": torch.arange(5)},
        )


class _MissingAndExtraTraceModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        value = ParticleTensor(flat.positions, particle_axis=1)
        trace_value(value, key="shared", slot="features", semantic_type="features")
        key = "left_only" if float(flat.positions[:, 0, 0].mean()) < 0.5 else "right_only"
        trace_value(value, key=key, slot="features", semantic_type="features")
        logabs = flat.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_permutation_orbit_and_full_model_antisymmetry_summary(tmp_path: Path) -> None:
    generated = PermutationOrbitGenerator(
        base_generator=_StaticGenerator(),
        permutations=[torch.tensor([1, 0])],
    ).generate(model=None, context=_context(tmp_path))

    bundle = FullModelAntisymmetryCalculator().calculate(
        model=_FermionicModel(),
        bundle=_bundle(generated),
        context=_context(tmp_path),
    )
    metrics = TransformConsistencySummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/full_model_antisymmetry",
    ).metrics

    assert generated.batch.sample_shape == (2, 2)
    assert metrics["logabs_max_abs_error"] == pytest.approx(0.0)
    assert metrics["sign_failure_count"] == 0


def test_full_model_antisymmetry_requires_permutation_parity(tmp_path: Path) -> None:
    generated = ExchangeOrbitGenerator(base_generator=_StaticGenerator()).generate(model=None, context=_context(tmp_path))

    with pytest.raises(ValueError, match="permutation_parity"):
        FullModelAntisymmetryCalculator().calculate(
            model=_SymmetricTraceModel(),
            bundle=_bundle(generated),
            context=_context(tmp_path),
        )


def test_rotation_and_exchange_transform_summaries(tmp_path: Path) -> None:
    rotation_generated = RotationOrbitGenerator(
        base_generator=_StaticGenerator(),
        n_rotations=2,
        seed=0,
    ).generate(model=None, context=_context(tmp_path))
    rotation_bundle = RotationConsistencyCalculator().calculate(
        model=_SymmetricTraceModel(),
        bundle=_bundle(rotation_generated),
        context=_context(tmp_path),
    )
    rotation_metrics = TransformConsistencySummary().summarize(
        bundle=rotation_bundle,
        context=_context(tmp_path),
        namespace="validation/rotation",
    ).metrics

    exchange_generated = ExchangeOrbitGenerator(base_generator=_StaticGenerator()).generate(model=None, context=_context(tmp_path))
    exchange_bundle = SpatialExchangeSymmetryCalculator().calculate(
        model=_SymmetricTraceModel(),
        bundle=_bundle(exchange_generated),
        context=_context(tmp_path),
    )
    exchange_metrics = TransformConsistencySummary().summarize(
        bundle=exchange_bundle,
        context=_context(tmp_path),
        namespace="validation/exchange",
    ).metrics

    assert rotation_metrics["logabs_max_abs_error"] == pytest.approx(0.0, abs=1.0e-10)
    assert exchange_metrics["sign_failure_count"] == 0


@pytest.mark.parametrize(
    ("case", "expected_name", "expected_kind"),
    [
        (
            "full_model_antisymmetry",
            TransformName.FULL_MODEL_ANTISYMMETRY,
            TransformKind.FULL_MODEL_ANTISYMMETRY,
        ),
        (
            "spatial_exchange_symmetry",
            TransformName.SPATIAL_EXCHANGE_SYMMETRY,
            TransformKind.SPATIAL_EXCHANGE,
        ),
        (
            "rotation_consistency",
            TransformName.ROTATION_CONSISTENCY,
            TransformKind.ROTATION_CONSISTENCY,
        ),
    ],
)
def test_transform_calculators_populate_required_typed_identity(
    tmp_path: Path,
    case: str,
    expected_name: TransformName,
    expected_kind: TransformKind,
) -> None:
    bundle = _calculate_identity_case(case, tmp_path)
    transform = bundle.transform
    assert transform is not None
    paired_positions = bundle.generated.batch.positions.reshape(
        -1,
        2,
        bundle.generated.batch.n_electrons,
        bundle.generated.batch.spatial_dim,
    )

    assert transform.transform_name is expected_name
    assert transform.transform_kind is expected_kind
    torch.testing.assert_close(
        transform.sample_index,
        bundle.generated.metadata["base_sample_index"],
    )
    torch.testing.assert_close(transform.original_positions, paired_positions[:, 0])
    torch.testing.assert_close(transform.transformed_positions, paired_positions[:, 1])
    weights = torch.arange(
        1,
        transform.original_positions.shape[-2] * transform.original_positions.shape[-1] + 1,
        dtype=transform.original_positions.dtype,
    ).reshape(transform.original_positions.shape[-2:])
    torch.testing.assert_close(
        transform.original_logabs,
        (transform.original_positions * weights).sum(dim=(1, 2)),
    )
    torch.testing.assert_close(
        transform.transformed_logabs,
        (transform.transformed_positions * weights).sum(dim=(1, 2)),
    )
    assert transform.finite.dtype == torch.bool
    assert bool(transform.finite.all().item())
    assert "transform_name" not in transform.metadata
    assert "transform_kind" not in transform.metadata


@pytest.mark.parametrize(
    "case",
    [
        "full_model_antisymmetry",
        "spatial_exchange_symmetry",
        "rotation_consistency",
    ],
)
def test_bounded_transform_writer_preserves_calculator_identity(tmp_path: Path, case: str) -> None:
    bundle = _calculate_identity_case(case, tmp_path)
    transform = bundle.transform
    assert transform is not None
    result = TransformRecordWriter(max_records=1).summarize(
        bundle=bundle,
        context=_records_context(tmp_path, namespace=f"validation/{case}"),
        namespace=f"validation/{case}",
    )

    assert result.artifacts[0].metadata["rows"] == 1
    with (tmp_path / "transform_records.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    required_columns = {
        "sample_index",
        "transform",
        "transform_kind",
        "finite",
        "original_geometry",
        "transformed_geometry",
        "original_logabs",
        "transformed_logabs",
    }
    assert required_columns <= set(reader.fieldnames or ())
    assert row["sample_index"] == str(int(transform.sample_index[0].item()))
    assert row["transform"] == transform.transform_name.value
    assert row["transform_kind"] == transform.transform_kind.value
    assert row["finite"] == str(bool(transform.finite[0].item()))
    assert json.loads(row["original_geometry"]) == transform.original_positions[0].tolist()
    assert json.loads(row["transformed_geometry"]) == transform.transformed_positions[0].tolist()
    assert float(row["original_logabs"]) == pytest.approx(
        float(transform.original_logabs[0].item())
    )
    assert float(row["transformed_logabs"]) == pytest.approx(
        float(transform.transformed_logabs[0].item())
    )


def test_transform_writer_preserves_calculator_sample_index_not_record_order(tmp_path: Path) -> None:
    bundle = _calculate_identity_case("rotation_consistency", tmp_path, n_rotations=2)
    transform = bundle.transform
    assert transform is not None
    TransformRecordWriter(max_records=3).summarize(
        bundle=bundle,
        context=_records_context(tmp_path, namespace="validation/rotation_consistency"),
        namespace="validation/rotation_consistency",
    )

    with (tmp_path / "transform_records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[2]["record_index"] == "2"
    assert int(transform.sample_index[2].item()) == 0
    assert rows[2]["sample_index"] == "0"


def test_transform_writer_preserves_calculator_nonfinite_status(tmp_path: Path) -> None:
    bundle = _calculate_identity_case(
        "spatial_exchange_symmetry",
        tmp_path,
        model=_NonfiniteIdentityModel(),
    )
    transform = bundle.transform
    assert transform is not None
    assert transform.finite.tolist() == [True, False]
    TransformRecordWriter(max_records=2).summarize(
        bundle=bundle,
        context=_records_context(tmp_path, namespace="validation/spatial_exchange_symmetry"),
        namespace="validation/spatial_exchange_symmetry",
    )

    with (tmp_path / "transform_records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[1]["finite"] == "False"
    assert rows[1]["original_logabs"] != "nan"
    assert rows[1]["transformed_logabs"] == "nan"


def test_trace_equivariance_calculator_compares_particle_tensor(tmp_path: Path) -> None:
    generated = PermutationOrbitGenerator(
        base_generator=_StaticGenerator(),
        permutations=[torch.tensor([1, 0])],
    ).generate(model=None, context=_context(tmp_path))
    bundle = TraceEquivarianceCalculator(compare_slots=["features"]).calculate(
        model=_SymmetricTraceModel(),
        bundle=_bundle(generated),
        context=_context(tmp_path),
    )
    metrics = TraceEquivarianceSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/trace_equivariance",
    ).metrics

    assert metrics["failure_count"] == 0
    assert metrics["compared_entry_count"] > 0
    assert metrics["comparison_error_count"] == 0
    assert metrics["max_abs_error"] == pytest.approx(0.0)


def test_trace_equivariance_rejects_vacuous_trace(tmp_path: Path) -> None:
    generated = PermutationOrbitGenerator(
        base_generator=_StaticGenerator(),
        permutations=[torch.tensor([1, 0])],
    ).generate(model=None, context=_context(tmp_path))

    with pytest.raises(ValueError, match="zero trace entries"):
        TraceEquivarianceCalculator().calculate(
            model=_FermionicModel(),
            bundle=_bundle(generated),
            context=_context(tmp_path),
        )


def test_trace_summary_reports_typed_per_key_coverage_and_missing_extra(tmp_path: Path) -> None:
    generated = PermutationOrbitGenerator(
        base_generator=_FiveSampleGenerator(),
        permutations=[torch.tensor([1, 0])],
    ).generate(model=None, context=_context(tmp_path))
    bundle = TraceEquivarianceCalculator(compare_slots=["features"]).calculate(
        model=_MissingAndExtraTraceModel(),
        bundle=_bundle(generated),
        context=_context(tmp_path),
    )
    metrics = TraceEquivarianceSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/trace_equivariance",
    ).metrics

    assert metrics["compared_sample_count"] == 5
    assert metrics["missing_key_count"] == 1
    assert metrics["extra_key_count"] == 1
    assert metrics["key/shared/count"] == 1
    assert metrics["key/shared/sample_count"] == 5
    assert metrics["key/left_only/missing_key_count"] == 1
    assert metrics["key/right_only/extra_key_count"] == 1


def test_feature_and_readout_trace_summaries(tmp_path: Path) -> None:
    generated = _StaticGenerator().generate(model=None, context=_context(tmp_path))
    feature_bundle = FeatureTraceCalculator(slots=["features"]).calculate(
        model=_SymmetricTraceModel(),
        bundle=_bundle(generated),
        context=_context(tmp_path),
    )
    readout_bundle = ReadoutTraceCalculator().calculate(
        model=_SymmetricTraceModel(),
        bundle=_bundle(generated),
        context=_context(tmp_path),
    )

    feature_metrics = FeatureTraceSummary().summarize(
        bundle=feature_bundle,
        context=_context(tmp_path),
        namespace="validation/feature_trace",
    ).metrics
    readout_metrics = ReadoutTraceSummary().summarize(
        bundle=readout_bundle,
        context=_context(tmp_path),
        namespace="validation/readout_trace",
    ).metrics

    assert feature_metrics["feature_rms_max"] > 0.0
    assert readout_metrics["pfaffian_near_zero_count"] >= 0


def _bundle(generated: GeneratedConfigurations):
    from tpen.evaluation.bundle import EvaluationBundle

    return EvaluationBundle(generated=generated)


def _calculate_identity_case(
    case: str,
    tmp_path: Path,
    *,
    n_rotations: int = 1,
    model: nn.Module | None = None,
):
    if case == "full_model_antisymmetry":
        generator = PermutationOrbitGenerator(
            base_generator=_StaticGenerator(),
            permutations=[torch.tensor([1, 0])],
        )
        calculator = FullModelAntisymmetryCalculator(compare_sign=False)
    elif case == "spatial_exchange_symmetry":
        generator = ExchangeOrbitGenerator(base_generator=_StaticGenerator())
        calculator = SpatialExchangeSymmetryCalculator(compare_sign=False)
    elif case == "rotation_consistency":
        generator = RotationOrbitGenerator(
            base_generator=_StaticGenerator(),
            n_rotations=n_rotations,
            seed=0,
        )
        calculator = RotationConsistencyCalculator(compare_sign=False)
    else:
        raise AssertionError(f"unknown transform identity case {case!r}")
    generated = generator.generate(model=None, context=_context(tmp_path))
    return calculator.calculate(
        model=_IdentityModel() if model is None else model,
        bundle=_bundle(generated),
        context=_context(tmp_path),
    )


def _records_context(tmp_path: Path, *, namespace: str) -> EvaluationContext:
    return EvaluationContext(
        namespace=namespace,
        artifact_level="records",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )
