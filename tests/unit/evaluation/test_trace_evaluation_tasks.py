"""Tests for orbit and trace evaluation tasks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.evaluation.bundle import GeneratedConfigurations
from spenn.evaluation.calculators import (
    FeatureTraceCalculator,
    FullModelAntisymmetryCalculator,
    ReadoutTraceCalculator,
    RotationConsistencyCalculator,
    SpatialExchangeSymmetryCalculator,
    TraceEquivarianceCalculator,
)
from spenn.evaluation.generators import (
    ExchangeOrbitGenerator,
    PermutationOrbitGenerator,
    RotationOrbitGenerator,
)
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.summaries import (
    FeatureTraceSummary,
    ReadoutTraceSummary,
    TraceEquivarianceSummary,
    TransformConsistencySummary,
)
from spenn.trace import ParticleTensor, trace_value


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
    from spenn.evaluation.bundle import EvaluationBundle

    return EvaluationBundle(generated=generated)
