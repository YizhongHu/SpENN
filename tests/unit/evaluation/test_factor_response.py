"""Typed factor-arm scoping and aligned common-configuration records."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.evaluation.bundle import EvaluationBundle, FactorResponseArmValues, FactorResponseValues, GeneratedConfigurations
from tpen.evaluation.calculators.factor_response import FactorArmCalculator
from tpen.evaluation.factor_response import FactorParameterScale, helium_factor_parameter_scale
from tpen.evaluation.factor_response import _clone_preserving_layout, _parameter_matches_snapshot, _storage_byte_view
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries.factor_response import FactorResponseSummary
from tpen.nn.cusp import CurvatureElectronNucleusCuspLaw, ElectronElectronCusp, ElectronNucleusCusp


class _HeliumFactorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        atoms = AtomicConfiguration(positions=torch.zeros((1, 3), dtype=torch.float64),
                                    charges=torch.tensor([2.0], dtype=torch.float64))
        self.factors = nn.ModuleList([ElectronElectronCusp(trainable_range=True), ElectronNucleusCusp(
            atoms=atoms, law=CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.01,
            curvature_range=1.0, trainable=True))])
        self.unrelated = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))


def _physical(model: _HeliumFactorModel) -> tuple[float, float, float]:
    ee = model.factors[0]
    en = model.factors[1].law
    assert isinstance(ee, ElectronElectronCusp)
    assert isinstance(en, CurvatureElectronNucleusCuspLaw)
    return (float(ee.opposite_range_parameter.detach().item()),
            float(en.curvature_coefficient.detach().item()), float(en.curvature_range.detach().item()))


def _context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(namespace="test", artifact_level="records", task_failure_policy="fail_fast",
                              device=torch.device("cpu"), dtype=torch.float64, seed=1,
                              run_dir=tmp_path, task_output_dir=tmp_path, metadata={})


def _bundle(*, metadata: dict | None = None) -> EvaluationBundle:
    batch = ElectronBatch(positions=torch.zeros((2, 2, 3), dtype=torch.float64),
                          spins=torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64))
    return EvaluationBundle(generated=GeneratedConfigurations(batch=batch, metadata=metadata or {}))


def test_parameter_byte_guard_distinguishes_signed_zero() -> None:
    negative_zero = torch.tensor([-0.0], dtype=torch.float64)
    positive_zero = torch.tensor([0.0], dtype=torch.float64)
    assert not _parameter_matches_snapshot(negative_zero, positive_zero)


def test_parameter_byte_guard_distinguishes_layout_and_storage() -> None:
    snapshot = torch.ones((2, 2), dtype=torch.float64)
    same_storage_different_stride = snapshot.transpose(0, 1)
    assert not _parameter_matches_snapshot(same_storage_different_stride, snapshot)

    same_stride_different_storage = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 9.0, 10.0], dtype=torch.float64
    ).as_strided((2, 2), (2, 1))
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    assert not _parameter_matches_snapshot(same_stride_different_storage, reference)


def test_parameter_byte_guard_uses_the_tensor_storage_window() -> None:
    backing = torch.tensor([0.0, 1.0, 2.0, 99.0, 3.0, 4.0], dtype=torch.float64)
    view = backing.as_strided((2, 2), (3, 1))
    snapshot = _clone_preserving_layout(view)
    assert view.stride() == snapshot.stride()
    assert _storage_byte_view(view).shape == _storage_byte_view(snapshot).shape
    assert _parameter_matches_snapshot(view, snapshot)


def test_factor_scale_changes_physical_values_and_restores_every_parameter() -> None:
    model = _HeliumFactorModel()
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before = _physical(model)
    arm = FactorParameterScale(label="response", b_ee=0.9, c_electron_nucleus=1.1, d_electron_nucleus=0.9)
    with helium_factor_parameter_scale(model, arm) as realized:
        assert _physical(model) == pytest.approx((before[0] * 0.9, before[1] * 1.1, before[2] * 0.9))
        assert tuple(realized.values()) == pytest.approx(_physical(model))
    assert _physical(model) == pytest.approx(before)
    assert all(torch.equal(value, before_state[name]) for name, value in model.state_dict().items())


def test_factor_scope_restores_after_delegate_failure() -> None:
    model = _HeliumFactorModel()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(RuntimeError, match="synthetic"):
        with helium_factor_parameter_scale(model, {"label": "bad", "b_ee": 1.1}):
            raise RuntimeError("synthetic")
    assert all(torch.equal(value, before[name]) for name, value in model.named_parameters())


def test_factor_arm_calculator_reapplies_matching_reequilibrated_arm(tmp_path: Path) -> None:
    model = _HeliumFactorModel()
    before = _physical(model)
    seen: list[tuple[float, float, float]] = []

    class _Delegate:
        def calculate(self, *, model, bundle, context):
            del context
            seen.append(_physical(model))
            return bundle

    calculator = FactorArmCalculator(calculator=_Delegate(), arm={"label": "b_plus", "b_ee": 1.1})
    bundle = _bundle(metadata={"comparison_kind": "re_equilibrated", "factor_arm": "b_plus"})
    assert calculator.calculate(model=model, bundle=bundle, context=_context(tmp_path)) is bundle
    assert seen[0][0] == pytest.approx(before[0] * 1.1)
    assert _physical(model) == pytest.approx(before)
    with pytest.raises(ValueError, match="does not match"):
        calculator.calculate(model=model, bundle=_bundle(metadata={"comparison_kind": "re_equilibrated",
            "factor_arm": "different"}), context=_context(tmp_path))


def test_common_configuration_summary_writes_complete_aligned_grid(tmp_path: Path) -> None:
    bundle = _bundle()
    arms = tuple(FactorResponseArmValues(label=label, parameter_scales={"b_ee": scale},
        realized_parameters={"b_ee": scale}, local_energy=torch.tensor(values, dtype=torch.float64),
        logabs=torch.tensor([-1.0, -2.0], dtype=torch.float64) * scale, sign=torch.ones(2, dtype=torch.float64),
        term_energies={"kinetic": torch.tensor(values, dtype=torch.float64)})
        for label, scale, values in (("baseline", 1.0, [-2.0, -3.0]), ("b_plus", 1.1, [-2.1, -3.2])))
    bundle = EvaluationBundle(generated=bundle.generated, factor_response=FactorResponseValues(
        comparison_kind="common_configuration", baseline_label="baseline", arms=arms, model_state_restored=True))
    result = FactorResponseSummary(max_records=4).summarize(bundle=bundle, context=_context(tmp_path), namespace="test")
    assert result.artifacts[0].metadata["rows"] == 4
    lines = result.artifacts[0].path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert "comparison_kind" in lines[0]
    with pytest.raises(ValueError, match="truncate"):
        FactorResponseSummary(filename="too_small.csv", max_records=3).summarize(bundle=bundle,
            context=_context(tmp_path), namespace="test")
