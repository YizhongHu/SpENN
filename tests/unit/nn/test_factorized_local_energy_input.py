"""Factorized local-energy input and wavefunction seam contracts."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, FactorizedLocalEnergyInput, WavefunctionOutput
from tpen.data.permutation import Permutation
from tpen.data.real import Feature
from tpen.nn import CurvatureElectronNucleusCuspLaw, ElectronNucleusCusp, TPENWaveFunction


class EmptyEncoder(nn.Module):
    def forward(self, batch: ElectronBatch, *, context=None) -> Feature:
        return Feature()


class ConstantReadout(nn.Module):
    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        sign = torch.tensor([-1.0, 1.0], device=batch.device, dtype=batch.dtype)[: batch.batch_size]
        return WavefunctionOutput(logabs=logabs, sign=sign)


def test_factorized_local_energy_input_excludes_bound_cusp_and_reconstructs_full_score() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    cusp = ElectronNucleusCusp(atoms, CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.3, curvature_range=1.5))

    class PositionReadout(nn.Module):
        def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
            logabs = batch.positions.square().sum(dim=(1, 2))
            return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))

    positions = torch.tensor([[[0.4]], [[0.8]]], dtype=torch.float64, requires_grad=True)
    batch = ElectronBatch(positions=positions)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=PositionReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )

    full = model(batch)
    factorized = model.factorized_local_energy_input(batch)
    reconstructed = factorized.regular_wavefunction_output.logabs + factorized.electron_nucleus_cusp_evaluation.pair_value.sum(
        dim=(1, 2)
    )
    torch.testing.assert_close(
        torch.exp(reconstructed - reconstructed[0]),
        torch.exp(full.logabs - full.logabs[0]),
    )
    full_score = torch.autograd.grad(full.logabs.sum(), positions, retain_graph=True)[0]
    regular_score = torch.autograd.grad(factorized.regular_wavefunction_output.logabs.sum(), positions, retain_graph=True)[0]
    cusp_score = torch.autograd.grad(
        factorized.electron_nucleus_cusp_evaluation.pair_value.sum(), positions, retain_graph=True
    )[0]
    torch.testing.assert_close(regular_score + cusp_score, full_score)


def test_factorized_local_energy_input_calls_body_and_provider_exactly_once(monkeypatch) -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    body_calls = 0
    analytic_calls = 0

    class CountingReadout(ConstantReadout):
        def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
            nonlocal body_calls
            body_calls += 1
            return super().forward(features, batch)

    original = cusp.analytic_evaluation

    def analytic_spy(batch: ElectronBatch):
        nonlocal analytic_calls
        analytic_calls += 1
        return original(batch)

    monkeypatch.setattr(cusp, "analytic_evaluation", analytic_spy)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=CountingReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )

    model.factorized_local_energy_input(ElectronBatch(positions=torch.ones(1, 1, 1, dtype=torch.float64)))

    assert body_calls == 1
    assert analytic_calls == 1


def test_ordinary_forward_never_requests_analytic_cusp_data() -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    calls = 0
    original = cusp.analytic_evaluation

    def analytic_spy(batch: ElectronBatch):
        nonlocal calls
        calls += 1
        return original(batch)

    cusp.analytic_evaluation = analytic_spy
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=ConstantReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )
    model(ElectronBatch(positions=torch.ones(1, 1, 1, dtype=torch.float64)))
    assert calls == 0


def test_analytic_provider_binding_rejects_missing_or_duplicate_identity() -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    other = ElectronNucleusCusp(atoms)
    kwargs = dict(embedding=EmptyEncoder(), layers=[nn.Identity()], readout=ConstantReadout())

    with pytest.raises(ValueError, match="unique participating"):
        TPENWaveFunction(**kwargs, factors=[cusp, cusp], analytic_cusp_provider=cusp)
    with pytest.raises(ValueError, match="unique participating"):
        TPENWaveFunction(**kwargs, factors=[other], analytic_cusp_provider=cusp)
    equal_parameter_copy = ElectronNucleusCusp(atoms)
    with pytest.raises(ValueError, match="unique participating"):
        TPENWaveFunction(**kwargs, factors=[cusp], analytic_cusp_provider=equal_parameter_copy)

    model = TPENWaveFunction(**kwargs, factors=[cusp])
    with pytest.raises(ValueError, match="requires an analytic_cusp_provider"):
        model.factorized_local_energy_input(ElectronBatch(positions=torch.ones(1, 1, 1, dtype=torch.float64)))


def test_provider_index_binds_live_factor_and_rejects_bad_factor_sets() -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    other = ElectronNucleusCusp(atoms)
    kwargs = dict(embedding=EmptyEncoder(), layers=[nn.Identity()], readout=ConstantReadout())

    model = TPENWaveFunction(**kwargs, factors=[nn.Identity(), cusp], analytic_cusp_provider_index=1)
    # A copied factor could have equal parameters but must not satisfy the seam.
    assert model.analytic_cusp_provider is model.factors[1]

    with pytest.raises(ValueError, match="unique participating"):
        TPENWaveFunction(**kwargs, factors=[nn.Identity(), cusp, other], analytic_cusp_provider_index=1)
    # If the indexed domain guard were removed, the later uniqueness check
    # would raise ValueError instead; this pins the intended user-facing error.
    with pytest.raises(TypeError, match="^analytic_cusp_provider must be an ElectronNucleusCusp$"):
        TPENWaveFunction(**kwargs, factors=[nn.Identity()], analytic_cusp_provider_index=0)


def test_factorized_local_energy_input_typed_contracts_cover_both_components() -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=ConstantReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )
    value = model.factorized_local_energy_input(ElectronBatch(positions=torch.ones(1, 2, 1, dtype=torch.float64)))
    round_trip = value.permute(Permutation((1, 0))).permute(Permutation((1, 0)))
    close, metrics = value.compare(round_trip)
    assert close
    assert metrics["max_abs_error"] == pytest.approx(0.0)
    assert isinstance(value, FactorizedLocalEnergyInput)
