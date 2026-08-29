"""Tests for the optional Numdifftools kinetic oracle."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.permutation import reversal_permutation
from tpen.physics.kinetic import kinetic_energy_from_logabs

nd = pytest.importorskip(
    "numdifftools",
    reason="finite-difference oracle tests require `uv sync --extra finite-difference`",
)

from tests.helpers.finite_difference_kinetic import finite_difference_kinetic


class GaussianModel(nn.Module):
    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        self.alpha = alpha
        self.seen: list[ElectronBatch] = []

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        self.seen.append(batch)
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class SignedNodalModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        signed_factor = batch.positions[:, 0, 0]
        logabs = torch.log(signed_factor.abs()) - 0.2 * batch.positions.square().sum(dim=(1, 2))
        sign = torch.sign(signed_factor)
        logabs = torch.where(sign == 0, torch.full_like(logabs, -torch.inf), logabs)
        return WavefunctionOutput(logabs=logabs, sign=sign)


class NonfiniteProbeModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        center = torch.full((batch.batch_size,), 0.4, dtype=batch.positions.dtype)
        is_probe = batch.positions[:, 0, 0] != center
        logabs = torch.where(is_probe, torch.full_like(center, torch.inf), torch.zeros_like(center))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_numdifftools_full_output_contract_is_pinned() -> None:
    assert nd.__version__ == "0.10.1"
    hessdiag = nd.Hessdiag(lambda x: np.sum(x**2), full_output=True)
    result = hessdiag(np.array([1.0, 2.0], dtype=np.float64))

    assert type(result).__name__ == "EstimateResult"
    assert type(result)._fields == ("estimate", "error_estimate", "final_step", "best_index")
    assert np.asarray(result.estimate).shape == (2,)
    assert np.asarray(result.error_estimate).shape == (2,)
    assert np.asarray(result.final_step).shape == (2,)
    assert np.asarray(result.best_index).shape == (2,)
    plain = nd.Hessdiag(lambda x: np.sum(x**2))(np.array([1.0, 2.0], dtype=np.float64))
    assert type(plain) is np.ndarray
    assert plain.shape == (2,)
    assert np.allclose(result.estimate, [2.0, 2.0])


def test_gaussian_oracle_returns_hessian_kinetic_and_diagnostics() -> None:
    positions = torch.tensor([[[1.0, 2.0], [0.5, -1.0]]], dtype=torch.float64)
    result = finite_difference_kinetic(GaussianModel(), ElectronBatch(positions=positions), tolerance=1.0e-6)

    expected_diagonal = torch.full_like(positions, -0.6)
    assert result.center.logabs.shape == (1,)
    assert set(result.statuses) == {"center_node", "nonfinite_probe", "exceeded_tolerance"}
    assert result.hessian_diagonal.shape == (1, 2, 2)
    assert torch.allclose(result.hessian_diagonal, expected_diagonal, atol=1.0e-6)
    assert torch.allclose(result.total_kinetic, torch.tensor([1.2], dtype=torch.float64), atol=1.0e-6)
    assert torch.allclose(result.per_electron_kinetic.sum(dim=1), result.total_kinetic, atol=1.0e-12)
    assert torch.all(torch.isfinite(result.error_estimate))
    assert torch.all(torch.isfinite(result.final_step))
    assert not bool(result.exceeded_tolerance.any())


def test_oracle_agrees_with_current_autodiff_kinetic() -> None:
    positions = torch.tensor(
        [[[1.0, 2.0], [0.5, -1.0]], [[-1.5, 0.25], [2.0, 0.0]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    model = GaussianModel()
    oracle = finite_difference_kinetic(model, batch, tolerance=1.0e-6)
    autodiff = kinetic_energy_from_logabs(model, batch)

    budget = oracle.error_estimate.sum(dim=(1, 2))
    assert torch.allclose(oracle.total_kinetic, autodiff, atol=1.0e-6)
    assert torch.all(torch.abs(oracle.total_kinetic - autodiff) <= budget + 1.0e-6)


def test_signed_amplitude_is_differentiated_not_logabs() -> None:
    positions = torch.tensor([[[0.4], [0.7]]], dtype=torch.float64)
    result = finite_difference_kinetic(SignedNodalModel(), ElectronBatch(positions=positions), tolerance=1.0e-5)

    # At q0, psi is x0*exp(-0.2*sum(q**2)); this value changes sign under
    # probes, so differentiating logabs would produce the wrong curvature.
    assert torch.all(torch.isfinite(result.hessian_diagonal))
    assert not bool(result.nonfinite_probe.any())
    assert result.hessian_diagonal[0, 0, 0].item() == pytest.approx(-1.1744, abs=1.0e-5)


def test_center_node_and_nonfinite_probe_statuses_are_explicit() -> None:
    batch = ElectronBatch(positions=torch.tensor([[[0.0]], [[0.4]]], dtype=torch.float64))
    result = finite_difference_kinetic(SignedNodalModel(), batch, tolerance=1.0e-6)

    assert torch.equal(result.center_node, torch.tensor([True, False]))
    assert torch.isnan(result.total_kinetic[0])
    assert not bool(result.nonfinite_probe.any())


def test_nonfinite_probe_status_is_explicit() -> None:
    batch = ElectronBatch(positions=torch.tensor([[[0.4]]], dtype=torch.float64))
    result = finite_difference_kinetic(NonfiniteProbeModel(), batch)

    assert bool(result.nonfinite_probe[0])
    assert torch.isnan(result.total_kinetic[0])


def test_probe_preserves_typed_batch_metadata_and_cpu_float64() -> None:
    system = type("System", (), {"n_electrons": 2})()
    batch = ElectronBatch(
        positions=torch.tensor([[[0.4], [0.7]]], dtype=torch.float32),
        system=system,
        nuclear_positions=torch.tensor([[1.0]], dtype=torch.float32),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float32),
        spins=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        aux={"tag": "preserve"},
    )
    model = GaussianModel()
    result = finite_difference_kinetic(model, batch)

    assert result.center.logabs.dtype is torch.float64
    for seen in model.seen:
        assert seen.positions.device.type == "cpu"
        assert seen.positions.dtype is torch.float64
        assert seen.system is system
        assert torch.equal(seen.nuclear_positions, batch.nuclear_positions.to(torch.float64))
        assert torch.equal(seen.nuclear_charges, batch.nuclear_charges.to(torch.float64))
        assert torch.equal(seen.spins, batch.spins.to(torch.float64))
        assert seen.aux == batch.aux


def test_permutation_permuted_batch_permuted_attribution() -> None:
    positions = torch.tensor([[[0.4], [0.7]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    permutation = reversal_permutation(2)
    permuted = batch.permute(permutation)

    first = finite_difference_kinetic(GaussianModel(), batch)
    second = finite_difference_kinetic(GaussianModel(), permuted)

    assert torch.allclose(first.total_kinetic, second.total_kinetic, atol=1.0e-6)
    assert torch.allclose(first.per_electron_kinetic, second.per_electron_kinetic[:, [1, 0]], atol=1.0e-6)


def test_exceeded_tolerance_status_is_reported() -> None:
    batch = ElectronBatch(positions=torch.tensor([[[0.4], [0.7]]], dtype=torch.float64))
    result = finite_difference_kinetic(GaussianModel(), batch, tolerance=0.0)

    assert bool(result.exceeded_tolerance.all())
