"""Tests for Hooke final-eval physics diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

import spenn.diagnostics.base as diagnostics_base
from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.diagnostics import (
    EvaluationContext,
    HookePairCenterOfMassProbe,
    HookePairDistanceProbe,
    PositionExchangeDiagnostic,
    RotationDiagnostic,
    TraceEquivarianceDiagnostic,
)
from spenn.equivariance import EquivariantMap
from spenn.physics.hamiltonian import LocalEnergyResult, local_energy
from spenn.physics.hooke import HookeSingletExact
from spenn.physics.kinetic import KineticEnergy
from spenn.physics.potential import ElectronElectronInteraction, HarmonicTrap


def _context(tmp_path: Path, model=None) -> EvaluationContext:
    model = HookeSingletExact() if model is None else model
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions, spins=torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64))
    terms = {
        "kinetic": KineticEnergy(),
        "harmonic_trap": HarmonicTrap(omega=0.5),
        "electron_electron": ElectronElectronInteraction(),
    }
    result = local_energy(terms, model, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    with torch.no_grad():
        output = model(batch)
    return EvaluationContext(
        model=model,
        batch=batch,
        wavefunction_output=output,
        local_energy=result.total,
        local_energy_terms=result.terms,
        sampler_stats={},
        hamiltonian_terms=terms,
        run_dir=tmp_path,
    )


def test_hooke_pair_distance_probe_writes_required_and_exact_columns(tmp_path: Path) -> None:
    context = _context(tmp_path)

    metrics = HookePairDistanceProbe(
        reference_energy=2.0,
        r12_min=0.1,
        r12_max=1.0,
        n_points=3,
        n_directions=1,
        center_of_mass_radii=[0.0],
    ).evaluate(context)

    path = tmp_path / "diagnostics" / "pair_distance_probe" / "probe.csv"
    assert path.exists()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for column in (
        "pair_distance",
        "model_logabs",
        "model_relative_abs_psi",
        "model_local_energy",
        "exact_logabs",
        "exact_relative_abs_psi",
        "aligned_logabs_error",
    ):
        assert column in rows[0]
    assert metrics["probe_pair_distance/nonfinite_count"] == 0


def test_hooke_pair_distance_probe_chunks_local_energy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path)
    calls = []

    def fake_local_energy(terms, wavefunction, batch, *, return_terms=False):
        calls.append(batch.batch_size)
        values = torch.arange(batch.batch_size, device=batch.device, dtype=batch.dtype)
        return LocalEnergyResult(
            total=values,
            terms={
                "kinetic": values,
                "harmonic_trap": values,
                "electron_electron": values,
            },
        )

    monkeypatch.setattr(diagnostics_base, "local_energy", fake_local_energy)

    HookePairDistanceProbe(
        r12_min=0.1,
        r12_max=1.0,
        n_points=5,
        n_directions=1,
        center_of_mass_radii=[0.0],
        local_energy_chunk_size=2,
    ).evaluate(context)

    assert calls == [2, 2, 1, 2, 2, 1]


def test_hooke_center_of_mass_probe_writes_artifact_and_index(tmp_path: Path) -> None:
    context = _context(tmp_path)

    metrics = HookePairCenterOfMassProbe(
        reference_energy=2.0,
        com_radius_min=0.0,
        com_radius_max=1.0,
        n_points=3,
        n_directions=1,
    ).evaluate(context)

    assert metrics["probe_center_of_mass/nonfinite_count"] == 0
    index = json.loads((tmp_path / "diagnostics" / "index.json").read_text())
    names = {entry["name"] for entry in index["artifacts"]}
    assert "center_of_mass_probe" in names


def test_position_exchange_diagnostic_checks_positions_only(tmp_path: Path) -> None:
    context = _context(tmp_path)

    metrics = PositionExchangeDiagnostic(max_samples=2).evaluate(context)

    assert metrics["checks/exchange/sign_failure_count"] == 0
    assert metrics["checks/exchange/logabs_max_abs_error"] == pytest.approx(0.0)
    assert (tmp_path / "diagnostics" / "exchange" / "trace.jsonl").exists()


def test_rotation_diagnostic_writes_trace_and_scalar_metrics(tmp_path: Path) -> None:
    context = _context(tmp_path)

    metrics = RotationDiagnostic(max_samples=1, n_rotations=2).evaluate(context)

    assert metrics["checks/rotation/nonfinite_count"] == 0
    assert metrics["checks/rotation/logabs_max_abs_error"] == pytest.approx(0.0, abs=1.0e-10)
    assert (tmp_path / "diagnostics" / "rotation" / "trace.jsonl").exists()


def test_rotation_diagnostic_chunks_rotated_local_energy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path)
    calls = []

    def fake_local_energy(terms, wavefunction, batch, *, return_terms=False):
        calls.append(batch.batch_size)
        return torch.arange(batch.batch_size, device=batch.device, dtype=batch.dtype)

    monkeypatch.setattr(diagnostics_base, "local_energy", fake_local_energy)

    RotationDiagnostic(max_samples=2, n_rotations=3, local_energy_chunk_size=2).evaluate(context)

    assert calls == [2, 2, 2]


class _FermionicToy(EquivariantMap):
    def __init__(self) -> None:
        super().__init__(trace_name="toy")

    def forward_impl(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        logabs = flat.positions.square().sum(dim=(1, 2)) * 0.0
        sign = torch.sign(flat.positions[:, 0, 0] - flat.positions[:, 1, 0])
        return WavefunctionOutput(logabs=logabs, sign=sign)


def test_trace_equivariance_diagnostic_uses_checker_without_callback(tmp_path: Path) -> None:
    context = _context(tmp_path, model=_FermionicToy())
    context = EvaluationContext(**{**context.__dict__, "local_energy": torch.zeros(2, dtype=torch.float64), "local_energy_terms": {}})

    metrics = TraceEquivarianceDiagnostic(max_samples=2, max_permutations=1).evaluate(context)

    assert metrics["checks/trace_equivariance/failure_count"] == 0
    assert (tmp_path / "diagnostics" / "trace_equivariance" / "trace.jsonl").exists()
