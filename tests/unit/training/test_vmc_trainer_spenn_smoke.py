"""Smoke test: one VMC trainer step over the real tiny SpENN stack."""

from __future__ import annotations

import math

import torch

from spenn.artifacts import RunContext
from spenn.physics.kinetic import KineticEnergy
from spenn.physics.potential import ElectronElectronInteraction, HarmonicTrap
from spenn.training.trainer import VMCTrainer
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn


class _StubContext(RunContext):
    """Minimal RunContext subclass: satisfies typing, logs to a list."""

    def __init__(self) -> None:
        self.loggers = []
        self.records: list[tuple[str, dict]] = []

    def log(self, metrics, *, step=None, namespace="run", event=None) -> None:
        self.records.append((namespace, dict(metrics)))


_FORBIDDEN_METRICS = {
    "reference_energy",
    "energy_error",
    "abs_energy_error",
    "exact_energy",
    "expected_energy",
}


def test_one_vmc_step_is_finite_and_vmc_native() -> None:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=True)

    state = trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None: None,
    )

    assert math.isfinite(float(state.loss))
    assert torch.isfinite(state.local_energy).all()
    # Native VMC metrics only -- no reference/exact comparison leaks in.
    assert _FORBIDDEN_METRICS.isdisjoint(state.metrics)
    for key in ("energy_mean", "loss", "grad_norm"):
        assert key in state.metrics
