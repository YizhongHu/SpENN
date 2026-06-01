"""Inactive broad phase-1 smoke tests.

This module is intentionally kept under a non-``test_*.py`` filename so pytest
does not collect it. The focused unit and integration tests own these behaviors
while the top-level config cleanup is deferred.
"""

from __future__ import annotations

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import nn

from spenn.data import FeatureDict, Par
from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.nn.readout.pfaffian import PfaffianReadout
from spenn.nn.wavefunction import SpENNWavefunction
from spenn.physics.hamiltonian import ElectronicHamiltonian
from spenn.physics.systems import ElectronicSystem
from tests.helpers import ROOT


class PairDifferenceEncoder(nn.Module):
    def forward(self, batch: ElectronBatch) -> FeatureDict:
        x = batch.positions[..., 0]
        carrier = (x.unsqueeze(2) - x.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        gate = torch.ones_like(carrier)
        one_body = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return FeatureDict({Par("H"): one_body, Par("S"): gate, Par("A"): carrier})


class GaussianTensorModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float64)

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        return -self.alpha * batch.positions.square().sum(dim=(1, 2))


class TrainableGaussianOutputModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _set_unit_readout_weights(readout: PfaffianReadout) -> None:
    with torch.no_grad():
        carrier = readout.carrier_projections[0]
        gate = readout.gate_projections[0]
        carrier.weight.fill_(1.0)
        gate.weight.zero_()
        gate.bias.fill_(1.0)


def test_hydra_phase1_cpu_config_smoke_runs_forward_loss_and_optimizer_instantiation() -> None:
    config_dir = ROOT / "configs"
    overrides = [
        "device=cpu",
        "dtype=float64",
        "model.encoder.channels=[0,2,2]",
        "model.spechtmp.layers.0.message_head.channels=[0,2,2]",
        "model.spechtmp.layers.0.update_head.channels=[0,2,2]",
        "model.spechtmp.layers.1.message_head.channels=[0,2,2]",
        "model.spechtmp.layers.1.update_head.channels=[0,2,2]",
        "model.spechtmp.layers.2.message_head.channels=[0,2,2]",
        "model.spechtmp.layers.2.update_head.channels=[0,2,2]",
        "model.spechtmp.layers.3.message_head.channels=[0,2,2]",
        "model.spechtmp.layers.3.update_head.channels=[0,2,2]",
        "sampler.n_walkers=4",
        "sampler.steps_per_iter=1",
        "sampler.warmup_steps=0",
        "trainer.max_steps=1",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)

    dtype = getattr(torch, str(cfg.dtype))
    torch.manual_seed(int(cfg.seed))
    system = instantiate(cfg.system).to(device="cpu", dtype=dtype)
    model = instantiate(cfg.model).to(device="cpu", dtype=dtype)
    hamiltonian = instantiate(cfg.hamiltonian, _partial_=True)(system=system)
    sampler = instantiate(cfg.sampler)
    loss_fn = instantiate(cfg.loss)
    walkers = sampler.initialize(system=system, device="cpu")
    batch = ElectronBatch(positions=walkers.positions, system=walkers.aux.get("system"))

    model(batch)
    optimizer = instantiate(cfg.optimizer, _partial_=True)(params=model.parameters())
    loss, metrics = loss_fn(model, hamiltonian, batch)

    assert optimizer.param_groups
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["energy"])
    assert torch.isfinite(metrics["variance"])


def test_end_to_end_wavefunction_path_returns_valid_signed_log_output_and_kernel_aux() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    encoder = PairDifferenceEncoder()
    readout = PfaffianReadout()
    readout.build_skew_kernel(encoder(batch), batch)
    _set_unit_readout_weights(readout)
    model = SpENNWavefunction(encoder=encoder, spechtmp=nn.Identity(), readout=readout)

    output = model(batch)

    assert output.logabs.shape == (2,)
    assert output.sign.shape == (2,)
    assert torch.all(torch.isfinite(output.logabs))
    assert torch.all(torch.isin(output.sign, torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)))
    assert "K" in output.aux
    assert torch.allclose(output.aux["K"] + output.aux["K"].transpose(-1, -2), torch.zeros_like(output.aux["K"]))


def test_local_energy_path_accepts_tensor_and_wavefunction_output_models() -> None:
    system = ElectronicSystem(n_electrons=2, spatial_dim=2, harmonic_omega=1.0, dtype=torch.float64)
    batch = ElectronBatch(
        positions=torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[-1.0, 1.0], [2.0, -0.5]]], dtype=torch.float64),
        system=system,
    )
    hamiltonian = ElectronicHamiltonian(system=system)
    tensor_energy = hamiltonian.local_energy(GaussianTensorModel(alpha=0.5), batch)
    output_model = TrainableGaussianOutputModel(alpha=0.25)
    output_energy = hamiltonian.local_energy(output_model, batch)
    output_energy.mean().backward()

    assert tensor_energy.shape == (2,)
    assert output_energy.shape == (2,)
    assert torch.all(torch.isfinite(tensor_energy))
    assert torch.all(torch.isfinite(output_energy))
    assert output_model.alpha.grad is not None
    assert torch.isfinite(output_model.alpha.grad)
