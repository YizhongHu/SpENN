"""Tiny real-model builders for Hooke pair smoke tests.

Single source of truth: everything is instantiated from the smoke training
fixture ``tests/integration/artifacts/hooke/pair_train.yaml`` (a copy of the experiments
config), so unit tests exercise the exact model/sampler the integration run uses.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from spenn.data.batch import ElectronBatch
from spenn.nn import SpENNWaveFunction
from spenn.sampling.metropolis import MetropolisSampler

PAIR_TRAIN_CONFIG = Path(__file__).resolve().parents[1] / "integration" / "artifacts" / "hooke" / "pair_train.yaml"


def _config() -> OmegaConf:
    return OmegaConf.load(PAIR_TRAIN_CONFIG)


def build_tiny_spenn() -> SpENNWaveFunction:
    """Instantiate the tiny `SpENNWaveFunction` from the smoke fixture config."""

    cfg = _config()
    model = instantiate(cfg.model)
    dtype = getattr(torch, str(cfg.runtime.dtype))
    device = torch.device(str(cfg.runtime.device))
    return model.to(device=device, dtype=dtype)


def build_tiny_sampler() -> MetropolisSampler:
    """Instantiate the fixed-spin Metropolis sampler from the smoke fixture config."""

    return instantiate(_config().sampler)


def build_tiny_hamiltonian_terms() -> dict:
    """Instantiate the named Hooke Hamiltonian terms from the smoke fixture config."""

    return dict(instantiate(_config().hamiltonian_terms))


def tiny_pair_batch(n_walkers: int = 4) -> ElectronBatch:
    """Return a tiny 2-electron batch with fixed (up, down) spins."""

    generator = torch.Generator().manual_seed(0)
    positions = torch.randn(n_walkers, 2, 3, generator=generator, dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0]] * n_walkers, dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)
