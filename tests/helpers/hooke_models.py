"""Tiny real-model builders for Hooke pair smoke tests.

Builds an actual `SpENNWaveFunction` (embedding + Pfaffian readout, no layers)
sized for a 2-electron CPU smoke run. Kept under ``tests/`` -- this is a test
fixture, not core API.
"""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch
from spenn.nn import Embedding, SpENNWaveFunction
from spenn.nn.readout import PfaffianReadout
from spenn.sampling.metropolis import MetropolisSampler


def build_tiny_spenn() -> SpENNWaveFunction:
    """Return a tiny, equivariant two-electron `SpENNWaveFunction`."""

    return SpENNWaveFunction(
        embedding=Embedding(
            max_order=2,
            out_channels=4,
            hidden_channels=8,
            num_hidden_layers=1,
            include_spins=True,
        ),
        layers=[],
        readout=PfaffianReadout(allow_odd_electron_bordered=True),
    )


def build_tiny_sampler(*, n_walkers: int = 4, burn_in: int = 1, n_steps: int = 1) -> MetropolisSampler:
    """Return a tiny fixed-spin (1 up, 1 down) Metropolis sampler."""

    return MetropolisSampler(
        n_walkers=n_walkers,
        burn_in=burn_in,
        n_steps=n_steps,
        proposal_scale=0.5,
        seed=0,
        n_electrons=2,
        spatial_dim=3,
        n_up=1,
        n_down=1,
        dtype=torch.float64,
    )


def tiny_pair_batch(n_walkers: int = 4) -> ElectronBatch:
    """Return a tiny 2-electron batch with fixed (up, down) spins."""

    generator = torch.Generator().manual_seed(0)
    positions = torch.randn(n_walkers, 2, 3, generator=generator, dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0]] * n_walkers, dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)
