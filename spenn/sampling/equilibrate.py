"""Sampler warmup and equilibration helpers."""

from __future__ import annotations

from spenn.data.batch import Walkers


def warmup(model, sampler, walkers: Walkers, n_steps: int):
    """Run a warmup phase using the sampler."""

    return sampler.sample(model, walkers, n_steps)
