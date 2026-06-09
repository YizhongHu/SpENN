"""Smoke test: a tiny real SpENNWaveFunction maps a pair batch to an output."""

from __future__ import annotations

import torch

from spenn.data.batch import WavefunctionOutput
from tests.helpers.hooke_models import build_tiny_spenn, tiny_pair_batch


def test_forward_returns_finite_wavefunction_output() -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=4)

    output = model(batch)

    assert isinstance(output, WavefunctionOutput)
    assert output.logabs.shape == (4,)
    assert output.sign.shape == (4,)
    assert torch.isfinite(output.logabs).all()
    assert torch.isfinite(output.sign).all()
    assert torch.all((output.sign == 1) | (output.sign == -1) | (output.sign == 0))
