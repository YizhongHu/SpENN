"""Close the remaining executable oracle gates T2 (full model) and T11.

MIG-TPEN-000 §5:

- T2 requires full-model antisymmetry pinned exhaustively at small n:
  ``logabs(pi x) == logabs(x)`` and ``sign(pi x) == sgn(pi) * sign(x)`` for
  every permutation. Layer- and readout-level pins exist; this adds the
  wavefunction-level gate, including the odd-n Pfaffian padding path.
- T11 requires that the forward/training path performs no path-metadata
  generation: model code reads cached metadata only. The generator is patched
  to raise so any regeneration attempt fails loudly.

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import pytest
import torch

import tpen.data.paths as paths_module
from tpen.data.batch import ElectronBatch
from tpen.data.permutation import all_permutations
from tests.helpers.hooke_models import build_tiny_spenn, tiny_pair_batch

_DTYPE = torch.float64


def _batch(n_electrons: int, *, n_walkers: int = 3, seed: int = 7) -> ElectronBatch:
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randn(n_walkers, n_electrons, 3, generator=generator, dtype=_DTYPE)
    # Fill spins up-first; permutations below act on positions and spins
    # together, which is the full particle-label exchange antisymmetry.
    up = (n_electrons + 1) // 2
    spins = torch.tensor([[1.0] * up + [-1.0] * (n_electrons - up)] * n_walkers, dtype=_DTYPE)
    return ElectronBatch(positions=positions, spins=spins)


@pytest.mark.parametrize("n_electrons", [2, 3])
def test_full_model_is_antisymmetric_for_all_permutations(n_electrons: int) -> None:
    # T2 at the wavefunction level: exhaustive over S_n, exercising both the
    # even-n skew Pfaffian and the odd-n padding readout path.
    model = build_tiny_spenn()
    batch = _batch(n_electrons)
    output = model(batch)

    for permutation in all_permutations(n_electrons):
        permuted = model(batch.permute(permutation))
        torch.testing.assert_close(permuted.logabs, output.logabs, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(permuted.sign, output.sign * float(permutation.sign))


def test_forward_and_backward_never_regenerate_path_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    # T11: metadata is deterministic and cached; the model must only read it.
    # Construction happens BEFORE patching (it loads the checked-in cache);
    # any generation attempt during forward/backward then raises.
    model = build_tiny_spenn()
    batch = tiny_pair_batch()

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("path metadata must never be regenerated inside forward/training")

    monkeypatch.setattr(paths_module, "generate_virtual_paths", _forbidden)
    if hasattr(paths_module.PathMetadata, "save"):
        monkeypatch.setattr(paths_module.PathMetadata, "save", _forbidden)

    output = model(batch)
    loss = output.logabs.square().sum()
    loss.backward()
    assert torch.isfinite(output.logabs).all()
