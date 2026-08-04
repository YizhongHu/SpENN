"""T7 cusp-contract pins for the TPEN migration (MIG-TPEN-000 §5).

Pins the spin handling of the electron-electron cusp so the future ``Cusp``
module cannot silently change slopes:

- opposite-spin pair slope 1/2 and same-spin pair slope 1/4 at coalescence;
- the Hooke reference pair (n_up=1, n_down=1) uses the opposite-spin slope;
- ``spins=None`` is a documented default (same-spin coefficient 1/4), not an
  accident — if the default ever changes this test must be updated together
  with the spec.

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch
from spenn.nn import Cusp

_DTYPE = torch.float64


def _pair_batch(separation: float, spins: tuple[int, int] | None) -> ElectronBatch:
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]], dtype=_DTYPE
    )
    spin_tensor = None if spins is None else torch.tensor([list(spins)], dtype=_DTYPE)
    return ElectronBatch(positions=positions, spins=spin_tensor)


def _slope_at_coalescence(spins: tuple[int, int] | None) -> float:
    # u(r) = a*r / (1 + b*r), so u(r)/r -> a as r -> 0. A small separation
    # recovers the analytic slope to first order.
    separation = 1.0e-9
    cusp = Cusp().to(dtype=_DTYPE)
    value = cusp(_pair_batch(separation, spins))
    return float(value.item() / separation)


def test_opposite_spin_slope_is_one_half() -> None:
    # T7: the Hooke reference pair is a singlet (n_up=1, n_down=1); its
    # coalescence slope must be the opposite-spin 1/2, not the same-spin 1/4.
    assert abs(_slope_at_coalescence((1, -1)) - 0.5) < 1.0e-6


def test_same_spin_slope_is_one_quarter() -> None:
    assert abs(_slope_at_coalescence((1, 1)) - 0.25) < 1.0e-6


def test_spinless_default_is_documented_same_spin_quarter() -> None:
    # Pinned contract: spins=None silently uses the same-spin coefficient.
    # This is the documented default; a Hooke config regression that drops
    # spins would halve the cusp slope, which is why the spins-present test
    # above exists. Changing this default requires a spec update (T7).
    assert abs(_slope_at_coalescence(None) - 0.25) < 1.0e-6


def test_cusp_value_is_permutation_invariant() -> None:
    # The envelope stack must be symmetric under particle exchange so the
    # readout keeps ownership of antisymmetry (MIG-TPEN-000 §2.5).
    cusp = Cusp().to(dtype=_DTYPE)
    positions = torch.tensor(
        [[[0.1, -0.2, 0.3], [0.7, 0.4, -0.5]]], dtype=_DTYPE
    )
    spins = torch.tensor([[1.0, -1.0]], dtype=_DTYPE)
    swapped = ElectronBatch(positions=positions.flip(1), spins=spins.flip(1))
    original = ElectronBatch(positions=positions, spins=spins)
    torch.testing.assert_close(cusp(original), cusp(swapped))
