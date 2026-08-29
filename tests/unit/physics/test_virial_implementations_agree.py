"""Keep the TPEN and independent experiment-fork virial formulas aligned."""

from __future__ import annotations

import pytest

from experiments.toolkit.virial import derive_virial_metrics as experiment_virial
from tpen.physics.virial import derive_virial_metrics as tpen_virial


@pytest.mark.parametrize(
    "values",
    [(1.0, 2.0, 0.5), (0.0, 0.0, 0.0), (None, 2.0, 0.5), (-1.25, 0.75, 3.0)],
)
def test_tpen_and_experiment_virial_formulas_agree(values):
    assert tpen_virial(*values) == experiment_virial(*values)
