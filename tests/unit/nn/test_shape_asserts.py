"""Unit tests for neural-network shape assertions."""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch
from spenn.nn import ElectronElectronCusp


def test_cusp_shape_asserts_preserve_batch_shape() -> None:
    positions = torch.zeros(4, 3, 2, dtype=torch.float64)
    output = ElectronElectronCusp()(ElectronBatch(positions))

    assert output.shape == (4,)
