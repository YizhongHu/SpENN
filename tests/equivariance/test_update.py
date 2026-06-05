"""Tests for real-feature update equivariance."""

from __future__ import annotations

import torch

from spenn.data import RealFeature, RealUpdate, zero_block
from spenn.nn import Update


def test_update_passes_runtime_equivariance_check() -> None:
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
        ]
    )
    update = RealUpdate(
        [
            zero_block(dtype=torch.float64),
            torch.ones(1, 2, 3, dtype=torch.float64),
        ]
    )
    module = Update(equivariance_check=True, check_probability=1.0)

    output = module(feature, update)

    torch.testing.assert_close(output.blocks[1], feature.blocks[1] + update.blocks[1])
