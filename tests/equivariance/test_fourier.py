"""Runtime equivariance tests for Fourier transforms."""

from __future__ import annotations

import torch

from spenn.data.irrep import IrrepFeature
from spenn.data.partition import Partition
from spenn.data.real import RealInteraction, zero_block
from spenn.reps import FourierTransform, InverseFourierTransform
from tests.helpers.equivariance import assert_equivariant_all


def test_fourier_transform_passes_forced_runtime_equivariance_check() -> None:
    real = RealInteraction(
        [
            zero_block(paths=0, dtype=torch.float64),
            torch.arange(1 * 2 * 1 * 3, dtype=torch.float64).reshape(1, 2, 1, 3),
            torch.arange(1 * 2 * 1 * 3 * 3, dtype=torch.float64).reshape(1, 2, 1, 3, 3),
        ]
    )
    transform = FourierTransform()

    output = transform(real)

    assert output.validate() is output
    assert_equivariant_all(transform, real)


def test_inverse_fourier_transform_passes_forced_runtime_equivariance_check() -> None:
    feature = IrrepFeature(
        {
            Partition((2,)): torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3, 1, 1),
            Partition((1, 1)): torch.ones(1, 2, 3, 3, 1, 1, dtype=torch.float64),
        }
    )
    transform = InverseFourierTransform()

    output = transform(feature)

    assert output.validate() is output
    assert_equivariant_all(transform, feature)
