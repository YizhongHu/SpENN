"""Tests for real-feature normalization modules."""

from __future__ import annotations

import pytest
import torch

from spenn.data.real import RealFeature, RealUpdate, zero_block
from spenn.nn import RMSNorm
from tests.helpers.equivariance import assert_equivariant_all


def test_rms_norm_is_particle_equivariant() -> None:
    feature = RealFeature(
        [
            zero_block(batch_size=3, dtype=torch.float64),
            torch.randn(3, 4, 4, dtype=torch.float64),
            torch.randn(3, 4, 4, 4, dtype=torch.float64),
        ]
    )

    assert_equivariant_all(RMSNorm(eps=1.0e-8), feature)


def test_rms_norm_preserves_concrete_real_state_type() -> None:
    update = RealUpdate(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.tensor([[[3.0, 4.0]], [[5.0, 12.0]]], dtype=torch.float64),
        ]
    )

    normalized = RMSNorm(eps=1.0e-8)(update)

    assert isinstance(normalized, RealUpdate)
    assert normalized.blocks[1].shape == update.blocks[1].shape


def test_rms_norm_rejects_nonpositive_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        RMSNorm(eps=0.0)
