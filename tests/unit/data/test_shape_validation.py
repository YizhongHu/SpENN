"""Unit tests for data-container shape validation."""

from __future__ import annotations

import torch

from spenn.data import BranchDict, FeatureDict, MessageDict, Par, TensorProductDict
from spenn.utils.tensor_utils import pairwise_distances


def test_container_validation_asserts_still_allow_valid_active_shapes() -> None:
    features = FeatureDict({Par("H"): torch.zeros(1, 2, 3, 1, 1)})
    messages = MessageDict({Par("S"): torch.zeros(1, 2, 3, 3, 1, 1)})
    products = TensorProductDict({Par("S"): {Par("H"): {Par("H"): torch.zeros(1, 2, 1, 3, 3, 3, 3, 1, 1)}}})
    branches = BranchDict({Par("S"): {Par("S"): torch.zeros(1, 2, 1, 3, 3, 3, 3, 1, 1)}})

    features.validate(batch_size=1, n_electrons=3)
    messages.validate(batch_size=1, n_electrons=3)
    products.validate(batch_size=1, n_electrons=3)
    branches.validate(batch_size=1, n_electrons=3)


def test_pairwise_distances_preserve_shape_dtype_and_smooth_diagonal() -> None:
    positions = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]], dtype=torch.float64)

    exact = pairwise_distances(positions, eps=0.0)
    smoothed = pairwise_distances(positions, eps=1.0e-3)

    assert exact.shape == (1, 2, 2, 1)
    assert exact.dtype == torch.float64
    assert torch.equal(exact[..., 0], torch.tensor([[[0.0, 5.0], [5.0, 0.0]]], dtype=torch.float64))
    assert torch.all(smoothed.diagonal(dim1=1, dim2=2) > 0.0)
    assert torch.allclose(smoothed[:, 0, 1, 0], torch.sqrt(torch.tensor([25.0 + 1.0e-6], dtype=torch.float64)))


def test_pairwise_distances_reject_wrong_rank() -> None:
    try:
        pairwise_distances(torch.zeros(2, 3))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected invalid-rank positions to raise")
