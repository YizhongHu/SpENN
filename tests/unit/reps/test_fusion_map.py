"""Tests for hard-coded M=2 fusion maps."""

from __future__ import annotations

import pytest
import torch

from spenn.data import FeatureDict, Par
from spenn.reps import FusionMap


def _features() -> FeatureDict:
    h = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
    s = torch.tensor(
        [[[[0.0, 4.0, 5.0], [4.0, 0.0, 6.0], [5.0, 6.0, 0.0]]]],
        dtype=torch.float64,
    ).unsqueeze(-1).unsqueeze(-1)
    a = torch.tensor(
        [[[[0.0, 7.0, 8.0], [-7.0, 0.0, 9.0], [-8.0, -9.0, 0.0]]]],
        dtype=torch.float64,
    ).unsqueeze(-1).unsqueeze(-1)
    return FeatureDict({Par("H"): h, Par("S"): s, Par("A"): a})


def test_fusion_map_emits_all_m2_products_and_valid_shapes() -> None:
    products = FusionMap()(_features())

    expected = {
        (Par("H"), Par("H"), Par("H")),
        (Par("S"), Par("H"), Par("H")),
        (Par("A"), Par("H"), Par("H")),
        (Par("S"), Par("H"), Par("S")),
        (Par("A"), Par("H"), Par("S")),
        (Par("S"), Par("H"), Par("A")),
        (Par("A"), Par("H"), Par("A")),
        (Par("S"), Par("S"), Par("H")),
        (Par("A"), Par("S"), Par("H")),
        (Par("S"), Par("A"), Par("H")),
        (Par("A"), Par("A"), Par("H")),
        (Par("S"), Par("S"), Par("S")),
        (Par("A"), Par("S"), Par("A")),
        (Par("A"), Par("A"), Par("S")),
        (Par("S"), Par("A"), Par("A")),
    }

    assert {(target, left, right) for target, left, right, _tensor in products.flat_items()} == expected
    products.validate(batch_size=1, n_electrons=3)


def test_hh_to_pair_has_expected_symmetry_and_assignments() -> None:
    products = FusionMap()(_features())
    symmetric = products.get(Par("S"), Par("H"), Par("H"))
    antisymmetric = products.get(Par("A"), Par("H"), Par("H"))

    assert symmetric.shape == (1, 1, 1, 3, 3, 3, 3, 1, 1)
    assert torch.allclose(symmetric[:, :, :, 0, 0], torch.zeros_like(symmetric[:, :, :, 0, 0]))
    assert torch.allclose(antisymmetric[:, :, :, 1, 1], torch.zeros_like(antisymmetric[:, :, :, 1, 1]))
    assert symmetric[0, 0, 0, 0, 1, 0, 1, 0, 0] == 1.0
    assert symmetric[0, 0, 0, 0, 1, 1, 0, 0, 0] == 1.0
    assert antisymmetric[0, 0, 0, 0, 1, 0, 1, 0, 0] == 1.0
    assert antisymmetric[0, 0, 0, 0, 1, 1, 0, 0, 0] == -1.0

    symmetric_target = symmetric[..., 0, 0].sum(dim=(-1, -2))
    antisymmetric_target = antisymmetric[..., 0, 0].sum(dim=(-1, -2))
    assert torch.allclose(symmetric_target, symmetric_target.transpose(3, 4))
    assert torch.allclose(antisymmetric_target, -antisymmetric_target.transpose(3, 4))


def test_hh_to_h_is_only_same_node_union() -> None:
    block = FusionMap().fuse_pair(_features(), Par("H"), Par("H"), Par("H"))

    assert block.shape == (1, 1, 1, 3, 3, 3, 1, 1)
    assert block[0, 0, 0, 0, 0, 0, 0, 0] == 1.0
    assert block[0, 0, 0, 1, 1, 1, 0, 0] == 4.0
    assert block[0, 0, 0, 1, 1, 2, 0, 0] == 0.0


def test_node_pair_and_pair_pair_products_have_expected_entries() -> None:
    products = FusionMap()(_features())
    hs_to_s = products.get(Par("S"), Par("H"), Par("S"))
    hs_to_a = products.get(Par("A"), Par("H"), Par("S"))
    sa_to_a = products.get(Par("A"), Par("S"), Par("A"))
    aa_to_s = products.get(Par("S"), Par("A"), Par("A"))

    assert hs_to_s[0, 0, 0, 0, 1, 0, 0, 1, 0, 0] == 2.0
    assert hs_to_s[0, 0, 0, 0, 1, 1, 1, 0, 0, 0] == 4.0
    assert hs_to_a[0, 0, 0, 0, 1, 0, 0, 1, 0, 0] == 2.0
    assert hs_to_a[0, 0, 0, 0, 1, 1, 1, 0, 0, 0] == -4.0
    assert sa_to_a[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0] == 14.0
    assert sa_to_a[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0] == 14.0
    assert aa_to_s[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0] == 24.5
    assert aa_to_s[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0] == 24.5


def test_fuse_pair_rejects_missing_sources_and_unsupported_products() -> None:
    fusion = FusionMap()
    features = _features()

    with pytest.raises(KeyError, match="Missing source"):
        fusion.fuse_pair(FeatureDict({Par("H"): features.get(Par("H"))}), Par("H"), Par("S"), Par("S"))
    with pytest.raises(ValueError, match="Unsupported"):
        fusion.fuse_pair(features, Par("H"), Par("S"), Par("H"))
