"""Tests for zero-based permutation helpers."""

from __future__ import annotations

import pytest

from spenn.data.permutation import Permutation


def test_identity_fixes_indices_and_tuples() -> None:
    permutation = Permutation.identity(4)

    assert permutation.image == (0, 1, 2, 3)
    assert permutation.apply_index(2) == 2
    assert permutation.apply_tuple((3, 0, 1)) == (3, 0, 1)


def test_inverse_and_composition_are_consistent() -> None:
    permutation = Permutation((2, 0, 1))
    identity = Permutation.identity(3)

    assert permutation.inverse().image == (1, 2, 0)
    assert permutation.compose(permutation.inverse()) == identity
    assert permutation.inverse().compose(permutation) == identity


def test_compose_applies_self_after_other() -> None:
    first = Permutation((2, 0, 1))
    second = Permutation((1, 2, 0))

    composed = first.compose(second)

    assert composed.apply_tuple((0, 1, 2)) == first.apply_tuple(second.apply_tuple((0, 1, 2)))


def test_sign_tracks_parity() -> None:
    assert Permutation.identity(3).sign == 1
    assert Permutation((1, 0, 2)).sign == -1
    assert Permutation((2, 0, 1)).sign == 1


def test_invalid_permutation_rejected() -> None:
    with pytest.raises(ValueError, match="zero-based bijection"):
        Permutation((0, 0, 2))
