"""Tests for virtual-support path enumeration."""

from __future__ import annotations

import pytest

from spenn.reps.paths import VirtualPath, enumerate_virtual_paths, validate_virtual_path


def test_virtual_path_generator_enforces_max_order() -> None:
    paths = enumerate_virtual_paths(max_order=2, target_order=1, left_order=1, right_order=1)

    assert paths
    assert all(path.support_order <= 2 for path in paths)


def test_virtual_path_generator_enforces_input_coverage() -> None:
    paths = enumerate_virtual_paths(max_order=3, target_order=1, left_order=1, right_order=2)

    assert paths
    assert all(path.input_support == set(range(path.support_order)) for path in paths)


def test_virtual_path_validation_rejects_uncovered_support() -> None:
    path = VirtualPath(
        support_order=2,
        target_order=1,
        left_order=1,
        right_order=1,
        target_injection=(0,),
        left_injection=(0,),
        right_injection=(0,),
    )

    with pytest.raises(ValueError, match="cover"):
        validate_virtual_path(path, max_order=2)
