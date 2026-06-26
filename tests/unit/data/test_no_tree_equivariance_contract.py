"""Regression guards: the generic tree-based equivariance helpers stay gone."""

from __future__ import annotations

from pathlib import Path

import spenn.data.equivariant_state as equivariant_state

_FORBIDDEN = ("permute_tree", "validate_tree")


def test_permute_tree_is_gone() -> None:
    assert "permute_tree" not in equivariant_state.__all__
    assert not hasattr(equivariant_state, "permute_tree")


def test_validate_tree_is_gone() -> None:
    assert "validate_tree" not in equivariant_state.__all__
    assert not hasattr(equivariant_state, "validate_tree")


def test_no_forbidden_tree_helpers_in_spenn_source() -> None:
    root = Path(__file__).resolve().parents[2] / "spenn"
    offenders = [
        (path, symbol)
        for path in root.rglob("*.py")
        for symbol in _FORBIDDEN
        if symbol in path.read_text()
    ]
    assert offenders == [], f"forbidden tree helpers found in spenn source: {offenders}"
