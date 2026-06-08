"""Regression guard: the spenn.testing package must not come back."""

from __future__ import annotations

import importlib.util


def test_spenn_testing_package_does_not_exist() -> None:
    assert importlib.util.find_spec("spenn.testing") is None
