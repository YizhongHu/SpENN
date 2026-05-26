"""Smoke tests for the Phase 1 project skeleton."""

from __future__ import annotations

import spenn


def test_package_import_is_lightweight() -> None:
    """The root package should import without training side effects."""
    assert spenn.__version__ == "0.0.0"
