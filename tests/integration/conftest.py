"""Integration-test pytest configuration."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag tests collected from this directory as integration tests."""

    integration_dir = Path(__file__).resolve().parent
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        if integration_dir in item_path.parents:
            item.add_marker(pytest.mark.integration)
