"""Reproducibility checks on the study README (experiments-owned)."""

from __future__ import annotations

from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    assert README.is_file(), "study README is required for reproducibility"
    return README.read_text(encoding="utf-8")


def test_readme_separates_validation_from_final_eval(readme_text: str) -> None:
    # Validation selects; it never sees the exact Hooke reference energy.
    assert "exact" in readme_text.lower()
    assert "reference energy" in readme_text.lower()
    assert "selection" in readme_text.lower()
    assert "final eval" in readme_text.lower()


def test_readme_documents_local_authority(readme_text: str) -> None:
    assert "authoritative" in readme_text.lower()
    assert "visualization" in readme_text.lower()  # W&B is visualization only
    assert "W&B" in readme_text


def test_readme_documents_tie_breaker_rule(readme_text: str) -> None:
    assert "tie-break" in readme_text.lower()
    assert "selection margin" in readme_text.lower() or "margin" in readme_text.lower()
    assert "energy_variance" in readme_text


def test_readme_includes_command_examples(readme_text: str) -> None:
    assert "collect.py" in readme_text
    assert "select.py" in readme_text
    assert "evaluate_selected.py" in readme_text
    assert "--dry-run" in readme_text or "dry-run" in readme_text
