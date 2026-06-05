"""Tests for Specht irrep metadata scaffolds."""

from __future__ import annotations

import pytest

from spenn.reps import SpechtIrrep, specht_irrep


def test_specht_irrep_records_orthogonal_basis_convention() -> None:
    irrep = specht_irrep((2, 1))
    metadata = irrep.metadata()

    assert irrep.basis == "orthogonal"
    assert metadata.basis == "orthogonal"
    assert metadata.partition.parts == (2, 1)


def test_specht_irrep_rejects_non_orthogonal_basis() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        SpechtIrrep((2, 1), basis="seminormal")
