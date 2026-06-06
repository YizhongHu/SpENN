"""Tests for Specht irrep metadata scaffolds."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spenn.data.partition import Partition
from spenn.data.permutation import all_permutations
from spenn.reps import (
    IrrepMetadata,
    SpechtIrrep,
    generate_irrep_tensor_cache,
    load_default_irrep_metadata,
    specht_irrep,
)
from spenn.reps.fixture_generators.sage_specht import generate_sage_irrep_tensor_cache


def test_specht_irrep_records_orthogonal_basis_convention() -> None:
    irrep = specht_irrep((2, 1))
    metadata = irrep.metadata()

    assert irrep.basis == "orthogonal"
    assert metadata.basis == "orthogonal"
    assert metadata.partition.parts == (2, 1)


def test_specht_irrep_rejects_non_orthogonal_basis() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        SpechtIrrep((2, 1), basis="seminormal")


def test_irrep_metadata_generates_discrete_metadata_and_dimensions() -> None:
    metadata = IrrepMetadata.generate(max_order=3)

    assert metadata.max_order == 3
    assert [partition.parts for partition in metadata.partitions[3]] == [(3,), (2, 1), (1, 1, 1)]
    assert metadata.dimension(Partition((3,))) == 1
    assert metadata.dimension(Partition((2, 1))) == 2
    assert metadata.dimension(Partition((1, 1, 1))) == 1


def test_irrep_metadata_roundtrip_json(tmp_path) -> None:
    metadata = IrrepMetadata.generate(max_order=3, tensor_cache=None)
    path = tmp_path / "irreps.json"

    metadata.save(path)
    loaded = IrrepMetadata.load(path)

    assert loaded.to_json_data() == metadata.to_json_data()
    assert loaded.tensor_cache == {}


def test_specht_representation_matrices_are_orthogonal_and_homomorphic() -> None:
    irrep = specht_irrep((2, 1))
    identity = torch.eye(2, dtype=torch.float64)
    permutation_group = all_permutations(3)

    for permutation in permutation_group:
        matrix = irrep.representation(permutation)
        torch.testing.assert_close(matrix.T @ matrix, identity)

    for first in permutation_group:
        for second in permutation_group:
            composed = first.compose(second)
            torch.testing.assert_close(
                irrep.representation(first) @ irrep.representation(second),
                irrep.representation(composed),
            )


def test_default_irrep_tensor_cache_contains_small_order_representations() -> None:
    cache = load_default_irrep_metadata().tensor_cache

    assert cache["basis_convention"] == "orthogonal"
    assert cache["generator"] == "sage_specht"
    assert cache["sage_version"]
    assert "3|2,1" in cache["representations"]
    assert cache["representations"]["3|2,1"]["1,0,2"].shape == (2, 2)


def test_sage_irrep_tensor_cache_generator_when_sage_is_available() -> None:
    pytest.importorskip("sage.all")

    cache = generate_sage_irrep_tensor_cache(max_order=3)

    assert cache["generator"] == "sage_specht"
    assert "3|2,1" in cache["representations"]


def test_public_irrep_tensor_cache_generator_accepts_torch_dtype_with_sage_executable() -> None:
    sage_executable = Path("/n/sw/sage-10.3/sage")
    if not sage_executable.exists():
        pytest.skip("FASRC Sage executable is unavailable")

    cache = generate_irrep_tensor_cache(
        max_order=1,
        dtype=torch.float64,
        sage_executable=str(sage_executable),
    )

    assert cache["generator"] == "sage_specht"
    assert cache["representations"]["1|1"]["0"].dtype is torch.float64


def test_project_irrep_metadata_loads_tensor_cache_file() -> None:
    metadata = IrrepMetadata.load(Path("spenn/cache/irreps.json"))

    assert metadata.tensor_cache_path is not None
    assert metadata.tensor_cache_path.name == "irreps_m3.pt"
    assert metadata.data["generator"] == "sage_specht"
    assert metadata.data["sage_version"]
    assert metadata.tensor_cache["generator"] == "sage_specht"
    assert metadata.tensor_cache["representations"]["3|2,1"]["1,0,2"].shape == (2, 2)
