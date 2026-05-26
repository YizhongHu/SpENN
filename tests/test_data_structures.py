"""Tests for public data-structure contracts."""

from __future__ import annotations

from itertools import combinations

import pytest
import torch

from spenn.data_structures import all_ordered_tuples, all_pairs, all_subsets, all_triples
from spenn.data_structures.batch import ElectronBatch, WavefunctionOutput, validate_output


def test_electron_batch_accepts_higher_rank_sample_shape() -> None:
    positions = torch.zeros(2, 3, 4, 5)
    spins = torch.zeros(2, 3, 4)
    nuclear_positions = torch.zeros(2, 3, 7, 5)
    nuclear_charges = torch.ones(2, 3, 7)

    batch = ElectronBatch(
        positions=positions,
        spins=spins,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        aux={"tag": "multi"},
    )

    assert batch.sample_shape == (2, 3)
    assert batch.n_configurations == 6
    assert batch.batch_size == 6
    assert batch.n_electrons == 4
    assert batch.spatial_dim == 5


def test_electron_batch_rejects_mismatched_nuclear_counts() -> None:
    with pytest.raises(ValueError, match="n_nuclei"):
        ElectronBatch(
            positions=torch.zeros(2, 4, 5),
            nuclear_positions=torch.zeros(7, 5),
            nuclear_charges=torch.ones(6),
        )


def test_electron_batch_flatten_samples_preserves_metadata() -> None:
    batch = ElectronBatch(
        positions=torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5),
        spins=torch.zeros(2, 3, 4),
        nuclear_positions=torch.zeros(2, 3, 7, 5),
        nuclear_charges=torch.ones(2, 3, 7),
        aux={"tag": "multi"},
    )

    flat = batch.flatten_samples()

    assert flat.positions.shape == (6, 4, 5)
    assert flat.spins is not None and flat.spins.shape == (6, 4)
    assert flat.nuclear_positions is not None and flat.nuclear_positions.shape == (6, 7, 5)
    assert flat.nuclear_charges is not None and flat.nuclear_charges.shape == (6, 7)
    assert flat.aux == {"tag": "multi"}


def test_wavefunction_output_accepts_exact_nodes_and_sample_shapes() -> None:
    logabs = torch.tensor([[0.0, -torch.inf], [-3.0, -4.0]])
    sign = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
    output = WavefunctionOutput(logabs=logabs, sign=sign)

    validate_output(output, sample_shape=(2, 2))
    validate_output(output, batch_size=4)
    validate_output(WavefunctionOutput(logabs=torch.zeros(3), sign=torch.ones(3)), batch_size=3)


def test_wavefunction_output_rejects_inconsistent_exact_nodes() -> None:
    with pytest.raises(ValueError, match="exact zeros"):
        WavefunctionOutput(logabs=torch.tensor([0.0]), sign=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="exact zeros"):
        WavefunctionOutput(logabs=torch.tensor([-torch.inf]), sign=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="phase"):
        WavefunctionOutput(logabs=torch.zeros(2), sign=torch.ones(2), phase=torch.zeros(3))


def test_subset_helpers_generalize_beyond_triples() -> None:
    assert all_subsets(5, 2) == all_pairs(5)
    assert all_subsets(5, 3) == all_triples(5)
    assert all_subsets(5, 4) == [tuple(item) for item in combinations(range(5), 4)]
    assert all_ordered_tuples(4, 4)[0] == (0, 1, 2, 3)
    assert len(all_ordered_tuples(4, 4)) == 24
    assert (0, 0, 0) in all_ordered_tuples(2, 3, distinct=False)
    assert len(all_ordered_tuples(2, 3, distinct=False)) == 8
