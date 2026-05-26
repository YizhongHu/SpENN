"""Tests for partition-backed feature metadata."""

from __future__ import annotations

import pytest
import torch

from spenn.data_structures import FeatureDict, IrrepTensor, Partition, normalize_partition


def test_partition_canonicalizes_equality_and_hashing() -> None:
    partition = Partition(order=3, parts=(1, 2))
    canonical = Partition(order=3, parts=(2, 1))

    assert partition.parts == (2, 1)
    assert partition == canonical
    assert hash(partition) == hash(canonical)


def test_partition_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        Partition(order=3, parts=(2, 2))
    with pytest.raises(ValueError):
        Partition(order=2, parts=(2, 0))
    with pytest.raises(TypeError):
        Partition(order=1, parts=(True,))
    with pytest.raises(ValueError):
        normalize_partition(2, Partition(order=3, parts=(2, 1)))


def test_normalize_partition_accepts_boundary_specs() -> None:
    assert normalize_partition(2, 2) == Partition(order=2, parts=(2,))
    assert normalize_partition(2, [1, 1]) == Partition(order=2, parts=(1, 1))
    assert normalize_partition(3, "(2,1)") == Partition(order=3, parts=(2, 1))
    assert normalize_partition(3, (1, 2)) == Partition(order=3, parts=(2, 1))


def test_feature_dict_stores_partition_keys_and_accepts_legacy_specs() -> None:
    tensor = torch.ones(1, 2, 2, 3)
    features = FeatureDict()
    features.set(2, (1, 1), tensor)

    partition = Partition(order=2, parts=(1, 1))
    assert features.get(2, partition) is tensor
    assert features.get(2, [1, 1]) is tensor
    assert features.get(2, "(1,1)") is tensor
    assert features.has(2, partition)
    assert list(features[2]) == [partition]
    assert isinstance(next(iter(features[2])), Partition)
    flat_order, flat_partition, flat_tensor = next(features.flat_items())
    assert (flat_order, flat_partition) == (2, partition)
    assert flat_tensor is tensor


def test_feature_dict_setitem_to_dict_and_supported_validation_use_partitions() -> None:
    tensor = torch.ones(1, 2, 2, 3)
    features = FeatureDict({2: {"(2)": tensor}})
    partition = Partition(order=2, parts=(2,))

    assert features.get(2, 2) is tensor
    assert list(features.to_dict()[2]) == [partition]
    features.validate(supported=[(2, partition)])
    with pytest.raises(KeyError):
        features.validate(supported=[(2, (1, 1))])


def test_irrep_tensor_stores_canonical_partition() -> None:
    tensor = torch.zeros(1, 3, 3, 3, 4, 2, 2)
    wrapped = IrrepTensor(order=3, irrep=(1, 2), tensor=tensor)

    assert wrapped.irrep == Partition(order=3, parts=(2, 1))
    assert wrapped.order == 3
    with pytest.raises(ValueError):
        IrrepTensor(order=2, irrep=Partition(order=3, parts=(2, 1)), tensor=tensor)


def test_order3_mixed_partition_requires_two_by_two_irrep_axes() -> None:
    features = FeatureDict({3: {(2, 1): torch.zeros(4, 5, 5, 5, 7, 2, 2)}})

    features.validate(batch_size=4, n_electrons=5)

    bad = FeatureDict({3: {(2, 1): torch.zeros(4, 5, 5, 5, 7, 1, 1)}})
    with pytest.raises(ValueError, match="trailing irrep axes"):
        bad.validate(batch_size=4, n_electrons=5)


def test_order3_one_dimensional_partitions_require_one_by_one_irrep_axes() -> None:
    FeatureDict({3: {(3): torch.zeros(4, 5, 5, 5, 7, 1, 1)}}).validate(batch_size=4, n_electrons=5)
    FeatureDict({3: {(1, 1, 1): torch.zeros(4, 5, 5, 5, 7, 1, 1)}}).validate(
        batch_size=4,
        n_electrons=5,
    )

    bad = FeatureDict({3: {(3): torch.zeros(4, 5, 5, 5, 7, 2, 2)}})
    with pytest.raises(ValueError, match="trailing irrep axes"):
        bad.validate(batch_size=4, n_electrons=5)
