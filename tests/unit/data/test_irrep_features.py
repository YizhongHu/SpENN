"""Tests for partition-backed feature metadata."""

from __future__ import annotations

import pytest
import torch

from spenn.data import FeatureDict, IrrepFeature, IrrepMessage, IrrepTensor, IrrepTensors, Par, Partition, as_partition, normalize_partition


def test_partition_canonicalizes_equality_and_hashing() -> None:
    partition = Par(order=3, parts=(1, 2))
    canonical = Par(3, (2, 1))

    assert partition.parts == (2, 1)
    assert partition == canonical
    assert hash(partition) == hash(canonical)


def test_partition_constructor_accepts_shorthand_specs() -> None:
    assert Par((1, 2)) == Par("(2, 1)") == Par(order=3, parts=(2, 1))
    assert Par([1, 1]) == Par("A")
    assert Par("h") == Par((1,))
    assert Par("S") == Par((2,))
    assert Par("T") == Par((3,))
    assert Par("V") == Par((2, 1))
    assert Par("E") == Par((1, 1, 1))
    assert Par("S4") == Par((4,))
    assert Par("A4") == Par((1, 1, 1, 1))
    assert Par("V4") == Par((3, 1))
    assert Par("V2") == Par("A")


def test_partition_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        Par(order=3, parts=(2, 2))
    with pytest.raises(ValueError):
        Par(order=2, parts=(2, 0))
    with pytest.raises(TypeError):
        Par(order=1, parts=(True,))
    with pytest.raises(TypeError):
        Par("not-a-partition")
    with pytest.raises(ValueError):
        Par("S0")
    with pytest.raises(ValueError):
        Par("V1")
    with pytest.raises(ValueError):
        normalize_partition(2, Par(order=3, parts=(2, 1)))


def test_normalize_partition_accepts_boundary_specs() -> None:
    assert normalize_partition(2, 2) == Par("S")
    assert normalize_partition(2, [1, 1]) == Par("A")
    assert normalize_partition(3, "(2,1)") == Par("V")
    assert normalize_partition(3, (1, 2)) == Par("V")
    assert as_partition("(2,1)") == Par("V")
    assert as_partition((1, 1)) == Par("A")


def test_irrep_feature_stores_partition_keys() -> None:
    tensor = torch.ones(1, 3, 2, 2, 1, 1)
    features = FeatureDict()

    partition = Par("A")
    features.set(partition, tensor)

    assert features.get(partition) is tensor
    assert features.has(partition)
    assert list(features) == [partition]
    assert isinstance(next(iter(features)), Partition)
    flat_partition, flat_tensor = next(features.flat_items())
    assert flat_partition == partition
    assert flat_tensor is tensor
    assert isinstance(features, IrrepFeature)
    assert isinstance(features, IrrepTensors)


def test_irrep_message_container_uses_irrep_feature_api() -> None:
    tensor = torch.ones(1, 3, 2, 2, 1, 1)
    messages = IrrepMessage({Par("A"): tensor})

    assert messages.get(Par("A")) is tensor
    assert messages.has(Par("A"))
    messages.validate(batch_size=1, n_electrons=2)


def test_irrep_tensors_require_common_channel_count() -> None:
    with pytest.raises(ValueError, match="channel count"):
        IrrepFeature(
            {
                Par("H"): torch.zeros(1, 2, 3, 1, 1),
                Par("S"): torch.zeros(1, 3, 3, 3, 1, 1),
            }
        )


def test_irrep_feature_setitem_to_dict_and_supported_validation_use_partitions() -> None:
    tensor = torch.ones(1, 3, 2, 2, 1, 1)
    partition = Par("S")
    features = FeatureDict({partition: tensor})

    assert features.get(partition) is tensor
    assert list(features.to_dict()) == [partition]
    features.validate(supported=[partition])
    with pytest.raises(KeyError):
        features.validate(supported=[Par("A")])


def test_irrep_feature_add_and_magic_add_sum_matching_keys() -> None:
    partition = Par("H")
    left = FeatureDict({partition: torch.ones(2, 3, 4, 1, 1)})
    right = FeatureDict({partition: 2.0 * torch.ones(2, 3, 4, 1, 1)})
    summed = left.add(right)
    magic = left + right

    assert torch.equal(summed.get(partition), 3.0 * torch.ones(2, 3, 4, 1, 1))
    assert torch.equal(magic.get(partition), summed.get(partition))
    assert torch.equal(left.get(partition), torch.ones(2, 3, 4, 1, 1))


def test_irrep_tensor_stores_canonical_partition() -> None:
    tensor = torch.zeros(1, 4, 3, 3, 3, 2, 2)
    wrapped = IrrepTensor(order=3, irrep=Par(order=3, parts=(1, 2)), tensor=tensor)

    assert wrapped.irrep == Par("V")
    assert wrapped.order == 3
    with pytest.raises(ValueError):
        IrrepTensor(order=2, irrep=Par("V"), tensor=tensor)


def test_order3_mixed_partition_requires_two_by_two_irrep_axes() -> None:
    partition = Par("V")
    features = FeatureDict({partition: torch.zeros(4, 7, 5, 5, 5, 2, 2)})

    features.validate(batch_size=4, n_electrons=5)

    bad = FeatureDict({partition: torch.zeros(4, 7, 5, 5, 5, 1, 1)})
    with pytest.raises(ValueError, match="trailing irrep axes"):
        bad.validate(batch_size=4, n_electrons=5)


def test_order3_one_dimensional_partitions_require_one_by_one_irrep_axes() -> None:
    trivial = Par("T")
    sign = Par("E")
    FeatureDict({trivial: torch.zeros(4, 7, 5, 5, 5, 1, 1)}).validate(batch_size=4, n_electrons=5)
    FeatureDict({sign: torch.zeros(4, 7, 5, 5, 5, 1, 1)}).validate(
        batch_size=4,
        n_electrons=5,
    )

    bad = FeatureDict({trivial: torch.zeros(4, 7, 5, 5, 5, 2, 2)})
    with pytest.raises(ValueError, match="trailing irrep axes"):
        bad.validate(batch_size=4, n_electrons=5)
