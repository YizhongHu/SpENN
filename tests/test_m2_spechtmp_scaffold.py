"""Tests for the M=2 SpechtMP scaffold surface."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data import BranchDict, FeatureDict, MessageDict, Par, TensorProductDict
from spenn.data.batch import ElectronBatch
from spenn.nn.activations import TensorProductActivation
from spenn.nn.encoding import BaseEncoder, ElectronPairEncoder
from spenn.nn.spechtmp import MessageHead, SpechtMP, SpechtMPLayer, UpdateHead
from spenn.nn.update import RawUpdate, ResidualUpdate
from spenn.reps import BranchMap, FusionMap


def test_tensor_product_dict_stores_and_validates_blocks() -> None:
    tensor = torch.ones(3, 4, 5, 2, 2, 2, 2, 1, 1)
    products = TensorProductDict()

    target = Par("S")
    source = Par("H")
    products.set(target, source, source, tensor)

    assert products.has(target, source, source)
    assert products.get(target, source, source) is tensor
    flat_target, flat_left, flat_right, flat_tensor = next(products.flat_items())
    assert (flat_target, flat_left, flat_right) == (target, source, source)
    assert flat_tensor is tensor

    products.validate(batch_size=3, n_electrons=2)
    moved = products.clone().to(dtype=torch.float64)
    assert moved.get(target, source, source).dtype == torch.float64


def test_message_dict_stores_and_validates_blocks() -> None:
    tensor = torch.ones(3, 4, 2, 2, 1, 1)

    partition = Par("S")
    messages = MessageDict({partition: tensor})

    assert messages.has(partition)
    assert messages.get(partition) is tensor
    flat_partition, flat_tensor = next(messages.flat_items())
    assert flat_partition == partition
    assert flat_tensor is tensor

    messages.validate(batch_size=3, n_electrons=2)
    moved = messages.clone().to(dtype=torch.float64)
    assert moved.get(partition).dtype == torch.float64


def test_branch_dict_stores_and_validates_blocks() -> None:
    tensor = torch.ones(3, 4, 5, 2, 2, 2, 2, 1, 1)
    branches = BranchDict()

    target = Par("S")
    source = Par("A")
    branches.set(target, source, tensor)

    assert branches.has(target, source)
    assert branches.get(target, source) is tensor
    flat_target, flat_source, flat_tensor = next(branches.flat_items())
    assert (flat_target, flat_source) == (target, source)
    assert flat_tensor is tensor

    branches.validate(batch_size=3, n_electrons=2)
    moved = branches.clone().to(dtype=torch.float64)
    assert moved.get(target, source).dtype == torch.float64
    assert list(branches.to_dict()[target]) == [source]


def test_scaffold_imports_and_constructors_match_config_surface() -> None:
    fusion_map = FusionMap(M=2, M_virtual=2, maps={})
    branch_map = BranchMap(M=2, M_virtual=2, maps={})
    message_head = MessageHead(
        M=2,
        M_virtual=2,
        channels={"order1": {"(1)": 4}},
        activation=nn.Identity(),
    )
    update_head = UpdateHead(M=2, channels={"order1": {"(1)": 4}}, activation=nn.Identity())
    layer = SpechtMPLayer(
        fusion_map=fusion_map,
        message_head=message_head,
        branch_map=branch_map,
        update_head=update_head,
    )
    stack = SpechtMP(M=2, M_virtual=2, num_layers=2, channels={"order1": {"(1)": 4}})
    encoder = ElectronPairEncoder(
        name="basic",
        max_order=2,
        channels=[0, 4, 3],
    )

    assert isinstance(layer, SpechtMPLayer)
    assert len(stack.layers) == 2
    assert layer.update_head is update_head
    assert all(isinstance(stack_layer.update_head, UpdateHead) for stack_layer in stack.layers)
    assert isinstance(layer.update, RawUpdate)
    assert all(isinstance(stack_layer.update, RawUpdate) for stack_layer in stack.layers)
    assert encoder.output_keys() == (Par("H"), Par("S"), Par("A"))


def test_stale_spechtmp_exports_are_removed() -> None:
    import spenn.nn.spechtmp as spechtmp

    assert spechtmp.__all__ == ["MessageHead", "SpechtMP", "SpechtMPLayer", "UpdateHead"]
    assert not hasattr(spechtmp, "SpechtFuser")
    assert not hasattr(spechtmp, "SpechtBrancher")
    assert not hasattr(spechtmp, "LowRankVirtualBrancher")


def test_stale_branching_exports_are_removed() -> None:
    import spenn.reps as reps

    assert reps.__all__ == ["BranchMap", "FusionMap"]
    assert hasattr(reps, "BranchMap")
    assert not hasattr(reps, "BranchingMap")


def test_tensor_product_activation_scaffold_uses_two_modules() -> None:
    scalar_activation = nn.Tanh()
    tensor_activation = nn.Identity()
    activation = TensorProductActivation(
        scalar_activation=scalar_activation,
        tensor_activation=tensor_activation,
    )

    assert activation.scalar_activation is scalar_activation
    assert activation.tensor_activation is tensor_activation
    with pytest.raises(NotImplementedError, match="TensorProductActivation.forward"):
        activation(FeatureDict())


def test_spechtmp_accepts_explicit_update_module() -> None:
    update = ResidualUpdate()
    update_head = UpdateHead()
    layer = SpechtMPLayer(update_head=update_head, update=update)
    stack = SpechtMP(num_layers=2, update_head=update_head, update=update)

    assert layer.update_head is update_head
    assert layer.update is update
    assert all(stack_layer.update_head is update_head for stack_layer in stack.layers)
    assert all(stack_layer.update is update for stack_layer in stack.layers)


def test_scaffold_boundaries_raise_not_implemented() -> None:
    features = FeatureDict()
    products = TensorProductDict()
    messages = MessageDict()
    branches = BranchDict()
    batch = ElectronBatch(positions=torch.zeros(2, 2, 3))
    source = Par("H")
    target = Par("S")

    products = FusionMap()(features)
    assert len(products) == 0
    with pytest.raises(KeyError, match="Missing source"):
        FusionMap().fuse_pair(features, source, source, target)
    assert len(MessageHead()(products)) == 0
    assert len(MessageHead().linear_messages(features)) == 0
    assert len(MessageHead().tensor_product_messages(products)) == 0
    assert len(MessageHead().apply_irrep_activation(messages)) == 0
    assert len(BranchMap()(messages)) == 0
    with pytest.raises(KeyError, match="Missing source"):
        BranchMap().branch_irrep(messages, target, source)
    assert len(UpdateHead()(branches)) == 0
    assert len(UpdateHead().linear_updates(branches)) == 0
    assert len(UpdateHead().apply_irrep_activation(features)) == 0
    with pytest.raises(NotImplementedError, match="BaseEncoder.output_keys"):
        BaseEncoder().output_keys()
    with pytest.raises(NotImplementedError, match="BaseEncoder.forward"):
        BaseEncoder()(batch)
    assert len(SpechtMP()(features)) == 0


def test_scaffolds_reject_unsupported_orders() -> None:
    with pytest.raises(ValueError, match="M <= 2"):
        FusionMap(M=3)
    with pytest.raises(ValueError, match="M <= 2"):
        BranchMap(M_virtual=3)
    with pytest.raises(ValueError, match="M <= 2"):
        MessageHead(M=3)
    with pytest.raises(ValueError, match="M <= 2"):
        UpdateHead(M=3)
    with pytest.raises(ValueError, match="M <= 2"):
        SpechtMP(M_virtual=3)
    with pytest.raises(ValueError, match="max_order <= 2"):
        ElectronPairEncoder(max_order=3)
