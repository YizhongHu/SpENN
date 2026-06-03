"""Tests for the real-space SpechtMP scaffold surface."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data.batch import ElectronBatch
from spenn.data.real_features import ConcatenatedState, RealFeature, RealMessage
from spenn.nn.real_space import (
    Convolution,
    Embedding,
    FeatureUpdate,
    MessageUpdate,
    Pooling,
    RealSpechtMPLayer,
    RealToIrrepFeatureUpdate,
    RealToIrrepMessageUpdate,
    SpechtFeatureActivation,
    SpechtMessageActivation,
)
from spenn.nn.spechtmp import MessageHead, SpechtMP, SpechtMPLayer, UpdateHead
from spenn.nn.utils.update import ResidualUpdate
from spenn.reps.branch import BranchMap
from spenn.reps.fusion import FusionMap


class ConstantConvolution(nn.Module):
    """Return a fixed real-space message proposal."""

    def __init__(self, messages: RealMessage) -> None:
        super().__init__()
        self.messages = messages

    def forward(self, features: RealFeature) -> RealMessage:
        """Return cloned messages independent of input features."""

        return self.messages.clone()


def test_real_space_scaffold_imports() -> None:
    assert Embedding.__name__ == "Embedding"
    assert Convolution.__name__ == "Convolution"
    assert Pooling.__name__ == "Pooling"
    assert RealSpechtMPLayer.__name__ == "RealSpechtMPLayer"
    assert RealFeature().__class__.__name__ == "RealFeature"
    assert RealMessage().__class__.__name__ == "RealMessage"
    assert ConcatenatedState().__class__.__name__ == "ConcatenatedState"


def test_pooling_pool_same_order_controls_same_key_copy() -> None:
    tensor = torch.ones(2, 3, 4)
    messages = RealMessage([torch.empty(2, 0), tensor])

    pooled = Pooling(pool_same_order=True)(messages)
    skipped = Pooling(pool_same_order=False)(messages)

    assert Pooling().pool_same_order is True
    assert torch.equal(pooled[1], tensor)
    assert pooled[1] is not tensor
    assert skipped[0].shape == (2, 0)
    assert skipped[1].shape == (2, 0, 4)


def test_math_heavy_real_space_boundaries_raise_not_implemented() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 3, 3))

    with pytest.raises(NotImplementedError, match="Embedding.forward"):
        Embedding()(batch)
    with pytest.raises(NotImplementedError, match="Convolution.forward"):
        Convolution()(RealFeature())
    with pytest.raises(NotImplementedError, match="requires convolution"):
        RealSpechtMPLayer()(ConcatenatedState())
    with pytest.raises(NotImplementedError, match="requires fourier"):
        RealToIrrepMessageUpdate()(None, RealMessage())
    with pytest.raises(NotImplementedError, match="requires fourier"):
        RealToIrrepFeatureUpdate()(RealFeature(), RealFeature())


def test_real_spechtmp_layer_composes_injected_components() -> None:
    old_tensor = torch.zeros(1, 1, 2)
    proposal_tensor = torch.ones(1, 1, 2)
    state = ConcatenatedState(features=RealFeature([torch.empty(1, 0), old_tensor]))
    layer = RealSpechtMPLayer(
        convolution=ConstantConvolution(RealMessage([torch.empty(1, 0), proposal_tensor])),
        message_activation=SpechtMessageActivation(),
        message_update=MessageUpdate(),
        pooling=Pooling(pool_same_order=True),
        feature_activation=SpechtFeatureActivation(),
        feature_update=FeatureUpdate(),
    )

    output = layer(state)

    assert output.messages is not None
    assert torch.equal(output.messages[1], proposal_tensor)
    assert torch.equal(output.features[1], proposal_tensor)
    assert torch.equal(state.features[1], old_tensor)


def test_legacy_spechtmp_constructors_warn_about_real_space_layer() -> None:
    with pytest.warns(DeprecationWarning, match="RealSpechtMPLayer"):
        message_head = MessageHead()
    with pytest.warns(DeprecationWarning, match="RealSpechtMPLayer"):
        update_head = UpdateHead()
    with pytest.warns(DeprecationWarning, match="RealSpechtMPLayer"):
        SpechtMPLayer(
            fusion_map=FusionMap(),
            message_head=message_head,
            branch_map=BranchMap(),
            update_head=update_head,
            update=ResidualUpdate(),
        )
    with pytest.warns(DeprecationWarning, match="RealSpechtMPLayer"):
        SpechtMP(layers=())
