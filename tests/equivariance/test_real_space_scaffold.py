"""Tests for the real-space SpechtMP scaffold surface."""

from __future__ import annotations

import pytest
import torch

from spenn.data.base import EquivariantMap
from spenn.data.batch import ElectronBatch
from spenn.data.irrep_features import IrrepFeature, IrrepMessage, IrrepTensors
from spenn.data.partitions import Par
from spenn.data.real_features import RealConcatenatedState, RealFeature, RealMessage
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
from spenn.reps.fourier import FourierTransform, InverseFourierTransform
from spenn.reps.fusion import FusionMap


class ConstantConvolution(EquivariantMap):
    """Return a fixed real-space message proposal."""

    def __init__(self, messages: RealMessage) -> None:
        super().__init__()
        self.messages = messages

    def forward(self, features: RealFeature) -> RealMessage:
        """Return cloned messages independent of input features."""

        return self.messages.clone()


class IdentityOrderOneFourier(FourierTransform):
    """Test-only order-1 identity projection."""

    def forward(self, tensors: RealFeature) -> IrrepFeature:
        """Return order-1 data with scalar irrep axes appended."""

        return IrrepFeature({Par("H"): tensors[1].unsqueeze(-1).unsqueeze(-1)})


class IdentityOrderOneInverseFourier(InverseFourierTransform):
    """Test-only order-1 identity reconstruction."""

    def forward(self, tensors: IrrepFeature) -> RealFeature:
        """Return order-1 real data with scalar irrep axes removed."""

        block = tensors.get(Par("H"))
        return RealFeature([torch.empty(block.shape[0], 0, dtype=block.dtype, device=block.device), block[..., 0, 0]])


def test_real_space_scaffold_imports() -> None:
    assert Embedding.__name__ == "Embedding"
    assert Convolution.__name__ == "Convolution"
    assert Pooling.__name__ == "Pooling"
    assert RealSpechtMPLayer.__name__ == "RealSpechtMPLayer"
    assert RealFeature().__class__.__name__ == "RealFeature"
    assert RealMessage().__class__.__name__ == "RealMessage"
    assert RealConcatenatedState().__class__.__name__ == "RealConcatenatedState"
    assert isinstance(IrrepFeature(), IrrepTensors)
    assert isinstance(IrrepMessage(), IrrepTensors)
    assert Convolution().include_linear is True


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
        RealSpechtMPLayer()(RealConcatenatedState())
    with pytest.raises(NotImplementedError, match="requires fourier"):
        RealToIrrepMessageUpdate()(None, RealMessage())
    with pytest.raises(NotImplementedError, match="requires fourier"):
        RealToIrrepFeatureUpdate()(RealFeature(), RealFeature())
    with pytest.raises(NotImplementedError, match="FourierTransform.forward"):
        FourierTransform(partitions=(Par("H"),))(RealFeature())
    with pytest.raises(NotImplementedError, match="InverseFourierTransform.forward"):
        InverseFourierTransform(partitions=(Par("H"),))(IrrepFeature())


def test_real_spechtmp_layer_composes_injected_components() -> None:
    old_tensor = torch.zeros(1, 1, 2)
    proposal_tensor = torch.ones(1, 1, 2)
    state = RealConcatenatedState(features=RealFeature([torch.empty(1, 0), old_tensor]))
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


def test_fourier_inverse_contract_roundtrip_with_identity_subclasses() -> None:
    features = RealFeature(
        [
            torch.empty(1, 0),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
        ]
    )

    reconstructed = IdentityOrderOneInverseFourier()(IdentityOrderOneFourier()(features))

    assert torch.equal(reconstructed[1], features[1])


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
