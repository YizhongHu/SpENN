"""Tests for activation routers and gate modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data import FeatureDict, Par
from spenn.nn.activations import (
    ActivationByIrrep,
    ActivationByType,
    ElementwiseFeatureActivation,
    GatedActivation,
    NormGateActivation,
)
from spenn.nn.gate import GateActivate, GateUpdate, ScalarGateActivate, ScalarGateUpdate


ORDER1 = Par("H")
ORDER2_SYM = Par("S")
ORDER2_SIGN = Par("A")
ORDER3_TENSOR = Par("V")


def _features() -> FeatureDict:
    return FeatureDict(
        {
            ORDER1: torch.ones(1, 1, 2, 1, 1),
            ORDER2_SYM: 2.0 * torch.ones(1, 1, 2, 2, 1, 1),
            ORDER2_SIGN: 3.0 * torch.ones(1, 1, 2, 2, 1, 1),
            ORDER3_TENSOR: 4.0 * torch.ones(1, 1, 2, 2, 2, 2, 2),
        }
    )


def _scalar_features(value: float) -> FeatureDict:
    return FeatureDict({ORDER1: value * torch.ones(1, 1, 2, 1, 1)})


class ScaleActivation(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, features: FeatureDict) -> FeatureDict:
        output = FeatureDict()
        for partition, tensor in features.flat_items():
            output.set(partition, self.scale * tensor)
        return output


def test_activation_by_type_routes_irreps_independently() -> None:
    activated = ActivationByType(
        symmetric=ScaleActivation(10.0),
        antisymmetric=ScaleActivation(20.0),
        tensor=ScaleActivation(30.0),
    )(_features())

    assert torch.equal(activated.get(ORDER1), 10.0 * _features().get(ORDER1))
    assert torch.equal(activated.get(ORDER2_SYM), 10.0 * _features().get(ORDER2_SYM))
    assert torch.equal(activated.get(ORDER2_SIGN), 20.0 * _features().get(ORDER2_SIGN))
    assert torch.equal(activated.get(ORDER3_TENSOR), 30.0 * _features().get(ORDER3_TENSOR))


def test_activation_by_irrep_routes_exact_partition_keys() -> None:
    activated = ActivationByIrrep(
        {
            ORDER1: ScaleActivation(2.0),
            ORDER2_SYM: ScaleActivation(3.0),
            ORDER2_SIGN: ScaleActivation(4.0),
            ORDER3_TENSOR: ScaleActivation(5.0),
        }
    )(_features())

    assert torch.equal(activated.get(ORDER1), 2.0 * _features().get(ORDER1))
    assert torch.equal(activated.get(ORDER2_SYM), 3.0 * _features().get(ORDER2_SYM))
    assert torch.equal(activated.get(ORDER2_SIGN), 4.0 * _features().get(ORDER2_SIGN))
    assert torch.equal(activated.get(ORDER3_TENSOR), 5.0 * _features().get(ORDER3_TENSOR))


def test_activation_by_irrep_rejects_missing_partition() -> None:
    with pytest.raises(KeyError, match="Missing activation module"):
        ActivationByIrrep({ORDER1: ScaleActivation(1.0)})(_features())


def test_activation_by_type_uses_explicit_parity_safe_nonlinearities() -> None:
    features = _features()

    activated = ActivationByType(
        symmetric=ElementwiseFeatureActivation(nn.Sigmoid()),
        antisymmetric=ElementwiseFeatureActivation(nn.Tanh()),
        tensor=NormGateActivation(nn.Sigmoid()),
    )(features)

    assert torch.allclose(activated.get(ORDER1), torch.sigmoid(features.get(ORDER1)))
    assert torch.allclose(activated.get(ORDER2_SYM), torch.sigmoid(features.get(ORDER2_SYM)))
    assert torch.allclose(activated.get(ORDER2_SIGN), torch.tanh(features.get(ORDER2_SIGN)))
    assert activated.get(ORDER3_TENSOR).shape == features.get(ORDER3_TENSOR).shape


def test_norm_gate_activation_scales_tensor_by_sigmoid_norm() -> None:
    tensor = torch.ones(1, 1, 2, 2, 2, 2, 2)
    features = FeatureDict({ORDER3_TENSOR: tensor})

    activated = NormGateActivation(nn.Sigmoid(), eps=0.0)(features)

    expected_gate = torch.sigmoid(torch.sqrt(tensor.square().sum(dim=(-2, -1), keepdim=True)))
    assert torch.allclose(activated.get(ORDER3_TENSOR), expected_gate * tensor)


def test_elementwise_feature_activation_preserves_shapes() -> None:
    features = _scalar_features(2.0)

    activated = ElementwiseFeatureActivation(nn.Tanh())(features)

    assert activated.get(ORDER1).shape == features.get(ORDER1).shape
    assert torch.allclose(activated.get(ORDER1), torch.tanh(features.get(ORDER1)))


def test_gate_templates_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="GateUpdate.forward"):
        GateUpdate(nn.Identity())(_scalar_features(1.0), _scalar_features(2.0))
    with pytest.raises(NotImplementedError, match="GateActivate.forward"):
        GateActivate(nn.Identity())(_scalar_features(1.0))


def test_scalar_gate_update_uses_scalar_update_component() -> None:
    gate = ScalarGateUpdate(nn.Identity())(_scalar_features(1.0), _scalar_features(2.0))

    assert torch.equal(gate.get(ORDER1), 2.0 * torch.ones(1, 1, 2, 1, 1))


def test_scalar_gate_activate_uses_scalar_feature_component() -> None:
    gate = ScalarGateActivate(nn.Identity())(_scalar_features(2.0))

    assert torch.equal(gate.get(ORDER1), 2.0 * torch.ones(1, 1, 2, 1, 1))


def test_scalar_gates_reject_missing_scalar_component() -> None:
    with pytest.raises(KeyError, match="Missing scalar"):
        ScalarGateActivate(nn.Identity())(FeatureDict())
    with pytest.raises(KeyError, match="Missing scalar"):
        ScalarGateUpdate(nn.Identity())(_scalar_features(1.0), FeatureDict())


def test_gated_activation_applies_feature_gate() -> None:
    gated = GatedActivation(ScalarGateActivate(nn.Identity()))(_scalar_features(2.0))

    assert torch.equal(gated.get(ORDER1), 4.0 * torch.ones(1, 1, 2, 1, 1))
