"""Tests for equivariant activation routers and gate modules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from spenn.data import FeatureDict, Par
from spenn.nn.utils import activations as activation_module
from spenn.nn.utils import gate as gate_module
from spenn.nn.utils.activations import Activation, ActivationByIrrep, ActivationByType, GatedActivation
from spenn.nn.utils.gate import GateActivate, GateUpdate, NormGateActivate, ScalarGateActivate, ScalarGateUpdate


ORDER1 = Par("H")
ORDER2_SYM = Par("S")
ORDER2_SIGN = Par("A")
ORDER3_TENSOR = Par("V")
PERMUTATIONS = (
    torch.tensor([1, 0, 2]),
    torch.tensor([2, 0, 1]),
    torch.tensor([0, 2, 1]),
)


def _features() -> FeatureDict:
    h = torch.tensor(
        [
            [[1.0, -2.0, 0.5], [0.25, 1.5, -0.75]],
            [[-1.25, 0.0, 2.25], [3.0, -1.0, 0.125]],
        ],
        dtype=torch.float64,
    ).unsqueeze(-1).unsqueeze(-1)
    pair_base = torch.tensor(
        [
            [
                [[0.1, 1.2, -0.7], [2.0, -0.3, 0.5], [1.1, -1.3, 0.9]],
                [[-0.8, 0.4, 1.6], [0.7, 1.5, -0.2], [2.2, -0.6, 0.3]],
            ],
            [
                [[1.0, -1.5, 0.25], [0.8, -0.4, 1.4], [-1.1, 0.6, 0.2]],
                [[0.5, 1.1, -0.9], [-1.4, 0.75, 1.2], [0.3, -0.8, -0.1]],
            ],
        ],
        dtype=torch.float64,
    )
    symmetric = 0.5 * (pair_base + pair_base.transpose(2, 3))
    antisymmetric = 0.5 * (pair_base - pair_base.transpose(2, 3))
    tensor = torch.arange(2 * 2 * 3 * 3 * 3 * 2 * 2, dtype=torch.float64).reshape(2, 2, 3, 3, 3, 2, 2)
    tensor = tensor / 37.0 - 2.0
    return FeatureDict(
        {
            ORDER1: h,
            ORDER2_SYM: symmetric.unsqueeze(-1).unsqueeze(-1),
            ORDER2_SIGN: antisymmetric.unsqueeze(-1).unsqueeze(-1),
            ORDER3_TENSOR: tensor,
        }
    )


def _scalar_features(value: float) -> FeatureDict:
    data = value * torch.tensor([[[1.0, -2.0, 3.0]]], dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
    return FeatureDict({ORDER1: data})


def _permute_features(features: FeatureDict, permutation: torch.Tensor) -> FeatureDict:
    output = FeatureDict()
    for partition, tensor in features.flat_items():
        output.set(partition, _permute_tensor(tensor, partition, permutation))
    return output


def _permute_tensor(tensor: torch.Tensor, partition, permutation: torch.Tensor) -> torch.Tensor:
    permuted = tensor
    for axis in range(2, 2 + partition.order):
        permuted = permuted.index_select(axis, permutation.to(tensor.device))
    return permuted


def _assert_featuredict_close(left: FeatureDict, right: FeatureDict) -> None:
    assert set(left.keys()) == set(right.keys())
    for partition, tensor in left.flat_items():
        assert torch.allclose(tensor, right.get(partition), atol=1.0e-10, rtol=1.0e-10)


def _check_gate_equivariance(module: nn.Module, features: FeatureDict) -> None:
    original = module(features)
    for permutation in PERMUTATIONS:
        transformed = module(_permute_features(features, permutation))
        expected = _permute_features(original, permutation)
        _assert_featuredict_close(transformed, expected)


def _assert_pair_symmetry_contract(features: FeatureDict) -> None:
    symmetric = features.get(ORDER2_SYM)
    antisymmetric = features.get(ORDER2_SIGN)
    assert torch.allclose(symmetric, symmetric.transpose(2, 3), atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(antisymmetric, -antisymmetric.transpose(2, 3), atol=1.0e-10, rtol=1.0e-10)
    diagonal = antisymmetric.diagonal(dim1=2, dim2=3)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1.0e-10, rtol=1.0e-10)


def _norm_activation(*, normalize: bool = True) -> GatedActivation:
    return GatedActivation(NormGateActivate(nn.Tanh(), eps=1.0e-12, normalize=normalize))


class ScaleActivation(Activation):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, features: FeatureDict) -> FeatureDict:
        output = FeatureDict()
        for partition, tensor in features.flat_items():
            output.set(partition, self.scale * tensor)
        return output


class PairRampGate(nn.Module):
    def forward(self, features: FeatureDict) -> FeatureDict:
        gates = FeatureDict()
        for partition, tensor in features.flat_items():
            if partition.order == 2:
                values = torch.arange(tensor.numel(), dtype=tensor.dtype, device=tensor.device).reshape(tensor.shape)
                gates.set(partition, 1.0 + values / float(tensor.numel()))
            else:
                gates.set(partition, torch.ones_like(tensor))
        return gates


class AlphaRampGate(nn.Module):
    def forward(self, features: FeatureDict) -> FeatureDict:
        gates = FeatureDict()
        for partition, tensor in features.flat_items():
            if partition == ORDER3_TENSOR:
                alpha = torch.tensor([1.0, 3.0], dtype=tensor.dtype, device=tensor.device).view(1, 1, 1, 1, 1, 2, 1)
                beta = torch.tensor([10.0, 20.0], dtype=tensor.dtype, device=tensor.device).view(1, 1, 1, 1, 1, 1, 2)
                gates.set(partition, alpha + beta)
            else:
                gates.set(partition, torch.ones_like(tensor))
        return gates


class ScalarBroadcastGate(nn.Module):
    def forward(self, features: FeatureDict) -> FeatureDict:
        gates = FeatureDict()
        for partition, tensor in features.flat_items():
            gates.set(partition, torch.tensor(0.5, dtype=tensor.dtype, device=tensor.device))
        return gates


class BadShapeGate(nn.Module):
    def forward(self, features: FeatureDict) -> FeatureDict:
        return FeatureDict(
            {partition: torch.ones(2, 3, dtype=tensor.dtype) for partition, tensor in features.flat_items()}
        )


class MissingGate(nn.Module):
    def forward(self, features: FeatureDict) -> FeatureDict:
        return FeatureDict()


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


def test_activation_template_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Activation.forward"):
        Activation()(_features())


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


@pytest.mark.parametrize("normalize", [True, False])
def test_norm_gate_activate_is_equivariant(normalize: bool) -> None:
    _check_gate_equivariance(NormGateActivate(nn.Tanh(), eps=1.0e-12, normalize=normalize), _features())


def test_norm_gate_activate_normalize_true_matches_expected_local_norm_gate() -> None:
    tensor = _features().get(ORDER3_TENSOR)
    features = FeatureDict({ORDER3_TENSOR: tensor})

    gates = NormGateActivate(nn.Tanh(), eps=1.0e-12, normalize=True)(features)

    norm = torch.linalg.vector_norm(tensor, dim=-2, keepdim=True)
    expected = torch.tanh(norm) / norm.clamp_min(1.0e-12)
    assert torch.allclose(gates.get(ORDER3_TENSOR), expected)


def test_norm_gate_activate_normalize_false_matches_expected_local_norm_gate() -> None:
    tensor = _features().get(ORDER3_TENSOR)
    features = FeatureDict({ORDER3_TENSOR: tensor})

    gates = NormGateActivate(nn.Tanh(), eps=1.0e-12, normalize=False)(features)

    norm = torch.linalg.vector_norm(tensor, dim=-2, keepdim=True)
    assert torch.allclose(gates.get(ORDER3_TENSOR), torch.tanh(norm))


def test_norm_gate_activate_normalize_false_docstring_warns_about_zero_origin_smoothness() -> None:
    assert "activation(0) == 0" in (NormGateActivate.__doc__ or "")


@pytest.mark.parametrize("normalize", [True, False])
def test_gated_activation_with_norm_gate_is_equivariant_and_preserves_contracts(normalize: bool) -> None:
    module = _norm_activation(normalize=normalize)
    features = _features()

    activated = module(features)

    _check_gate_equivariance(module, features)
    _assert_pair_symmetry_contract(activated)
    for partition, tensor in features.flat_items():
        value = activated.get(partition)
        assert value.shape == tensor.shape
        assert value.dtype == tensor.dtype
        assert value.device == tensor.device


def test_gated_activation_projects_non_symmetric_pair_gates_to_permutation_blocks() -> None:
    features = FeatureDict({ORDER2_SYM: _features().get(ORDER2_SYM), ORDER2_SIGN: _features().get(ORDER2_SIGN)})

    activated = GatedActivation(PairRampGate())(features)

    _assert_pair_symmetry_contract(activated)
    ramp = PairRampGate()(features).get(ORDER2_SYM)
    expected_gate = 0.5 * (ramp + ramp.transpose(2, 3))
    assert torch.allclose(activated.get(ORDER2_SYM), expected_gate * features.get(ORDER2_SYM))
    assert torch.allclose(activated.get(ORDER2_SIGN), expected_gate * features.get(ORDER2_SIGN))


def test_gated_activation_projects_mixed_irrep_gates_over_alpha_not_beta() -> None:
    tensor = _features().get(ORDER3_TENSOR)
    features = FeatureDict({ORDER3_TENSOR: tensor})

    activated = GatedActivation(AlphaRampGate())(features)

    beta_gate = torch.tensor([12.0, 22.0], dtype=tensor.dtype, device=tensor.device).view(1, 1, 1, 1, 1, 1, 2)
    expected_gate = beta_gate.expand_as(tensor)
    assert torch.allclose(activated.get(ORDER3_TENSOR), expected_gate * tensor)


def test_gated_activation_accepts_rank_reduced_broadcastable_gates() -> None:
    features = _features()

    activated = GatedActivation(ScalarBroadcastGate())(features)

    for partition, tensor in features.flat_items():
        assert torch.allclose(activated.get(partition), 0.5 * tensor)


def test_gated_activation_rejects_missing_gate_key() -> None:
    with pytest.raises(KeyError, match="Missing gate"):
        GatedActivation(MissingGate())(_features())


def test_gated_activation_rejects_invalid_gate_shape() -> None:
    with pytest.raises(ValueError, match="must broadcast"):
        GatedActivation(BadShapeGate())(_features())


def test_activation_by_type_with_gated_norm_activations_is_equivariant() -> None:
    module = ActivationByType(
        symmetric=_norm_activation(),
        antisymmetric=_norm_activation(),
        tensor=_norm_activation(),
    )

    _check_gate_equivariance(module, _features())
    _assert_pair_symmetry_contract(module(_features()))


def test_activation_by_irrep_with_gated_norm_activations_is_equivariant() -> None:
    module = ActivationByIrrep(
        {
            ORDER1: _norm_activation(),
            ORDER2_SYM: _norm_activation(),
            ORDER2_SIGN: _norm_activation(),
            ORDER3_TENSOR: _norm_activation(),
        }
    )

    _check_gate_equivariance(module, _features())
    _assert_pair_symmetry_contract(module(_features()))


def test_gate_templates_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="GateUpdate.forward"):
        GateUpdate(nn.Identity())(_scalar_features(1.0), _scalar_features(2.0))
    with pytest.raises(NotImplementedError, match="GateActivate.forward"):
        GateActivate(nn.Identity())(_scalar_features(1.0))


def test_scalar_gate_update_is_equivariant_and_uses_scalar_update_component() -> None:
    gate = ScalarGateUpdate(nn.Identity())
    features = _scalar_features(1.0)
    updates = _scalar_features(2.0)

    original = gate(features, updates)
    for permutation in PERMUTATIONS:
        transformed = gate(_permute_features(features, permutation), _permute_features(updates, permutation))
        expected = _permute_features(original, permutation)
        _assert_featuredict_close(transformed, expected)
    assert torch.equal(original.get(ORDER1), updates.get(ORDER1))


def test_scalar_gate_activate_is_equivariant_and_uses_scalar_feature_component() -> None:
    gate = ScalarGateActivate(nn.Identity())

    _check_gate_equivariance(gate, _scalar_features(2.0))
    assert torch.equal(gate(_scalar_features(2.0)).get(ORDER1), _scalar_features(2.0).get(ORDER1))


def test_scalar_gates_reject_missing_scalar_component() -> None:
    with pytest.raises(KeyError, match="Missing scalar"):
        ScalarGateActivate(nn.Identity())(FeatureDict())
    with pytest.raises(KeyError, match="Missing scalar"):
        ScalarGateUpdate(nn.Identity())(_scalar_features(1.0), FeatureDict())


def test_removed_activation_classes_are_not_exported_or_used_in_configs() -> None:
    assert not hasattr(activation_module, "ElementwiseFeatureActivation")
    assert not hasattr(activation_module, "NormGateActivation")
    assert not hasattr(gate_module, "NormGateActivation")

    repo = Path(__file__).resolve().parents[3]
    stale_names = ("ElementwiseFeatureActivation", "NormGateActivation")
    search_roots = (
        repo / "configs",
        repo / "experiments" / "hooke" / "configs",
        repo / "tests" / "integration" / "artifacts" / "hooke",
    )
    for root in search_roots:
        for path in root.rglob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            for stale_name in stale_names:
                assert stale_name not in text, f"{stale_name} remains in {path}"


def test_norm_gate_activate_is_exported_from_gate_module() -> None:
    assert hasattr(gate_module, "NormGateActivate")
