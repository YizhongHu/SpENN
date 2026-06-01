"""Encoder, fixed-map, and SpechtMP equivariance tests."""

from __future__ import annotations

import pytest
import torch

from spenn.data import FeatureDict, MessageDict, Par
from spenn.data.batch import ElectronBatch
from spenn.nn.utils.activations import ActivationByType, GatedActivation
from spenn.nn.encoding import ElectronPairEncoder
from spenn.nn.spechtmp import MessageHead, SpechtMP, SpechtMPLayer, UpdateHead
from spenn.nn.utils.gate import NormGateActivate
from spenn.nn.utils.update import ResidualUpdate
from spenn.reps import BranchMap, FusionMap


ORDER1 = Par("H")
ORDER2_SYM = Par("S")
ORDER2_SIGN = Par("A")


def _batch() -> ElectronBatch:
    return ElectronBatch(
        positions=torch.tensor(
            [
                [[0.0, 1.0], [2.0, -1.0], [4.0, 0.5]],
                [[1.0, 0.0], [-0.5, 2.0], [3.0, -1.5]],
            ],
            dtype=torch.float64,
        )
    )


def _features() -> FeatureDict:
    h = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
    s = torch.tensor(
        [[[[0.0, 4.0, 5.0], [4.0, 0.0, 6.0], [5.0, 6.0, 0.0]]]],
        dtype=torch.float64,
    ).unsqueeze(-1).unsqueeze(-1)
    a = torch.tensor(
        [[[[0.0, 7.0, 8.0], [-7.0, 0.0, 9.0], [-8.0, -9.0, 0.0]]]],
        dtype=torch.float64,
    ).unsqueeze(-1).unsqueeze(-1)
    return FeatureDict({ORDER1: h, ORDER2_SYM: s, ORDER2_SIGN: a})


def _assert_pair_symmetry(features: FeatureDict) -> None:
    symmetric = features.get(ORDER2_SYM)
    antisymmetric = features.get(ORDER2_SIGN)
    assert torch.allclose(symmetric, symmetric.transpose(2, 3))
    assert torch.allclose(antisymmetric, -antisymmetric.transpose(2, 3))
    assert torch.allclose(antisymmetric.diagonal(dim1=2, dim2=3), torch.zeros_like(antisymmetric.diagonal(dim1=2, dim2=3)))


def _permute_features(features: FeatureDict, permutation: torch.Tensor) -> FeatureDict:
    return FeatureDict(
        {
            ORDER1: features.get(ORDER1)[:, :, permutation],
            ORDER2_SYM: features.get(ORDER2_SYM)[:, :, permutation][:, :, :, permutation],
            ORDER2_SIGN: features.get(ORDER2_SIGN)[:, :, permutation][:, :, :, permutation],
        }
    )


def _message_activation() -> ActivationByType:
    activation = GatedActivation(NormGateActivate(torch.nn.Tanh(), normalize=True))
    return ActivationByType(
        symmetric=activation,
        antisymmetric=GatedActivation(NormGateActivate(torch.nn.Tanh(), normalize=True)),
        tensor=GatedActivation(NormGateActivate(torch.nn.Tanh(), normalize=True)),
    )


def _spechtmp_stack(num_layers: int) -> SpechtMP:
    return SpechtMP(
        layers=[
            SpechtMPLayer(
                fusion_map=FusionMap(M=2, M_virtual=2),
                message_head=MessageHead(M=2, M_virtual=2, channels=[0, 2, 2], activation=_message_activation()),
                branch_map=BranchMap(M=2, M_virtual=2),
                update_head=UpdateHead(M=2, channels=[0, 2, 2]),
                update=ResidualUpdate(),
            )
            for _ in range(num_layers)
        ]
    )


def _reduced_pair_values(tensor: torch.Tensor, target, *sources) -> torch.Tensor:
    source_start = 3 + target.order
    source_stop = source_start + sum(source.order for source in sources)
    reduced = tensor.sum(dim=tuple(range(source_start, source_stop)))
    return reduced[..., 0, 0]


def test_encoder_pair_features_have_phase1_symmetry_contract() -> None:
    features = ElectronPairEncoder(channels=[0, 2, 3])(_batch())

    _assert_pair_symmetry(features)


def test_spechtmp_stack_preserves_pair_symmetry_classes() -> None:
    torch.manual_seed(0)
    features = ElectronPairEncoder(channels=[0, 2, 2])(_batch())
    stack = _spechtmp_stack(num_layers=2).to(dtype=torch.float64)

    output = stack(features)

    _assert_pair_symmetry(output)


def test_spechtmp_stack_is_equivariant_under_fixed_transposition() -> None:
    torch.manual_seed(0)
    encoder = ElectronPairEncoder(channels=[0, 2, 2])
    stack = _spechtmp_stack(num_layers=1).to(dtype=torch.float64)
    batch = _batch()
    permutation = torch.tensor([1, 0, 2])

    original = stack(encoder(batch))
    transformed = stack(encoder(ElectronBatch(positions=batch.positions[:, permutation])))
    expected = _permute_features(original, permutation)

    assert torch.allclose(transformed.get(ORDER1), expected.get(ORDER1))
    assert torch.allclose(transformed.get(ORDER2_SYM), expected.get(ORDER2_SYM))
    assert torch.allclose(transformed.get(ORDER2_SIGN), expected.get(ORDER2_SIGN))


def test_fixed_maps_preserve_target_keys_shapes_dtype_and_reduced_pair_symmetry() -> None:
    features = _features()
    products = FusionMap()(features)
    branches = BranchMap()(MessageDict(features.to_dict()))

    products.validate(batch_size=1, n_electrons=3)
    branches.validate(batch_size=1, n_electrons=3)
    assert all(tensor.dtype == torch.float64 for *_keys, tensor in products.flat_items())
    assert all(tensor.dtype == torch.float64 for *_keys, tensor in branches.flat_items())

    for target, left, right, tensor in products.flat_items():
        if target == ORDER2_SYM:
            values = _reduced_pair_values(tensor, target, left, right)
            assert torch.allclose(values, values.transpose(-1, -2))
        if target == ORDER2_SIGN:
            values = _reduced_pair_values(tensor, target, left, right)
            assert torch.allclose(values, -values.transpose(-1, -2))

    for target, source, tensor in branches.flat_items():
        if target == ORDER2_SYM:
            values = _reduced_pair_values(tensor, target, source)
            assert torch.allclose(values, values.transpose(-1, -2))
        if target == ORDER2_SIGN:
            values = _reduced_pair_values(tensor, target, source)
            assert torch.allclose(values, -values.transpose(-1, -2))


def test_phase1_boundaries_reject_orders_above_m2() -> None:
    with pytest.raises(ValueError, match="max_order <= 2"):
        ElectronPairEncoder(max_order=3)
    with pytest.raises(ValueError, match="M <= 2"):
        FusionMap(M=3)
    with pytest.raises(ValueError, match="M <= 2"):
        BranchMap(M_virtual=3)
    with pytest.raises(ValueError, match="M <= 2"):
        MessageHead(M=3)
    with pytest.raises(ValueError, match="M <= 2"):
        UpdateHead(M=3)
