"""Tests for hard-coded M=2 SpechtMP middle stages."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data import FeatureDict, MessageDict, Par
from spenn.nn.utils.activations import ActivationByType, GatedActivation
from spenn.nn.spechtmp import MessageHead, SpechtMP, SpechtMPLayer, UpdateHead
from spenn.nn.utils.gate import NormGateActivate
from spenn.nn.utils.update import ResidualUpdate
from spenn.reps import BranchMap, FusionMap


def _features(dtype: torch.dtype = torch.float64) -> FeatureDict:
    h = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=dtype).unsqueeze(-1).unsqueeze(-1)
    s = torch.tensor(
        [[[[0.0, 4.0, 5.0], [4.0, 0.0, 6.0], [5.0, 6.0, 0.0]]]],
        dtype=dtype,
    ).unsqueeze(-1).unsqueeze(-1)
    a = torch.tensor(
        [[[[0.0, 7.0, 8.0], [-7.0, 0.0, 9.0], [-8.0, -9.0, 0.0]]]],
        dtype=dtype,
    ).unsqueeze(-1).unsqueeze(-1)
    return FeatureDict({Par("H"): h, Par("S"): s, Par("A"): a})


def _message_activation() -> ActivationByType:
    return ActivationByType(
        symmetric=GatedActivation(NormGateActivate(nn.Tanh(), normalize=True)),
        antisymmetric=GatedActivation(NormGateActivate(nn.Tanh(), normalize=True)),
        tensor=GatedActivation(NormGateActivate(nn.Tanh(), normalize=True)),
    )


def _spechtmp_stack() -> SpechtMP:
    return SpechtMP(
        layers=[
            SpechtMPLayer(
                fusion_map=FusionMap(M=2, M_virtual=2),
                message_head=MessageHead(M=2, M_virtual=2, channels=[0, 2, 2], activation=_message_activation()),
                branch_map=BranchMap(M=2, M_virtual=2),
                update_head=UpdateHead(M=2, channels=[0, 2, 2]),
                update=ResidualUpdate(),
            )
        ]
    )


def test_message_head_mixes_fusion_routes_and_linear_features() -> None:
    features = _features()
    products = FusionMap()(features)
    head = MessageHead(channels=[0, 2, 3], include_linear=True).to(dtype=torch.float64)

    messages = head(products, features=features)

    assert set(messages) == {Par("H"), Par("S"), Par("A")}
    assert messages.get(Par("H")).shape == (1, 2, 3, 1, 1)
    assert messages.get(Par("S")).shape == (1, 3, 3, 3, 1, 1)
    assert messages.get(Par("A")).shape == (1, 3, 3, 3, 1, 1)
    messages.validate(batch_size=1, n_electrons=3)
    assert messages.get(Par("H")).dtype == torch.float64

    loss = sum(tensor.sum() for _partition, tensor in messages.flat_items())
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_branch_map_emits_all_m2_subset_routes_with_expected_signs() -> None:
    messages = MessageDict(_features().to_dict())

    branches = BranchMap()(messages)

    assert {(target, source) for target, source, _tensor in branches.flat_items()} == {
        (Par("H"), Par("H")),
        (Par("H"), Par("S")),
        (Par("H"), Par("A")),
        (Par("S"), Par("S")),
        (Par("A"), Par("A")),
    }
    branches.validate(batch_size=1, n_electrons=3)
    h_from_h = branches.get(Par("H"), Par("H"))
    assert h_from_h[0, 0, 0, 1, 1, 0, 0] == messages.get(Par("H"))[0, 0, 1, 0, 0]
    assert h_from_h[0, 0, 0, 1, 2, 0, 0] == 0

    h_from_s = branches.get(Par("H"), Par("S"))
    h_from_a = branches.get(Par("H"), Par("A"))
    assert h_from_s[0, 0, 0, 0, 0, 1, 0, 0] == 4.0
    assert h_from_s[0, 0, 0, 1, 0, 1, 0, 0] == 4.0
    assert h_from_a[0, 0, 0, 0, 0, 1, 0, 0] == 7.0
    assert h_from_a[0, 0, 0, 1, 0, 1, 0, 0] == -7.0


def test_update_head_mixes_branch_routes_into_feature_updates() -> None:
    branches = BranchMap()(MessageDict(_features().to_dict()))
    head = UpdateHead(channels=[0, 4, 5]).to(dtype=torch.float64)

    updates = head(branches)

    assert set(updates) == {Par("H"), Par("S"), Par("A")}
    assert updates.get(Par("H")).shape == (1, 4, 3, 1, 1)
    assert updates.get(Par("S")).shape == (1, 5, 3, 3, 1, 1)
    assert updates.get(Par("A")).shape == (1, 5, 3, 3, 1, 1)
    updates.validate(batch_size=1, n_electrons=3)
    assert updates.get(Par("S")).dtype == torch.float64

    loss = sum(tensor.sum() for _partition, tensor in updates.flat_items())
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_spechtmp_runs_fusion_message_branch_update_pipeline() -> None:
    features = _features()
    stack = _spechtmp_stack().to(dtype=torch.float64)

    output = stack(features)

    assert set(output) == {Par("H"), Par("S"), Par("A")}
    assert output.get(Par("H")).shape == (1, 2, 3, 1, 1)
    assert output.get(Par("S")).shape == (1, 2, 3, 3, 1, 1)
    assert output.get(Par("A")).shape == (1, 2, 3, 3, 1, 1)
    output.validate(batch_size=1, n_electrons=3)
