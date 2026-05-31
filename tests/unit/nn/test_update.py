"""Tests for feature-state update modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data import FeatureDict, Par, Partition
from spenn.nn.update import CompositeUpdate, GatedUpdate, RawUpdate, ResidualUpdate, Update, UpdateByIrrep, UpdateByType


ORDER1 = Par("H")
ORDER2_SYM = Par("S")
ORDER2_SIGN = Par("A")
ORDER3_TENSOR = Par("V")


def _features() -> FeatureDict:
    return FeatureDict(
        {
            ORDER1: torch.ones(1, 1, 2, 1, 1),
            ORDER2_SYM: torch.ones(1, 1, 2, 2, 1, 1),
            ORDER2_SIGN: torch.ones(1, 1, 2, 2, 1, 1),
            ORDER3_TENSOR: torch.ones(1, 1, 2, 2, 2, 2, 2),
        }
    )


def _updates() -> FeatureDict:
    return FeatureDict(
        {
            ORDER1: 2.0 * torch.ones(1, 1, 2, 1, 1),
            ORDER2_SYM: 3.0 * torch.ones(1, 1, 2, 2, 1, 1),
            ORDER2_SIGN: 4.0 * torch.ones(1, 1, 2, 2, 1, 1),
            ORDER3_TENSOR: 5.0 * torch.ones(1, 1, 2, 2, 2, 2, 2),
        }
    )


class ConstantGate(nn.Module):
    def __init__(self, value: float, omit: Partition | None = None) -> None:
        super().__init__()
        self.value = value
        self.omit = omit

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        gates = FeatureDict()
        for partition, tensor in updates.flat_items():
            if partition != self.omit:
                gates.set(partition, torch.full_like(tensor, self.value))
        return gates


class ScaleUpdate(Update):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        scaled = FeatureDict()
        for partition, tensor in updates.flat_items():
            scaled.set(partition, self.scale * tensor)
        return scaled


def test_update_template_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Update.forward"):
        Update()(_features(), _updates())


def test_raw_and_residual_updates() -> None:
    features = _features()
    updates = _updates()

    assert RawUpdate()(features, updates) is updates
    residual = ResidualUpdate()(features, updates)
    assert torch.equal(residual.get(ORDER1), features.get(ORDER1) + updates.get(ORDER1))


def test_composite_update_chains_inner_then_outer_rule() -> None:
    features = _features()
    updates = _updates()

    composite = CompositeUpdate(first=ResidualUpdate(), second=ResidualUpdate())
    result = composite(features, updates)

    assert torch.equal(result.get(ORDER1), features.get(ORDER1) + 2.0 * updates.get(ORDER1))


def test_gated_update_applies_gate_delta_rule() -> None:
    features = _features()
    updates = _updates()

    gated = GatedUpdate(ConstantGate(0.25))(features, updates)

    assert torch.equal(gated.get(ORDER2_SYM), features.get(ORDER2_SYM) + 0.25 * updates.get(ORDER2_SYM))


def test_gated_update_rejects_missing_gate_key() -> None:
    with pytest.raises(KeyError, match="Missing gate"):
        GatedUpdate(ConstantGate(0.25, omit=ORDER2_SIGN))(_features(), _updates())


def test_update_by_type_routes_irreps_independently() -> None:
    routed = UpdateByType(
        symmetric=ScaleUpdate(10.0),
        antisymmetric=ScaleUpdate(20.0),
        tensor=ScaleUpdate(30.0),
    )(_features(), _updates())

    assert torch.equal(routed.get(ORDER1), 10.0 * _updates().get(ORDER1))
    assert torch.equal(routed.get(ORDER2_SYM), 10.0 * _updates().get(ORDER2_SYM))
    assert torch.equal(routed.get(ORDER2_SIGN), 20.0 * _updates().get(ORDER2_SIGN))
    assert torch.equal(routed.get(ORDER3_TENSOR), 30.0 * _updates().get(ORDER3_TENSOR))


def test_update_by_irrep_routes_exact_partition_keys() -> None:
    routed = UpdateByIrrep(
        {
            ORDER1: ScaleUpdate(2.0),
            ORDER2_SYM: ScaleUpdate(3.0),
            ORDER2_SIGN: ScaleUpdate(4.0),
            ORDER3_TENSOR: ScaleUpdate(5.0),
        }
    )(_features(), _updates())

    assert torch.equal(routed.get(ORDER1), 2.0 * _updates().get(ORDER1))
    assert torch.equal(routed.get(ORDER2_SYM), 3.0 * _updates().get(ORDER2_SYM))
    assert torch.equal(routed.get(ORDER2_SIGN), 4.0 * _updates().get(ORDER2_SIGN))
    assert torch.equal(routed.get(ORDER3_TENSOR), 5.0 * _updates().get(ORDER3_TENSOR))


def test_update_by_irrep_rejects_missing_partition() -> None:
    with pytest.raises(KeyError, match="Missing update module"):
        UpdateByIrrep({ORDER1: RawUpdate()})(_features(), _updates())
