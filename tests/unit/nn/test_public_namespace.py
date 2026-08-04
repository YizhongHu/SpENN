"""Tests for the top-level neural-network namespace."""

from __future__ import annotations

import spenn.nn as spenn_nn
from spenn.nn.coordinate_envelopes import GaussianCoordinateEnvelope, GaussianDecayGate, RealCoordinateEnvelope
from spenn.nn.initialization import SeededLinear, TorchInitializer
from spenn.nn.tpen_stack import TPENStack
from spenn.nn.update import ResidualUpdate


def test_spenn_nn_namespace_keeps_baseline_surface() -> None:
    assert spenn_nn.GaussianCoordinateEnvelope is GaussianCoordinateEnvelope
    assert spenn_nn.RealCoordinateEnvelope is RealCoordinateEnvelope
    assert spenn_nn.GaussianDecayGate is GaussianDecayGate
    assert spenn_nn.ResidualUpdate is ResidualUpdate
    assert spenn_nn.SeededLinear is SeededLinear
    assert spenn_nn.TorchInitializer is TorchInitializer
    assert spenn_nn.TPENStack is TPENStack
    assert not hasattr(spenn_nn, "ActivationByType")
    assert not hasattr(spenn_nn, "ActivationByIrrep")
    assert not hasattr(spenn_nn, "ChannelMappedUpdate")
    assert not hasattr(spenn_nn, "NormGatedUpdate")
    assert not hasattr(spenn_nn, "ReplaceUpdate")
