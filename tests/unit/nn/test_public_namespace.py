"""Tests for the top-level neural-network namespace."""

from __future__ import annotations

import tpen.nn as spenn_nn
from tpen.nn.coordinate_envelopes import GaussianCoordinateEnvelope, GaussianDecayGate
from tpen.nn.initialization import SeededLinear, TorchInitializer
from tpen.nn.tpen_stack import TPENStack
from tpen.nn.update import ResidualUpdater


def test_spenn_nn_namespace_keeps_baseline_surface() -> None:
    assert spenn_nn.GaussianCoordinateEnvelope is GaussianCoordinateEnvelope
    assert spenn_nn.GaussianDecayGate is GaussianDecayGate
    assert spenn_nn.ResidualUpdater is ResidualUpdater
    assert spenn_nn.SeededLinear is SeededLinear
    assert spenn_nn.TorchInitializer is TorchInitializer
    assert spenn_nn.TPENStack is TPENStack
    assert not hasattr(spenn_nn, "ActivationByType")
    assert not hasattr(spenn_nn, "ActivationByIrrep")
    assert not hasattr(spenn_nn, "ChannelMappedUpdater")
    assert not hasattr(spenn_nn, "NormGatedUpdater")
    assert not hasattr(spenn_nn, "ReplaceUpdater")
