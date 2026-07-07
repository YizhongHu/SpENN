"""Tests for the top-level neural-network namespace."""

from __future__ import annotations

import spenn.nn as spenn_nn
from spenn.nn.activation import GaussianActivation, GatedNormActivation
from spenn.nn.coordinate_envelopes import GaussianCoordinateEnvelope, RealCoordinateEnvelope
from spenn.nn.initialization import SeededLinear, TorchInitializer
from spenn.nn.real_gates import RealGaussianNormGate, RealRMSGate
from spenn.nn.scalar_gates import GaussianDecayGate, RMSInverseGate, SigmoidGate, TanhGate
from spenn.nn.update import ResidualUpdate


def test_spenn_nn_namespace_keeps_baseline_activation_update_surface() -> None:
    assert spenn_nn.GatedNormActivation is GatedNormActivation
    assert spenn_nn.GaussianActivation is GaussianActivation
    assert spenn_nn.GaussianCoordinateEnvelope is GaussianCoordinateEnvelope
    assert spenn_nn.RealCoordinateEnvelope is RealCoordinateEnvelope
    assert spenn_nn.RealGaussianNormGate is RealGaussianNormGate
    assert spenn_nn.RealRMSGate is RealRMSGate
    assert spenn_nn.GaussianDecayGate is GaussianDecayGate
    assert spenn_nn.RMSInverseGate is RMSInverseGate
    assert spenn_nn.SigmoidGate is SigmoidGate
    assert spenn_nn.TanhGate is TanhGate
    assert spenn_nn.ResidualUpdate is ResidualUpdate
    assert spenn_nn.SeededLinear is SeededLinear
    assert spenn_nn.TorchInitializer is TorchInitializer
    assert not hasattr(spenn_nn, "ActivationByType")
    assert not hasattr(spenn_nn, "ActivationByIrrep")
    assert not hasattr(spenn_nn, "ChannelMappedUpdate")
    assert not hasattr(spenn_nn, "NormGatedUpdate")
    assert not hasattr(spenn_nn, "ReplaceUpdate")
