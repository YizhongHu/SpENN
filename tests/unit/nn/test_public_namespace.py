"""Tests for the top-level neural-network namespace."""

from __future__ import annotations

import spenn.nn as spenn_nn
from spenn.nn.activation import GatedNormActivation
from spenn.nn.update import ResidualUpdate


def test_spenn_nn_namespace_keeps_baseline_activation_update_surface() -> None:
    assert spenn_nn.GatedNormActivation is GatedNormActivation
    assert spenn_nn.ResidualUpdate is ResidualUpdate
    assert not hasattr(spenn_nn, "ActivationByType")
    assert not hasattr(spenn_nn, "ActivationByIrrep")
    assert not hasattr(spenn_nn, "ChannelMappedUpdate")
    assert not hasattr(spenn_nn, "NormGatedUpdate")
    assert not hasattr(spenn_nn, "ReplaceUpdate")
