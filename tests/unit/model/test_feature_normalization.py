"""Unit tests for feature normalization modes and wiring (PR8.8)."""

from __future__ import annotations

import copy
import types

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from spenn.data.batch import ElectronBatch
from spenn.data.real import RealFeature, zero_block
from spenn.nn import FeatureNormalization, IrrepRMSNorm
from spenn.nn.normalization import FEATURE_NORMALIZATION_MODES
from tests.helpers.equivariance import assert_equivariant_all

NUM_LAYERS = 2


def _batch() -> ElectronBatch:
    generator = torch.Generator().manual_seed(99)
    positions = torch.randn(3, 4, 3, generator=generator, dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0, 1.0, -1.0]] * 3, dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)


def _model_config(mode: str) -> dict:
    """Return a tiny SpENN model config with the given normalization mode."""

    norm = None if mode == "none" else {"_target_": "spenn.nn.IrrepRMSNorm", "eps": 1.0e-8}
    layer = {
        "_target_": "spenn.nn.SpENNLayer",
        "mixing": {
            "_target_": "spenn.nn.EquivariantMixing",
            "max_order": 2,
            "max_virtual_order": 2,
            "channels": 4,
            "implementation": "slow",
            "initial_weight": 0.5,
        },
        "fourier": {"_target_": "spenn.reps.FourierTransform"},
        "activation": {"_target_": "spenn.nn.GatedNormActivation", "gate": {"_target_": "torch.nn.SiLU"}},
        "path_aggregation": {
            "_target_": "spenn.nn.PathAggregation",
            "max_order": 2,
            "max_virtual_order": 2,
            "channels": 4,
        },
        "inverse_fourier": {"_target_": "spenn.reps.InverseFourierTransform"},
        "update": {"_target_": "spenn.nn.ResidualUpdate"},
    }
    config: dict = {
        "_target_": "spenn.nn.SpENNWaveFunction",
        "trace_name": "spenn",
        "embedding": {
            "_target_": "spenn.nn.Embedding",
            "max_order": 2,
            "spatial_dim": 3,
            "out_channels": 4,
            "hidden_channels": 8,
            "num_hidden_layers": 1,
            "include_spins": True,
        },
        "layers": [copy.deepcopy(layer) for _ in range(NUM_LAYERS)],
        "envelope": {"_target_": "spenn.nn.HookeGaussianEnvelope", "omega": 0.5},
        "readout": {"_target_": "spenn.nn.readout.PfaffianReadout", "channels": 4},
    }
    config["feature_normalization"] = {
        "_target_": "spenn.nn.FeatureNormalization",
        "mode": mode,
        "norm": norm,
    }
    return config


def _build(mode: str):
    return instantiate(OmegaConf.create(_model_config(mode))).to(dtype=torch.float64)


def test_feature_normalization_modes_instantiate() -> None:
    for mode in FEATURE_NORMALIZATION_MODES:
        norm = None if mode == "none" else IrrepRMSNorm(eps=1.0e-8)
        choice = FeatureNormalization(mode=mode, norm=norm)
        assert choice.mode == mode


def test_non_none_mode_requires_norm_module() -> None:
    with pytest.raises(ValueError):
        FeatureNormalization(mode="update", norm=None)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        FeatureNormalization(mode="not_a_mode", norm=IrrepRMSNorm(eps=1.0e-8))


def test_irrep_rms_norm_is_particle_equivariant() -> None:
    feature = RealFeature(
        [
            zero_block(batch_size=3, dtype=torch.float64),
            torch.randn(3, 4, 4, dtype=torch.float64),
            torch.randn(3, 4, 4, 4, dtype=torch.float64),
        ]
    )
    assert_equivariant_all(IrrepRMSNorm(eps=1.0e-8), feature)


def test_N0_is_noop() -> None:
    none_model = _build("none")
    assert none_model.feature_normalization.mode == "none"
    assert none_model.feature_normalization.norm is None
    assert all(layer.update_norm is None for layer in none_model.layers)

    # A model with no feature_normalization at all must produce identical output.
    bare_config = _model_config("none")
    bare_config.pop("feature_normalization")
    bare_model = instantiate(OmegaConf.create(bare_config)).to(dtype=torch.float64)
    bare_model.load_state_dict(none_model.state_dict())

    batch = _batch()
    torch.testing.assert_close(none_model(batch).logabs, bare_model(batch).logabs)


@pytest.mark.parametrize("mode", ["post_embedding", "post_feature_layer", "update", "pre_readout"])
def test_N1_N2_N3_N4_forward_smoke(mode: str) -> None:
    model = _build(mode)
    output = model(_batch())
    assert torch.isfinite(output.logabs).all()
    assert output.logabs.shape == (3,)


def test_update_mode_is_wired_into_every_layer() -> None:
    model = _build("update")
    norm = model.feature_normalization.norm
    assert norm is not None
    # N3 normalizes the per-layer update increment, so each layer hosts the norm.
    assert all(layer.update_norm is norm for layer in model.layers)


@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    [
        ("post_embedding", 1),  # N1: once after embedding
        ("post_feature_layer", NUM_LAYERS),  # N2: once after each feature layer
        ("update", NUM_LAYERS),  # N3: once per layer update increment
        ("pre_readout", 1),  # N4: once before readout
    ],
)
def test_normalization_runs_at_the_intended_site(mode: str, expected_calls: int) -> None:
    model = _build(mode)
    norm = model.feature_normalization.norm
    calls = {"count": 0}
    original = norm.forward_impl

    def counting(self, features):  # noqa: ANN001 - patched bound method
        calls["count"] += 1
        return original(features)

    norm.forward_impl = types.MethodType(counting, norm)
    model(_batch())
    assert calls["count"] == expected_calls
