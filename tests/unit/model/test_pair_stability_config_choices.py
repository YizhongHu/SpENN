"""Tests that the pair-stability choice libraries resolve by scalar key (PR8.8)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

import spenn.config  # noqa: F401 - registers the basis_feature_dim resolver
from spenn.nn import GaussianActivation

ROOT = Path(__file__).resolve().parents[3]
PAIR_STABILITY = ROOT / "experiments" / "hooke" / "pair_stability" / "configs" / "pair_stability.yaml"

MAIN_ARCHITECTURES = {
    "raw_envelope": ("RawCoordinateBasis", None),
    "hermite_o2_envelope": ("HookeHermiteBasis", ("max_order", 2)),
    "hermite_o3_envelope": ("HookeHermiteBasis", ("max_order", 3)),
    "orbital_s1_envelope": ("HookeOrbitalBasis", ("max_shell", 1)),
    "orbital_s2_envelope": ("HookeOrbitalBasis", ("max_shell", 2)),
}

NORMALIZATION_MODES = {
    "N0": "none",
    "N1": "post_embedding",
    "N2": "post_feature_layer",
    "N3": "update",
    "N4": "pre_readout",
}


@pytest.mark.parametrize(("architecture", "expected"), MAIN_ARCHITECTURES.items())
def test_choice_library_resolves_selected_architecture(architecture: str, expected) -> None:
    target_suffix, hyperparam = expected
    cfg = OmegaConf.load(PAIR_STABILITY)
    cfg.run_parameters.architecture = architecture
    basis = OmegaConf.to_container(cfg.model.basis, resolve=True)
    assert str(basis["_target_"]).endswith(target_suffix)
    if hyperparam is not None:
        key, value = hyperparam
        assert basis[key] == value


@pytest.mark.parametrize("architecture", MAIN_ARCHITECTURES)
def test_main_architectures_resolve_gaussian_envelope(architecture: str) -> None:
    cfg = OmegaConf.load(PAIR_STABILITY)
    cfg.run_parameters.architecture = architecture
    envelope = OmegaConf.to_container(cfg.model.envelope, resolve=True)
    # AdditiveEnvelope of the architecture Gaussian envelope plus the constant cusp.
    targets = [str(component.get("_target_", "")) for component in envelope["envelopes"]]
    assert any(target.endswith("HookeGaussianEnvelope") for target in targets)
    assert any(target.endswith("ElectronElectronCusp") for target in targets)


@pytest.mark.parametrize(("normalization", "mode"), NORMALIZATION_MODES.items())
def test_choice_library_resolves_selected_normalization(normalization: str, mode: str) -> None:
    cfg = OmegaConf.load(PAIR_STABILITY)
    cfg.run_parameters.normalization = normalization
    feature_normalization = OmegaConf.to_container(cfg.model.feature_normalization, resolve=True)
    assert feature_normalization["mode"] == mode
    if mode == "none":
        assert feature_normalization["norm"] is None
    else:
        assert str(feature_normalization["norm"]["_target_"]).endswith("IrrepRMSNorm")


def test_embedding_in_features_tracks_selected_basis() -> None:
    cfg = OmegaConf.load(PAIR_STABILITY)
    # Hermite o3 in 3D with spin: 3 * (3 + 1) + 1 = 13.
    cfg.run_parameters.architecture = "hermite_o3_envelope"
    assert int(OmegaConf.select(cfg, "model.embedding.in_features")) == 13
    # Orbital s2 in 3D with spin: 3 * (2 + 1) + 1 = 10.
    cfg.run_parameters.architecture = "orbital_s2_envelope"
    assert int(OmegaConf.select(cfg, "model.embedding.in_features")) == 10


def test_default_run_parameters_are_scalar() -> None:
    cfg = OmegaConf.load(PAIR_STABILITY)
    params = OmegaConf.to_container(cfg.run_parameters, resolve=True)
    assert set(params) == {"architecture", "normalization", "lr", "channels", "seed"}


def test_gaussian_gate_activation_target_instantiates() -> None:
    cfg = OmegaConf.load(PAIR_STABILITY)
    cfg.model_params.gate_activation = "gaussian"

    gate = instantiate(cfg.model.layers[0].activation.gate)

    assert isinstance(gate, GaussianActivation)
