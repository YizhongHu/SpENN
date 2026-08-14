"""Static contracts for the minimal all-electron helium study."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments" / "atomistic" / "he-v1"
TRAIN = STUDY / "configs" / "train.yaml"
EVAL = STUDY / "configs" / "eval.yaml"
REGISTRY = ROOT / "experiments" / "baselines" / "systems.yaml"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def _he_reference() -> float:
    systems = _load(REGISTRY)["systems"]
    helium = next(system for system in systems if system["id"] == "he_atom")
    return float(helium["reference_energy_hartree"])


def _assert_he_system(config: dict) -> None:
    system = config["system"]
    assert system["id"] == "he_atom"
    assert system["n_particles"] == 2
    assert system["spin"] == {"n_up": 1, "n_down": 1}
    assert system["nuclei"] == {"positions": [[0.0, 0.0, 0.0]], "charges": [2.0]}
    assert float(system["reference_energy"]) == _he_reference()


def _assert_factorized_model(config: dict) -> None:
    model = config["model"]
    assert "envelope" not in model
    nuclear = model["nuclear_envelope"]
    assert nuclear["_target_"] == "tpen.nn.NuclearFactorizedEnvelope"
    assert nuclear["nuclear_confinement"]["_target_"] == "tpen.nn.NuclearConfinement"
    regular = nuclear["regular_envelope"]
    assert regular["_target_"] == "tpen.nn.AdditiveEnvelope"
    assert [entry["_target_"] for entry in regular["envelopes"]] == ["tpen.nn.ElectronElectronCusp"]
    assert "GaussianConfinement" not in str(model)


def test_he_train_config_owns_batch_nuclear_context_and_coulomb_terms() -> None:
    config = _load(TRAIN)
    _assert_he_system(config)
    _assert_factorized_model(config)
    sampler = config["sampler"]
    assert sampler["nuclear_positions"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.positions}",
    }
    assert sampler["nuclear_charges"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.charges}",
    }
    assert set(config["hamiltonian_terms"]) == {"kinetic", "electron_nucleus", "electron_electron"}


def test_he_eval_config_restores_same_model_and_uses_mcmc_reference_energy() -> None:
    train = _load(TRAIN)
    evaluation = _load(EVAL)
    _assert_he_system(evaluation)
    _assert_factorized_model(evaluation)
    assert evaluation["model"] == train["model"]
    assert evaluation["load"]["strict"] is True
    task = evaluation["evaluation_tasks"]["mcmc_energy"]
    assert task["generator"]["_target_"] == "tpen.evaluation.generators.MCMCGenerator"
    reference = task["summaries"][-1]
    assert reference == {
        "_target_": "tpen.evaluation.summaries.ReferenceEnergySummary",
        "reference_energy": "${system.reference_energy}",
    }


def test_he_eval_invariant_tasks_use_mcmc_batches_that_preserve_nuclear_context() -> None:
    config = _load(EVAL)
    for name in ("full_model_antisymmetry", "trace_equivariance"):
        generator = config["evaluation_tasks"][name]["generator"]
        assert generator["_target_"] == "tpen.evaluation.generators.PermutationOrbitGenerator"
        assert generator["base_generator"]["_target_"] == "tpen.evaluation.generators.MCMCGenerator"


def test_he_train_targets_instantiate_and_consume_the_same_nuclear_context() -> None:
    """Resolve Hydra targets through one actual nuclear-context forward pass."""

    config = OmegaConf.load(TRAIN)
    model = instantiate(config.model)
    sampler = instantiate(config.sampler)
    terms = instantiate(config.hamiltonian_terms)
    assert sampler.nuclear_positions is not None
    assert sampler.nuclear_charges is not None
    assert torch.equal(sampler.nuclear_positions, torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))
    assert torch.equal(sampler.nuclear_charges, torch.tensor([2.0], dtype=torch.float64))
    batch = ElectronBatch(
        positions=torch.tensor([[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]], dtype=torch.float64),
        spins=torch.tensor([[1, -1]]),
        nuclear_positions=sampler.nuclear_positions,
        nuclear_charges=sampler.nuclear_charges,
    )
    output = model(batch)
    output.validate(batch_size=batch.batch_size)
    assert set(terms) == {"kinetic", "electron_nucleus", "electron_electron"}
