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
    assert "nuclear_envelope" not in model
    factors = model["factors"]
    assert [entry["_target_"] for entry in factors] == [
        "tpen.nn.ElectronElectronCusp",
        "tpen.nn.ElectronNucleusCusp",
    ]
    assert factors[1]["atoms"] == "${atoms}"
    assert config["atoms"]["_target_"] == "tpen.data.atomic_configuration.AtomicConfiguration"
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
    hamiltonian_terms = config["hamiltonian_terms"]
    assert set(hamiltonian_terms) == {"kinetic", "electron_nucleus", "electron_electron", "nucleus_nucleus"}
    assert hamiltonian_terms["electron_nucleus"]["_target_"] == "tpen.physics.potential.ElectronNucleusPotential"
    assert hamiltonian_terms["electron_nucleus"]["atoms"] == "${atoms}"
    assert hamiltonian_terms["electron_nucleus"]["eps"] == 0.0
    assert hamiltonian_terms["nucleus_nucleus"] == {
        "_target_": "tpen.physics.potential.NucleusNucleusPotential",
        "atoms": "${atoms}",
    }


def test_he_eval_config_restores_same_model_and_uses_mcmc_reference_energy() -> None:
    train = _load(TRAIN)
    evaluation = _load(EVAL)
    _assert_he_system(evaluation)
    _assert_factorized_model(evaluation)
    assert evaluation["model"] == train["model"]
    assert evaluation["hamiltonian_terms"]["electron_nucleus"]["eps"] == 0.0
    assert evaluation["hamiltonian_terms"]["electron_nucleus"]["atoms"] == "${atoms}"
    assert evaluation["load"]["strict"] is True
    task = evaluation["evaluation_tasks"]["mcmc_energy"]
    assert task["generator"]["_target_"] == "tpen.evaluation.generators.MCMCGenerator"
    assert task["summaries"][0]["_target_"] == "tpen.evaluation.summaries.LocalEnergySummary"
    reference = task["summaries"][-1]
    assert reference == {
        "_target_": "tpen.evaluation.summaries.ReferenceEnergySummary",
        "reference_energy": "${system.reference_energy}",
    }


def test_he_eval_registers_atom_owned_radial_profiles_after_typed_contracts() -> None:
    config = _load(EVAL)
    tasks = config["evaluation_tasks"]
    radial = tasks["he_radial_profiles"]
    generator = radial["generator"]

    assert generator["_target_"] == "tpen.evaluation.generators.HeliumRadialGridGenerator"
    assert all(float(radius) > 0.0 for radius in generator["cusp_radii"])
    assert max(generator["cusp_radii"]) < min(generator["tail_radii"])
    assert generator["nuclear_positions"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.positions}",
    }
    assert generator["nuclear_charges"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.charges}",
    }
    assert radial["calculators"] == [
        {
            "_target_": "tpen.evaluation.calculators.ElectronNucleusRadialCalculator",
            "chunk_size": 32,
        }
    ]
    assert [summary["_target_"] for summary in radial["summaries"]] == [
        "tpen.evaluation.summaries.ElectronNucleusCuspSummary",
        "tpen.evaluation.summaries.ElectronNucleusTailSummary",
        "tpen.evaluation.summaries.ElectronNucleusRadialProfileWriter",
    ]
    assert "Hooke" not in str(radial)


def test_he_eval_separates_spatial_exchange_from_label_antisymmetry() -> None:
    config = _load(EVAL)
    tasks = config["evaluation_tasks"]
    spatial = tasks["spatial_exchange_symmetry"]

    assert spatial["generator"]["_target_"] == "tpen.evaluation.generators.ExchangeOrbitGenerator"
    assert spatial["generator"]["exchange"] == "opposite_spin_pair"
    assert spatial["calculators"] == [
        {"_target_": "tpen.evaluation.calculators.SpatialExchangeSymmetryCalculator"}
    ]
    assert tasks["full_model_antisymmetry"]["calculators"] == [
        {"_target_": "tpen.evaluation.calculators.FullModelAntisymmetryCalculator"}
    ]


def test_he_eval_external_chain_statistics_remain_explicitly_unwired() -> None:
    config = _load(EVAL)
    text = str(config)

    assert "tau_int" not in text
    assert "ess" not in text.lower()
    assert "external" not in text.lower()
    assert not any("Timing" in callback["_target_"] for callback in config["callbacks"])


def test_he_eval_invariant_tasks_use_mcmc_batches_that_preserve_nuclear_context() -> None:
    config = _load(EVAL)
    expected_generators = {
        "full_model_antisymmetry": "tpen.evaluation.generators.PermutationOrbitGenerator",
        "spatial_exchange_symmetry": "tpen.evaluation.generators.ExchangeOrbitGenerator",
        "trace_equivariance": "tpen.evaluation.generators.PermutationOrbitGenerator",
    }
    for name, expected_generator in expected_generators.items():
        generator = config["evaluation_tasks"][name]["generator"]
        assert generator["_target_"] == expected_generator
        assert generator["base_generator"]["_target_"] == "tpen.evaluation.generators.MCMCGenerator"


def test_he_train_targets_instantiate_and_consume_the_same_nuclear_context() -> None:
    """Resolve Hydra targets through one actual nuclear-context forward pass."""

    config = OmegaConf.load(TRAIN)
    model = instantiate(config.model)
    # Runner owns the configured floating-point policy.  Mirror that launch
    # boundary for this direct target-instantiation contract test.
    model.to(dtype=torch.float64)
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
    assert set(terms) == {"kinetic", "electron_nucleus", "electron_electron", "nucleus_nucleus"}
    from tpen.physics.hamiltonian import local_energy

    result = local_energy(terms, model, batch, return_terms=True)
    assert torch.equal(result.terms["nucleus_nucleus"], torch.zeros_like(result.total))

    # A5 ownership resolution: the wavefunction's ElectronNucleusCusp and the
    # Hamiltonian's ElectronNucleusPotential/NucleusNucleusPotential each
    # instantiate their own `AtomicConfiguration` from the same declarative
    # `${atoms}` config source -- Hydra's recursive `instantiate` re-runs a
    # `_target_` for every tree position it appears at, so this is NOT one
    # shared Python object. Assert both facts explicitly: the objects are
    # distinct, and no code path anywhere relies on them being identical --
    # every Hamiltonian term instead reconciles construction-time `atoms`
    # against transported batch context by explicit value comparison
    # (`_validate_batch_atoms_context`, exercised above via `local_energy`).
    en_cusp = next(f for f in model.factors if type(f).__name__ == "ElectronNucleusCusp")
    hamiltonian_en_atoms = terms["electron_nucleus"].atoms
    hamiltonian_nn_atoms = terms["nucleus_nucleus"].atoms
    assert en_cusp.atoms is not hamiltonian_en_atoms
    assert en_cusp.atoms is not hamiltonian_nn_atoms
    assert hamiltonian_en_atoms is not hamiltonian_nn_atoms
    assert en_cusp.atoms == hamiltonian_en_atoms
    assert en_cusp.atoms == hamiltonian_nn_atoms
