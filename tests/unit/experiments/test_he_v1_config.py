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


#: Reserved gate-spec block holding namespace bindings, stripped before the spec
#: reaches the gates. Spelled here rather than imported for the reason below.
_METRIC_NAMESPACE_SPEC_KEY = "metric_namespaces"


def _grid_thresholds() -> dict:
    """Return the production grid's gate thresholds, bindings removed.

    Deliberately reads the YAML instead of calling the study's
    ``collect.split_gate_spec``/``plan.load_grid_config``. THE STUDY MODULES ARE
    NOT SAFELY IMPORTABLE FROM HERE, and loading them by path is not sufficient:
    ``experiments/`` holds four ``collect.py`` and three ``plan.py``, and
    ``collect.py`` reaches its siblings through BARE imports (``import plan as
    plan_stage``), so whichever study populated ``sys.modules['plan']`` first
    wins for every study afterwards. A path-based loader fixes the name this
    file imports and cannot fix the names the loaded module imports.

    Measured, not assumed: at fff76631 this test passed alone and failed in the
    composed suite with ``AttributeError: module 'plan' has no attribute
    'load_grid_config'`` -- an error naming a function that demonstrably exists,
    which reads as a broken function rather than a wrong module.

    The study-local suite still exercises the real ``split_gate_spec`` against
    this same file, so the parsing contract is covered where it can be imported
    safely; the repo-wide bare-sibling-import defect is filed separately.
    """

    payload = _load(STUDY / "configs" / "production_grid.yaml")
    spec = dict(payload["gate_spec"])
    spec.pop(_METRIC_NAMESPACE_SPEC_KEY, None)
    return spec


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
    assert task["generator"]["_target_"] == "tpen.evaluation.generators.TrajectoryMCMCGenerator"
    assert task["summaries"][0]["_target_"] == "tpen.evaluation.summaries.LocalEnergySummary"
    targets = [summary["_target_"] for summary in task["summaries"]]
    assert "tpen.evaluation.summaries.ReferenceEnergySummary" in targets
    reference = next(
        summary
        for summary in task["summaries"]
        if summary["_target_"] == "tpen.evaluation.summaries.ReferenceEnergySummary"
    )
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


def test_he_eval_wires_correlation_aware_chain_statistics() -> None:
    """A6-C is wired, inverting the pre-H-F1 "explicitly unwired" contract.

    This assertion used to require that no external chain statistics appeared,
    because no fixed-model trajectory producer existed. The producer merged and
    then sat unreferenced for a day: "A6-C done" did NOT mean "He runs have
    MCSE", and a He run at dev tip emitted none at all. H-F1 wires it, so the
    contract flips from "must be absent" to "must be present and joinable".
    """

    config = _load(EVAL)
    task = config["evaluation_tasks"]["mcmc_energy"]
    summaries = {summary["_target_"]: summary for summary in task["summaries"]}

    trajectory = summaries["tpen.evaluation.summaries.TrajectoryStatisticsSummary"]
    # The join identity admits no blanks: every field is supplied, and each
    # resolves to a `???` the driver must override rather than to a default.
    identity = config["trajectory_identity"]
    for field in ("stage", "run_id", "attempt_id", "evaluator_id", "checkpoint_file", "config_sha256"):
        assert identity[field] == "???", field
        assert trajectory[field.replace("checkpoint_file", "checkpoint_path")] == (
            "${trajectory_identity." + field + "}"
        )

    # The checkpoint FILE and the restore DIRECTORY are different paths on
    # purpose. Passing the directory where the file is wanted raises
    # IsADirectoryError inside the summary, after the chain has been sampled.
    assert trajectory["checkpoint_path"] == "${trajectory_identity.checkpoint_file}"
    assert config["load"]["path"] == "???"

    # The generator must be the trajectory one: only it publishes the explicit
    # [draw, walker] ObservableTrajectory the producer consumes.
    assert task["generator"]["_target_"] == "tpen.evaluation.generators.TrajectoryMCMCGenerator"
    assert int(task["generator"]["n_draws"]) >= 128


def test_he_eval_energy_task_keeps_the_calculator_its_summaries_consume() -> None:
    """`calculators: []` fails at the SUMMARY stage, after all sampling is paid.

    That defect cannot fail early by construction -- it must first do every
    expensive thing correctly and then die last -- and it cost this lane about
    three GPU-hours. Pinned here so the empty list cannot come back.
    """

    task = _load(EVAL)["evaluation_tasks"]["mcmc_energy"]
    calculators = [entry["_target_"] for entry in task["calculators"]]
    assert "tpen.evaluation.calculators.LocalEnergyCalculator" in calculators
    summaries = [entry["_target_"] for entry in task["summaries"]]
    assert "tpen.evaluation.summaries.LocalEnergySummary" in summaries


def test_he_eval_reports_sampler_health_and_retains_raw_records() -> None:
    """Two capabilities that existed, were exported, and were never named.

    `production-grid-v0` requires MCMC health beside every evaluation and raw
    [draw, walker] retention. Both were satisfiable by config alone and neither
    was configured, so an eval row carried no acceptance rate and a
    records-level run retained nothing.
    """

    config = _load(EVAL)
    task = config["evaluation_tasks"]["mcmc_energy"]
    summaries = {entry["_target_"]: entry for entry in task["summaries"]}

    assert "tpen.evaluation.summaries.SamplerStatsSummary" in summaries

    # artifact_level ALONE retains nothing: every raw-record writer returns
    # early unless the level is `records`, but each only runs if it is in a
    # task's summaries list. Setting the level without the writer satisfies the
    # contract's letter and keeps no records at all.
    assert config["evaluation"]["artifact_level"] == "records"
    writer = summaries["tpen.evaluation.summaries.SampledRecordWriter"]

    # max_samples must cover the whole [draw, walker] product. The default
    # 100000 would silently keep 9.5% of it and look entirely healthy.
    draws = int(task["generator"]["n_draws"])
    walkers = int(config["evaluation_sampler"]["n_walkers"])
    assert int(writer["max_samples"]) >= draws * walkers


def test_he_eval_sampler_carries_the_predeclared_stride_and_burn_in() -> None:
    sampler = _load(EVAL)["evaluation_sampler"]
    # Stride 20 is a DECLARED TIE-BREAK on MCSE and inflation, not a throughput
    # winner: ESS/second could not distinguish 10 from 20 at the measured noise
    # floor. Burn-in 100 was demonstrated sufficient ON THE MEAN.
    assert sampler["n_steps"] == 20
    assert sampler["burn_in"] == 100
    assert sampler["proposal_scale"] == 0.5


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


def test_he_production_width_reproduces_the_calibrated_mapping() -> None:
    """The measured capacity numbers describe THIS model, not a similar one.

    H-C1 measured 38921.1 MB and 1.7107 steps/s for "c32" using its own
    `apply_width` mapping. A different spelling of c32 -- most easily by leaving
    `hidden_channels` behind -- would leave those numbers perfectly true about
    H-C1's model and perfectly false about the production arm, with no test
    catching it and the capacity argument still reading as fully cited.
    """

    channels = 32
    for config in (_load(TRAIN), _load(EVAL)):
        model = config["model"]
        assert model["embedding"]["out_channels"] == channels
        # The one knob most easily forgotten. `hidden_channels = channels * 4`
        # is the mapping H-C1 actually ran; the pre-production 4/16 pair was
        # already this same ratio.
        assert model["embedding"]["hidden_channels"] == channels * 4
        assert model["readout"]["channels"] == channels
        for layer in model["layers"]:
            assert layer["mixing"]["channels"] == channels
            assert layer["path_aggregation"]["channels"] == channels


def test_he_configs_enable_both_trainable_cusp_ranges_identically() -> None:
    train = _load(TRAIN)["model"]["factors"]
    evaluation = _load(EVAL)["model"]["factors"]
    # Enabling the trainable law adds exactly two state-dict keys, so a strict
    # model-only restore fails in BOTH directions across the change. A
    # train/eval mismatch is not a degraded run, it is a run that cannot start.
    assert train == evaluation

    ee = next(f for f in train if f["_target_"] == "tpen.nn.ElectronElectronCusp")
    assert ee["trainable_range"] is True

    en = next(f for f in train if f["_target_"] == "tpen.nn.ElectronNucleusCusp")
    law = en["law"]
    assert law["_target_"] == "tpen.nn.TrainableCurvatureElectronNucleusCuspLaw"
    assert law["trainable"] is True
    # NONZERO on purpose. At exactly c = 0 the gradient with respect to the
    # range parameter d is identically zero, so a defaults-instantiated
    # trainable range cannot move on step one -- and the law's own defaults
    # (trainable=True, curvature_coefficient=0.0) land in that trap.
    assert float(law["curvature_coefficient"]) != 0.0
    assert float(law["curvature_range"]) > 0.0


def test_he_tail_band_brackets_the_law_s_own_outer_tail_slope() -> None:
    """The tail gate is centered on `outer_tail_slope`, never on a hardcoded -Z.

    At c = 0 the law is exactly `-Z r` and the expected slope is -Z. The moment
    c != 0 it is `-Z + c/d`, so a band calibrated against the pure linear law
    must not be applied unchanged. This asserts the predeclared band actually
    contains the value the configured law returns, rather than a value someone
    assumed it returns.
    """

    thresholds = _grid_thresholds()

    law = instantiate(OmegaConf.create(_load(TRAIN)["model"]["factors"][1]["law"]))
    charges = torch.tensor(_load(TRAIN)["system"]["nuclei"]["charges"], dtype=torch.float64)
    expected = float(law.outer_tail_slope(charges).reshape(-1)[0].item())

    # The configured initialization must sit inside the band the gate declares,
    # or the gate would fail a correctly initialized model on step one.
    assert thresholds["tail_outer_slope_mean_min"] <= expected
    assert expected <= thresholds["tail_outer_slope_mean_max"]
    # And it must NOT be -Z: asserting that keeps this test honest about the
    # fact that the curved law shifted the center at all.
    assert expected != -float(charges[0].item())
    assert abs(expected - (-float(charges[0].item()))) < 0.1

    # A decaying tail requires c/d < Z. The law does not enforce it; the gate's
    # sign requirement is the executable proxy, so it must demand every ray.
    assert thresholds["tail_negative_slope_fraction_min"] == 1.0


def test_trainable_law_breaks_strict_restore_in_both_directions() -> None:
    """The predeclared checkpoint break, asserted rather than discovered.

    H-C2's pilot checkpoint was trained with the linear law, so it cannot be
    restored under the production config and its measured tau describes a
    different model. Pinning the break here means a future config change that
    silently reverts the law fails a test instead of failing a restore inside an
    allocation.
    """

    from tpen.nn import ElectronNucleusCusp, TrainableCurvatureElectronNucleusCuspLaw

    atoms_cfg = OmegaConf.create(_load(TRAIN)["atoms"])
    OmegaConf.update(atoms_cfg, "positions.data", _load(TRAIN)["system"]["nuclei"]["positions"])
    OmegaConf.update(atoms_cfg, "charges.data", _load(TRAIN)["system"]["nuclei"]["charges"])
    atoms = instantiate(atoms_cfg)

    linear = ElectronNucleusCusp(atoms)
    trainable = ElectronNucleusCusp(
        atoms,
        law=TrainableCurvatureElectronNucleusCuspLaw(
            curvature_coefficient=0.01, curvature_range=1.0, trainable=True
        ),
    )

    # Exactly two keys, and the default law contributes none -- which is why
    # A5's bit-identity guarantee holds until a config actually switches.
    assert linear.state_dict() == {}
    assert sorted(trainable.state_dict()) == [
        "law.raw_curvature_coefficient",
        "law.raw_curvature_range",
    ]

    import pytest  # noqa: PLC0415

    with pytest.raises(RuntimeError):
        linear.load_state_dict(trainable.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        trainable.load_state_dict(linear.state_dict(), strict=True)
