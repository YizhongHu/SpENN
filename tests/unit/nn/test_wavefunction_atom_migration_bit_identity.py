"""A5 bit-identity gate: pre/post He wavefunction+Hamiltonian migration.

This is a binding numerical gate, not an argument by inspection. It loads the
retired pre-A5 `tpen.nn.envelope`/`tpen.nn.spenn_wave_function` source
directly from git history (the A4 tip, immediately before this migration) and
instantiates the OLD `NuclearFactorizedEnvelope`/`nuclear_envelope`-based
`TPENWaveFunction` and the OLD batch-transported `ElectronNucleusInteraction`
Hamiltonian term side by side with the CURRENT generic
`factors`-pipeline/`ElectronNucleusCusp` wavefunction and the CURRENT
`AtomicConfiguration`-owned `ElectronNucleusPotential`/`NucleusNucleusPotential`
Hamiltonian terms, built from the real He `train.yaml` config. The embedding,
TPEN layers, and readout module objects are constructed once and *shared by
Python reference* between the old and new wrapper constructions, so the two
pipelines differ only in the migrated post-readout envelope/factor stage and
the migrated electron-nucleus Hamiltonian term -- isolating the comparison to
exactly what A5 changed. At fixed parameters/seed/dtype/spins/walker
coordinates, every representation-only migration must produce bit-identical
logabs/sign/phase and every individual local-energy term (and the total).
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.nn.envelope import ElectronElectronCusp, ElectronNucleusCusp
from tpen.nn.tpen_wave_function import TPENWaveFunction
from tpen.physics.hamiltonian import local_energy
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import (
    ElectronElectronInteraction,
    ElectronNucleusInteraction,
    ElectronNucleusPotential,
    NucleusNucleusPotential,
)

ROOT = Path(__file__).resolve().parents[3]
TRAIN_CONFIG = ROOT / "experiments" / "atomistic" / "he-v1" / "configs" / "train.yaml"
# A4 tip: the last commit before this A5 migration. `tpen/nn/envelope.py` and
# `tpen/nn/spenn_wave_function.py` at this SHA still carry the retired
# NuclearFactorizedEnvelope/nuclear_envelope mutual-exclusion path.
PRE_A5_SHA = "dc99163e15bdb7f2b4cb482e570ff0e9a0e9a8ec"


def _git_show(path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{PRE_A5_SHA}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _exec_module(source: str, *, register_name: str, filename: str) -> types.ModuleType:
    module = types.ModuleType(register_name)
    module.__file__ = filename
    sys.modules[register_name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def _load_pre_a5_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Load the retired pre-A5 envelope/wavefunction classes from git history.

    `tpen/nn/spenn_wave_function.py` at `PRE_A5_SHA` imports
    ``NuclearConfinementEvaluation``/``NuclearFactorizedEnvelope`` from
    ``tpen.nn.envelope`` by name, so the pre-A5 envelope module is registered
    under the real ``tpen.nn.envelope`` name only for the duration of that
    exec, then the current (post-A5) module is restored. Both loaded modules
    are returned so the test constructs old classes from exactly one exec of
    each retired file.
    """

    real_envelope_module = sys.modules.get("tpen.nn.envelope")
    try:
        old_envelope = _exec_module(
            _git_show("tpen/nn/envelope.py"),
            register_name="tpen.nn.envelope",
            filename=f"<git:{PRE_A5_SHA}:tpen/nn/envelope.py>",
        )
        old_wavefunction = _exec_module(
            _git_show("tpen/nn/spenn_wave_function.py"),
            register_name="tests._pre_a5_spenn_wave_function",
            filename=f"<git:{PRE_A5_SHA}:tpen/nn/spenn_wave_function.py>",
        )
    finally:
        if real_envelope_module is not None:
            sys.modules["tpen.nn.envelope"] = real_envelope_module
        else:
            sys.modules.pop("tpen.nn.envelope", None)
    return old_envelope, old_wavefunction


def _fixed_he_batch() -> ElectronBatch:
    # Fixed, non-trivial (non-coalescent, non-symmetric) walker coordinates so
    # the comparison exercises real electron-electron and electron-nucleus
    # separations rather than degenerate zeros.
    positions = torch.tensor(
        [
            [[0.3, 0.1, -0.2], [-0.4, 0.2, 0.5]],
            [[1.1, -0.3, 0.2], [0.05, 0.4, -0.6]],
            [[0.02, 0.0, 0.01], [0.9, -0.8, 0.3]],
        ],
        dtype=torch.float64,
    )
    spins = torch.tensor([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)
    nuclear_positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
    return ElectronBatch(
        positions=positions,
        spins=spins,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
    )


def _shared_trunk():
    """Build embedding/layers/readout once from the real He config.

    Returned module objects are shared *by Python reference* between the old
    and new wrapper models below, so any output difference is attributable
    only to the migrated envelope/factor and Hamiltonian electron-nucleus
    terms, not to independently-initialized trunk parameters.
    """

    config = OmegaConf.load(TRAIN_CONFIG)
    torch.manual_seed(0)
    embedding = instantiate(config.model.embedding)
    layers = [instantiate(layer_cfg) for layer_cfg in config.model.layers]
    readout = instantiate(config.model.readout)
    return embedding, layers, readout


def test_he_wavefunction_and_hamiltonian_migration_is_bit_identical() -> None:
    old_envelope, old_wavefunction_module = _load_pre_a5_modules()

    embedding, layers, readout = _shared_trunk()

    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )

    new_model = TPENWaveFunction(
        embedding=embedding,
        layers=list(layers),
        readout=readout,
        factors=[ElectronElectronCusp(), ElectronNucleusCusp(atoms=atoms)],
    ).to(dtype=torch.float64)

    OldTPENWaveFunction = old_wavefunction_module.TPENWaveFunction
    old_model = OldTPENWaveFunction(
        embedding=embedding,
        layers=list(layers),
        readout=readout,
        nuclear_envelope=old_envelope.NuclearFactorizedEnvelope(
            old_envelope.AdditiveEnvelope([old_envelope.ElectronElectronCusp()]),
            old_envelope.NuclearConfinement(),
        ),
    ).to(dtype=torch.float64)

    batch = _fixed_he_batch()

    old_output = old_model(batch)
    new_output = new_model(batch)

    assert torch.equal(old_output.logabs, new_output.logabs)
    assert torch.equal(old_output.sign, new_output.sign)
    assert (old_output.phase is None) == (new_output.phase is None)
    if old_output.phase is not None:
        assert torch.equal(old_output.phase, new_output.phase)

    old_terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusInteraction(eps=0.0),
        "electron_electron": ElectronElectronInteraction(),
    }
    new_terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms=atoms, eps=0.0),
        "electron_electron": ElectronElectronInteraction(),
        "nucleus_nucleus": NucleusNucleusPotential(atoms=atoms),
    }

    old_result = local_energy(old_terms, old_model, batch, return_terms=True)
    new_result = local_energy(new_terms, new_model, batch, return_terms=True)

    # He has one nucleus: the newly-wired generic nucleus_nucleus term must
    # contribute exactly zero, so its addition cannot change the total even
    # though the old Hamiltonian never carried that term.
    assert torch.equal(new_result.terms["nucleus_nucleus"], torch.zeros_like(new_result.total))
    assert torch.equal(old_result.total, new_result.total)
    for name in ("kinetic", "electron_nucleus", "electron_electron"):
        assert torch.equal(old_result.terms[name], new_result.terms[name]), (
            f"local-energy term {name!r} diverged under migration"
        )
