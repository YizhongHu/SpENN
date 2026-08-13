"""The four-level basis axis is constructible, trainable, and restorable.

Drives every level of ``experiments/hooke/choices/basis_levels.yaml`` through the
real stack -- ``run_from_config`` -> ``Train`` -> ``TPENWaveFunction`` ->
``MetropolisSampler`` -> Hooke Hamiltonian -> ``VMCTrainer`` -- and then restores
each level's checkpoint under ``mode: model_only, strict: true``.

Two properties here are corrections to an earlier level table that did not
construct at all, and both are the kind of regression a green suite would
otherwise ratify:

``box_size: 2``
    ``_cartesian_box_multi_indices`` admits ``order < box_size``
    (``tpen/nn/basis.py:604-608``), so ``box_size`` counts admitted orders while
    ``max_shell``/``max_total_shell`` are maximum orders. At ``box_size: 1`` the
    basis is a single spherically symmetric channel, and because the basis is the
    only path by which coordinates reach the trainable body, that arm cannot
    represent r12 dependence at all. ``test_cartesian_box_*`` pins both the index
    set and the behavioural consequence.

``include_gaussian_factor``
    ``product_v2`` raises ``ValueError`` when it is unset
    (``tpen/nn/basis.py:443-444``); ``axiswise_v1`` silently defaults it to
    ``True``. ``test_every_hooke_level_sets_include_gaussian_factor_explicitly``
    reads the *unresolved* config so a level that merely inherits the default
    fails, not just one that throws.

The ``axiswise_v1`` ``FutureWarning`` is a retained legacy mode's contract, not a
defect: it must fire and must not be suppressed or escalated. It fires more than
once per run because the ``basis_feature_dim`` resolver instantiates the basis
again to size the embedding, so presence is asserted and multiplicity never is.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.run import run_from_config

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY = REPO_ROOT / "experiments" / "hooke" / "choices" / "basis_levels.yaml"
HOST = Path(__file__).resolve().parents[1] / "artifacts" / "hooke" / "basis_levels_train.yaml"

# Per-particle one-body width each level presents to the embedding, at the
# fixture's ``spatial_dim: 3`` with spin included. Hard-coded rather than
# recomputed from the enumerators, because a test that re-derives the width from
# the code under test cannot detect a change in that code.
#
#   no-basis             3 coordinates + 1 spin, derived by the embedding
#   hooke-axiswise-v1    3 axes * (max_shell 1 + 1) orders + 1 spin
#   hooke-total-shell    |n| <= 1 in 3D -> (0,0,0) + 3 unit indices, + 1 spin
#   hooke-cartesian-box  n_k < 2 in 3D -> 2**3 = 8 indices, + 1 spin
#
# The cartesian-box width is 9, NOT the 8 multi-indices the box admits:
# ``ElectronBasis.out_features`` adds the spin channel (``tpen/nn/basis.py:143-147``).
EXPECTED_FEATURE_DIM = {
    "no-basis": 4,
    "hooke-axiswise-v1": 7,
    "hooke-total-shell": 5,
    "hooke-cartesian-box": 9,
}
LEVELS = tuple(EXPECTED_FEATURE_DIM)
HOOKE_LEVELS = tuple(level for level in LEVELS if level != "no-basis")

# Every multi-index a 3D ``box_size: 2`` basis admits, in the canonical
# shell-major order. Written out rather than generated: this literal is what
# distinguishes the order-matched box from the single-channel ``box_size: 1``.
CARTESIAN_BOX_MULTI_INDICES = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)

# The fixture logs every step and runs exactly this many.
EXPECTED_TRAIN_STEPS = int(OmegaConf.load(HOST).trainer.max_steps)


def _config(slot: str, root: Path) -> DictConfig:
    """Return the host config merged with the library and pinned to one level."""

    cfg = OmegaConf.merge(OmegaConf.load(HOST), OmegaConf.load(LIBRARY))
    cfg.run_parameters.basis_slot = slot
    cfg.run.root = str(root)
    return cfg


def _basis(slot: str, tmp_path: Path):
    """Instantiate one level's basis straight from the choice library."""

    return instantiate(_config(slot, tmp_path).choices.basis[slot].basis)


def _records(run_dir: Path) -> list[dict]:
    """Return the run's logged metric records."""

    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _metrics(run_dir: Path, namespace: str) -> list[dict]:
    """Return the metric payloads logged under one namespace."""

    return [record["metrics"] for record in _records(run_dir) if record.get("namespace") == namespace]


def _events(run_dir: Path) -> list[str]:
    """Return the durable event names a run recorded, in written order."""

    return [
        json.loads(line)["event"]
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _run(cfg: DictConfig, root: Path, config_path: Path) -> Path:
    """Execute one configured run under `root` and return its run directory."""

    exit_code = run_from_config(
        cfg, config_path=str(config_path), command="pytest", raise_exceptions=True
    )
    assert exit_code == 0
    run_dirs = list(root.glob("hooke_basis_levels/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir under {root}, found {run_dirs}"
    return run_dirs[0]


@dataclass(frozen=True)
class _Trained:
    """One level's completed training run."""

    slot: str
    run_dir: Path
    config_path: Path


@pytest.fixture(scope="module", params=LEVELS)
def trained(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> _Trained:
    """Train one level once per module and share the run across assertions.

    The merged config is written out and reloaded before running. ``OmegaConf.save``
    preserves interpolations, so the saved file is a genuine single-file config of
    the shape ``run.py --config`` consumes -- which is how a study will consume the
    library, since ``tpen.run.load_config`` reads exactly one YAML file with no
    include mechanism.
    """

    slot = request.param
    root = tmp_path_factory.mktemp("basis_level_train")
    config_path = root / "merged_config.yaml"
    OmegaConf.save(_config(slot, root), config_path)
    return _Trained(slot=slot, run_dir=_run(OmegaConf.load(config_path), root, config_path), config_path=config_path)


# ---------------------------------------------------------------------------
# The library itself
# ---------------------------------------------------------------------------


def test_library_defines_exactly_the_four_levels(tmp_path: Path) -> None:
    """The axis has four levels; adding, dropping, or renaming one is a change."""

    assert tuple(_config("no-basis", tmp_path).choices.basis) == LEVELS


@pytest.mark.parametrize("slot", LEVELS)
def test_in_features_is_a_slot_key_that_matches_that_slots_own_basis(slot: str, tmp_path: Path) -> None:
    """Each slot carries its own embedding width, and it is the right one.

    ``in_features`` cannot be a bare ``${tpen.basis_feature_dim:${model.basis}}``
    at the model level: the resolver instantiates its argument and reads
    ``out_features``, and ``instantiate(None)`` has neither, so the ``no-basis``
    level raises ``TypeError`` (``tpen/config.py:66-70``). Resolving to ``null``
    inside the slot is what lets the embedding fall back to its derived width.

    The equality against the slot's *own* basis is what catches an ``in_features``
    interpolation left pointing at a neighbouring level after a copy-paste.
    """

    slot_config = _config(slot, tmp_path).choices.basis[slot]
    if slot == "no-basis":
        assert slot_config.basis is None
        assert slot_config.in_features is None
        return
    assert int(slot_config.in_features) == int(instantiate(slot_config.basis).out_features)


@pytest.mark.parametrize("slot", HOOKE_LEVELS)
def test_every_hooke_level_sets_include_gaussian_factor_explicitly(slot: str, tmp_path: Path) -> None:
    """The Gaussian factor is written down on every level, never inherited.

    ``product_v2`` raises ``ValueError`` without it (``tpen/nn/basis.py:443-444``)
    and ``axiswise_v1`` defaults it to ``True`` (``:411-413``). Reading the
    unresolved config fails a level that relies on either behaviour, so the three
    arms stay comparable by inspection rather than by remembering a default. It is
    outcome-determining: with the factor on, features decay as
    ``exp(-omega r^2 / 2)`` and are ~1e-7 at the evaluation tail stratum.
    """

    raw = OmegaConf.to_container(_config(slot, tmp_path).choices.basis[slot].basis, resolve=False)
    assert raw["include_gaussian_factor"] is True


@pytest.mark.parametrize("slot", HOOKE_LEVELS)
def test_basis_omega_is_threaded_from_system_omega(slot: str, tmp_path: Path) -> None:
    """Every level reads the system frequency instead of hard-coding one."""

    cfg = _config(slot, tmp_path)
    cfg.system.omega = 1.25
    assert instantiate(cfg.model).basis.omega == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Construction and feature widths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot", LEVELS)
def test_every_level_constructs_with_its_pinned_feature_width(slot: str, tmp_path: Path) -> None:
    """All four levels build a model whose embedding is sized for its basis."""

    model = instantiate(_config(slot, tmp_path).model)
    expected = EXPECTED_FEATURE_DIM[slot]

    assert model.embedding.particle_input_channels == expected
    if slot == "no-basis":
        # The embedding builds (coordinates, spin) itself; no override is applied.
        assert model.basis is None
        assert model.embedding.in_features is None
    else:
        assert model.basis is not None
        assert model.basis.out_features == expected


def test_cartesian_box_admits_the_full_order_one_box(tmp_path: Path) -> None:
    """``box_size: 2`` is the order-matched value; ``1`` admits one index.

    ``box_size`` counts admitted orders, so ``box_size: 1`` yields exactly
    ``((0, 0, 0),)`` -- one spherically symmetric channel. This pins the eight
    indices of the order-one box explicitly so reverting the value fails here
    rather than only downstream.
    """

    basis = _basis("hooke-cartesian-box", tmp_path)

    assert basis.box_size == 2
    assert basis.multi_indices == CARTESIAN_BOX_MULTI_INDICES


def test_cartesian_box_features_separate_geometries_that_share_every_radius(tmp_path: Path) -> None:
    """The box arm can see relative geometry, not only per-electron radius.

    Both configurations place both electrons at unit radius, so every
    radius-only feature map -- including the single ``(0, 0, 0)`` channel a
    ``box_size: 1`` basis would produce -- returns identical features for them.
    They differ only in separation (2 versus sqrt(2)). Because the basis is the
    sole route from coordinates to the trainable body, a basis that cannot
    separate these cannot represent the r12 factor the exact Hooke solution needs.
    """

    basis = _basis("hooke-cartesian-box", tmp_path)
    spins = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    opposed = ElectronBatch(
        positions=torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float64),
        spins=spins,
    )
    orthogonal = ElectronBatch(
        positions=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64),
        spins=spins,
    )

    # Guard the premise: the two configurations really are radius-degenerate.
    assert torch.allclose(
        opposed.positions.norm(dim=-1), orthogonal.positions.norm(dim=-1)
    )

    assert not torch.allclose(basis(opposed).one_body, basis(orthogonal).one_body)


# ---------------------------------------------------------------------------
# The retained legacy warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot", LEVELS)
def test_only_the_axiswise_level_warns(slot: str, tmp_path: Path) -> None:
    """``axiswise_v1`` warns; the other three levels are silent.

    Presence, never multiplicity: the warning fires at least twice per run
    because the ``basis_feature_dim`` resolver instantiates the basis a second
    time to size the embedding. Constructing the model inside a recorder also
    shows the warning is not escalated -- an escalated ``FutureWarning`` would
    raise here instead of being recorded.
    """

    cfg = _config(slot, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = instantiate(cfg.model)

    assert model is not None
    legacy = [
        entry
        for entry in caught
        if issubclass(entry.category, FutureWarning) and "axiswise_v1" in str(entry.message)
    ]
    if slot == "hooke-axiswise-v1":
        assert legacy, "the retained legacy level must announce itself"
    else:
        assert not legacy, f"{slot} emitted the axiswise_v1 deprecation warning"


# ---------------------------------------------------------------------------
# Training, runtime equivariance, and checkpoint restore
# ---------------------------------------------------------------------------


def test_every_level_trains_with_finite_energy(trained: _Trained) -> None:
    """Each level completes its configured steps with finite local energies."""

    train = _metrics(trained.run_dir, "train")
    assert len(train) == EXPECTED_TRAIN_STEPS >= 2

    for metrics in train:
        assert math.isfinite(metrics["energy"])
        assert math.isfinite(metrics["energy_variance"])
        # DataIntegrity is configured to fail on any non-finite sample; assert
        # the count directly so a silently relaxed threshold is still caught.
        assert metrics["local_energy_nonfinite_count"] == 0
        assert metrics["local_energy_n_finite"] > 0

    assert json.loads((trained.run_dir / "status.json").read_text())["status"] == "completed"


def test_every_level_trains_with_finite_gradients(trained: _Trained) -> None:
    """Each level produces gradients, and every one of them is finite."""

    gradients = _metrics(trained.run_dir, "checks/gradient")
    assert len(gradients) == EXPECTED_TRAIN_STEPS

    for metrics in gradients:
        # Assert something was measured before asserting it was healthy: every
        # bound below holds vacuously over an empty gradient set.
        assert metrics["n_grad_elements"] > 0
        assert metrics["nonfinite_grad_fraction"] == 0.0
        assert math.isfinite(metrics["global_grad_norm"])
        assert metrics["passed"] is True


def test_runtime_equivariance_passes_for_every_level(trained: _Trained) -> None:
    """Both runtime checkers compare something, and pass, at every level."""

    full_model = _metrics(trained.run_dir, "checks/equivariance/full_model")
    assert len(full_model) == EXPECTED_TRAIN_STEPS
    for metrics in full_model:
        # n_particles = 2 admits exactly one non-identity permutation.
        assert metrics["n_comparisons"] == 1
        assert metrics["passed"] is True

    trace = _metrics(trained.run_dir, "checks/equivariance/trace")
    assert len(trace) == EXPECTED_TRAIN_STEPS
    for metrics in trace:
        # A trace checker that recorded nothing satisfies every ratio below at
        # zero, so the counts are asserted before the verdict.
        assert metrics["n_trace_entries"] > 0
        assert metrics["n_comparisons"] > 0
        assert metrics["passed"] is True


def test_checkpoint_from_every_level_restores_model_only_strict(
    trained: _Trained, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Each level's checkpoint restores into the same model wiring.

    The evaluation config is derived from the config that produced the
    checkpoint rather than written out separately: a ``model_only`` restore
    verifies the ``model`` and ``hamiltonian_terms`` config hashes, so deriving
    it makes those sections identical by construction instead of by two files
    being kept in sync. A level whose model block did not survive the round trip
    fails the hash gate here.

    ``checkpoint_restored`` is asserted because a completed evaluation run is
    not evidence of a restore -- a run that silently skipped the load would also
    complete.
    """

    checkpoints = trained.run_dir / "checkpoints"
    assert (checkpoints / "latest.json").is_file(), "training wrote no checkpoint to restore"

    root = tmp_path_factory.mktemp("basis_level_eval")
    cfg = OmegaConf.load(trained.config_path)
    cfg.run.root = str(root)
    cfg.experiment.run_name = "hooke_basis_levels_eval"
    cfg.load = {
        "path": str(checkpoints),
        "mode": "model_only",
        "strict": True,
        "allow_protocol_mismatch": False,
    }
    cfg.runner = {
        "_target_": "tpen.runner.Evaluate",
        "model": "${model}",
        "load": "${load}",
        "evaluator": "${evaluator}",
    }
    cfg.evaluator = {
        "_target_": "tpen.evaluation.Evaluator",
        "namespace": "eval",
        "tasks": [
            {
                "name": "null_task",
                "namespace": "eval/null_task",
                "output_dir": "${run.dir}/null_task",
                "generator": {"_target_": "tests.helpers.evaluation_components.NullGenerator"},
                "calculators": [
                    {"_target_": "tests.helpers.evaluation_components.IdentityCalculator"}
                ],
                "summaries": [{"_target_": "tests.helpers.evaluation_components.MetricSummary"}],
            }
        ],
    }
    # The training callback battery has nothing to observe in an evaluation run,
    # and `events.jsonl` is written by the run context rather than a callback.
    cfg.callbacks = []

    run_dir = _run(cfg, root, trained.config_path)
    assert "checkpoint_restored" in _events(run_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
@pytest.mark.parametrize("slot", LEVELS)
def test_every_level_trains_on_the_accelerator(slot: str, tmp_path: Path) -> None:
    """Each level completes its steps on GPU, where the study will run them.

    The CPU runs above prove the wiring; this proves the same four levels reach
    finite energies and gradients on the device the scan actually uses. It skips
    where there is no accelerator, so it is a no-op off the cluster.
    """

    cfg = _config(slot, tmp_path)
    cfg.runtime.device = "cuda"
    run_dir = _run(cfg, tmp_path, HOST)

    train = _metrics(run_dir, "train")
    assert len(train) == EXPECTED_TRAIN_STEPS >= 2
    for metrics in train:
        assert math.isfinite(metrics["energy"])
        assert metrics["local_energy_nonfinite_count"] == 0

    gradients = _metrics(run_dir, "checks/gradient")
    assert len(gradients) == EXPECTED_TRAIN_STEPS
    for metrics in gradients:
        assert metrics["n_grad_elements"] > 0
        assert metrics["nonfinite_grad_fraction"] == 0.0
        assert metrics["passed"] is True
