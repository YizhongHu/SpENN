"""Contracts for the trajectory-statistics consumer wiring.

Every test that claims an end-to-end result drives a *real* gradient-requiring
local energy: a real `KineticEnergy` term differentiating a real model's
``logabs`` twice with respect to positions, over a real serially-correlated
walker chain. Nothing here asserts against a hand-built receipt, because a
hand-built record would prove only that the assertion matches the fixture.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from tpen.checkpoint.hashing import file_sha256
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.geometry import electron_nuclear_displacements
from tpen.data.batch.walkers import Walkers
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.evaluation.generators import TRAJECTORY_METADATA_KEY, TrajectoryMCMCGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries import LocalEnergySummary, TrajectoryStatisticsSummary
from tpen.evaluation.summaries.trajectory_statistics import DEFAULT_SIDECAR_NAME
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, ElectronNucleusInteraction
from tpen.sampling.stats import SamplerStats
from tpen.statistics import TrajectoryStatisticsIdentity, TrajectoryStatisticsSidecar

STAGE = "eval"
RUN_ID = "he-v1-test"
ATTEMPT_ID = "attempt-1"
EVALUATOR_ID = "he_mcse/v1"
CONFIG = {"system": {"name": "helium"}, "evaluation": {"namespace": "eval"}}


class HeliumSlaterModel(torch.nn.Module):
    """``log|psi| = -alpha * sum_i r_i`` with a trainable ``alpha``.

    Real and twice-differentiable: the kinetic term takes two autograd
    derivatives of ``logabs`` with respect to positions, so this model exercises
    the gradient path rather than standing in for it. The parameter exists so
    the collector's freeze-and-fingerprint contract has something to act on.
    """

    def __init__(self, alpha: float = 1.8) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        distance = electron_nuclear_displacements(batch).norm(dim=-1)
        logabs = -self.alpha * distance.sum(dim=(-2, -1))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class CorrelatedHeliumSampler:
    """Persistent AR(1) walker chain around the helium nucleus.

    Positions carry over between calls and move by a small increment, so
    consecutive draws are genuinely serially correlated -- which is the whole
    property the autocorrelation estimate is supposed to detect. A sampler that
    redrew independently every call would make ``tau_int`` approximately one and
    the test would pass without exercising anything.
    """

    def __init__(self, *, n_walkers: int = 4, n_steps: int = 2, rho: float = 0.85, seed: int = 11) -> None:
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.rho = rho
        self.generator = torch.Generator().manual_seed(seed)
        self.nuclear_positions = torch.zeros(1, 3, dtype=torch.float64)
        self.nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
        # Start well away from the nucleus: the Coulomb terms diverge at r = 0
        # and a non-finite draw would be reported as `unresolved`, masking the
        # behaviour under test.
        self.positions = 0.9 + 0.2 * torch.rand(
            (n_walkers, 2, 3), generator=self.generator, dtype=torch.float64
        )
        self.call_count = 0

    def collect_samples(
        self,
        model: torch.nn.Module,
        *,
        reset: bool = False,
        device: torch.device | str | None = None,
    ) -> tuple[Walkers, SamplerStats]:
        self.call_count += 1
        noise = 0.05 * torch.randn(
            self.positions.shape, generator=self.generator, dtype=torch.float64
        )
        # Mean-reverting around unit radius so the chain neither collapses onto
        # the nucleus nor wanders off.
        self.positions = 1.0 + self.rho * (self.positions - 1.0) + noise
        walkers = Walkers(
            positions=self.positions.clone(),
            nuclear_positions=self.nuclear_positions,
            nuclear_charges=self.nuclear_charges,
        )
        stats = SamplerStats(0.6, self.n_walkers, 0, self.n_steps, 0.1, seed=self.call_count)
        return walkers, stats


def _terms() -> tuple[object, ...]:
    return (KineticEnergy(), ElectronElectronInteraction(), ElectronNucleusInteraction())


def _context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        namespace="eval/mcmc_energy",
        artifact_level="summaries",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=3,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _checkpoint(tmp_path: Path, *, payload: bytes = b"he-v1-checkpoint-bytes") -> Path:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(payload)
    return path


def _summary(tmp_path: Path, checkpoint: Path, **overrides) -> TrajectoryStatisticsSummary:
    kwargs = {
        "stage": STAGE,
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "checkpoint_path": checkpoint,
        "evaluator_id": EVALUATOR_ID,
        "config": CONFIG,
    }
    kwargs.update(overrides)
    return TrajectoryStatisticsSummary(**kwargs)


def _generate(tmp_path: Path, *, n_draws: int = 48, model: HeliumSlaterModel | None = None):
    generator = TrajectoryMCMCGenerator(
        sampler=CorrelatedHeliumSampler(),
        hamiltonian_terms=_terms(),
        n_draws=n_draws,
        discard_draws=2,
        chunk_size=2,
    )
    return generator.generate(model=model or HeliumSlaterModel(), context=_context(tmp_path))


def test_real_local_energy_trajectory_yields_available_mcse(tmp_path: Path) -> None:
    """Drive the whole path on a real autograd local energy, end to end."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)
    trajectory = generated.metadata[TRAJECTORY_METADATA_KEY]

    # The trajectory is a real time series of real local energies, not a stub.
    assert trajectory.observable == "local_energy"
    assert trajectory.values.shape == (48, 4)
    assert trajectory.draw_stride == 2
    assert trajectory.burn_in_draws == 2
    assert torch.isfinite(trajectory.values).all()
    assert trajectory.values.std() > 0.0

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    assert result.metrics["local_energy_trajectory_statistics_available"] is True
    assert result.metrics["local_energy_mcse"] > 0.0
    assert result.metrics["local_energy_ess"] > 0.0
    assert result.metrics["local_energy_tau_int"] > 0.0
    assert result.metrics["local_energy_trajectory_walkers"] == 4
    assert result.metrics["local_energy_trajectory_draws_per_walker"] == 48
    assert result.metrics["local_energy_trajectory_total_draws"] == 192

    (artifact,) = result.artifacts
    assert artifact.metadata["status"] == "available"
    assert artifact.metadata["reason"] is None
    assert artifact.metadata["estimator_id"] == "pooled_geyer_ips"


def test_kinetic_term_genuinely_requires_grad(tmp_path: Path) -> None:
    """Prove the observable needs autograd rather than merely tolerating it.

    Without this, a future `torch.no_grad()` wrapper around collection could be
    added and every other test here would still pass -- the collected values
    would just be wrong or the failure would surface somewhere unrelated. Under
    `no_grad` the second derivative cannot be taken at all, so the path raises.
    """

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            _generate(tmp_path, n_draws=8)


def test_mcse_and_iid_stderr_are_both_published_and_distinct(tmp_path: Path) -> None:
    """Both estimators survive, labelled, and the IID bar is never relabelled."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)
    context = _context(tmp_path)

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=context,
        namespace="eval/mcmc_energy",
    )

    mcse = result.metrics["local_energy_mcse"]
    stderr_iid = result.metrics["local_energy_stderr_iid"]
    assert stderr_iid > 0.0
    # Both are published under distinct names; neither is the other.
    assert "local_energy_mcse" in result.metrics
    assert "local_energy_stderr_iid" in result.metrics
    assert mcse != stderr_iid
    # A serially correlated chain must inflate the honest bar, never shrink it
    # below the naive one -- that direction would flatter the result.
    assert result.metrics["local_energy_mcse_inflation"] == pytest.approx(mcse / stderr_iid)
    assert result.metrics["local_energy_mcse_inflation"] > 1.0

    # The pre-existing snapshot IID stderr is a different sample and is left
    # untouched by this consumer: it must still be emitted under its own name.
    snapshot = LocalEnergySummary().summarize(
        bundle=EvaluationBundle(generated=generated),
        context=context,
        namespace="eval/mcmc_energy",
    )
    assert "local_energy_stderr" in snapshot.metrics
    assert "local_energy_stderr" not in result.metrics
    assert "local_energy_mcse" not in snapshot.metrics


def test_exactly_one_receipt_is_written_and_matches_the_identity(tmp_path: Path) -> None:
    """One sidecar line, keyed by the seven-part content-addressed identity."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)

    _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    sidecar_path = tmp_path / DEFAULT_SIDECAR_NAME
    lines = [line for line in sidecar_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["identity"]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert record["identity"]["stage"] == STAGE
    assert record["identity"]["run_id"] == RUN_ID
    assert record["identity"]["attempt_id"] == ATTEMPT_ID
    assert record["identity"]["observable"] == "local_energy"
    assert record["identity"]["evaluator_id"] == EVALUATOR_ID
    # Content-addressed, never path-derived: no part of the checkpoint location
    # may appear in the key.
    assert str(checkpoint) not in json.dumps(record["identity"])


def test_moved_run_tree_still_joins(tmp_path: Path) -> None:
    """A relocated run tree joins on the same key; the hash is over contents."""

    original = tmp_path / "original"
    original.mkdir()
    checkpoint = _checkpoint(original)
    generated = _generate(tmp_path)
    context = EvaluationContext(
        namespace="eval/mcmc_energy",
        artifact_level="summaries",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=3,
        run_dir=original,
        task_output_dir=original,
        metadata={},
    )
    _summary(original, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=context,
        namespace="eval/mcmc_energy",
    )

    moved = tmp_path / "moved"
    shutil.move(str(original), str(moved))

    # Rebuild the join key from the checkpoint at its NEW location.
    identity = TrajectoryStatisticsIdentity(
        stage=STAGE,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        checkpoint_sha256=file_sha256(moved / "checkpoint.pt"),
        config_sha256=json.loads((moved / DEFAULT_SIDECAR_NAME).read_text().splitlines()[0])[
            "identity"
        ]["config_sha256"],
        observable="local_energy",
        evaluator_id=EVALUATOR_ID,
    )
    receipt = TrajectoryStatisticsSidecar(moved / DEFAULT_SIDECAR_NAME).get(identity)
    assert receipt is not None
    assert receipt.status == "available"


def test_absent_trajectory_reports_reason_and_omits_statistics(tmp_path: Path) -> None:
    """A snapshot generator yields `absent` with a reason, never a zero."""

    checkpoint = _checkpoint(tmp_path)
    generated = GeneratedConfigurations(
        batch=_generate(tmp_path, n_draws=8).batch,
        metadata={},
    )

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    assert result.metrics["local_energy_trajectory_statistics_available"] is False
    # Omitted, not zero-filled: a zero MCSE would read as "no correlation".
    for key in ("local_energy_mcse", "local_energy_ess", "local_energy_tau_int", "local_energy_stderr_iid"):
        assert key not in result.metrics

    (artifact,) = result.artifacts
    assert artifact.metadata["status"] == "absent"
    assert artifact.metadata["reason"]
    assert TRAJECTORY_METADATA_KEY in str(artifact.metadata["reason"])

    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    assert receipt.status == "absent"
    assert receipt.payload is None


def test_too_few_draws_reports_unresolved_with_reason(tmp_path: Path) -> None:
    """An unresolvable real trajectory stays `unresolved`, never imputed."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path, n_draws=3)

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    assert result.metrics["local_energy_trajectory_statistics_available"] is False
    assert "local_energy_mcse" not in result.metrics
    # Shape is still reported: three draws is exactly what explains the outcome.
    assert result.metrics["local_energy_trajectory_draws_per_walker"] == 3

    (artifact,) = result.artifacts
    assert artifact.metadata["status"] == "unresolved"
    assert artifact.metadata["reason"]

    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    assert receipt.status == "unresolved"
    assert receipt.payload is None
    assert receipt.reason


def test_summary_requires_a_config_identity(tmp_path: Path) -> None:
    """The join key admits no blanks, so a missing config hash fails loudly."""

    with pytest.raises(ValueError, match="config_sha256 or config"):
        TrajectoryStatisticsSummary(
            stage=STAGE,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            checkpoint_path=_checkpoint(tmp_path),
            evaluator_id=EVALUATOR_ID,
        )


def test_trajectory_metadata_must_be_typed(tmp_path: Path) -> None:
    """A wrong-typed trajectory is a defect, not an `absent` receipt."""

    generated = GeneratedConfigurations(
        batch=_generate(tmp_path, n_draws=8).batch,
        metadata={TRAJECTORY_METADATA_KEY: torch.zeros(4, 2)},
    )
    with pytest.raises(TypeError, match="ObservableTrajectory"):
        _summary(tmp_path, _checkpoint(tmp_path)).summarize(
            bundle=EvaluationBundle(generated=generated),
            context=_context(tmp_path),
            namespace="eval/mcmc_energy",
        )
