"""Tests for the minimal Evaluate runner (sampled local-energy evaluation).

Evaluate is deliberately minimal before PR6 diagnostics: it samples, computes
intrinsic local-energy metrics, logs them through the RunContext, and emits
evaluation lifecycle events. It does not read reference energy and does not
accept diagnostics or callbacks/loggers (those are RunContext-owned).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import spenn.runner as runner_module
from spenn.artifacts import RunContext
from spenn.callback import Callback, Event
from spenn.physics.hamiltonian import summarize_local_energy
from spenn.physics.kinetic import KineticEnergy
from spenn.physics.potential import ElectronElectronInteraction, HarmonicTrap
from spenn.run import run_from_config
from spenn.runner import Evaluate
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hooke"


def test_evaluate_uses_summary_helper_from_physics_hamiltonian() -> None:
    assert runner_module.summarize_local_energy is summarize_local_energy
    assert "summarize_local_energy" not in runner_module.__all__


def test_evaluate_accepts_only_minimal_constructor_args() -> None:
    params = set(inspect.signature(Evaluate.__init__).parameters)
    assert params == {"self", "model", "sampler", "hamiltonian_terms", "return_terms"}


def test_evaluate_rejects_reference_energy_api() -> None:
    with pytest.raises(TypeError):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], evaluation={"reference_energy": 2.0})


def test_evaluate_rejects_diagnostics_before_pr6() -> None:
    with pytest.raises(TypeError):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], diagnostics=[])


def test_evaluate_rejects_callbacks_and_loggers() -> None:
    with pytest.raises(TypeError):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], callbacks=[])
    with pytest.raises(TypeError):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], loggers=[])


@pytest.mark.parametrize("fixture", ["exact_singlet.yaml", "exact_triplet.yaml"])
def test_evaluate_config_is_root_owned_and_has_no_reference_energy(fixture: str) -> None:
    cfg = OmegaConf.load(FIXTURES / fixture)
    # Callbacks and loggers are config-root / RunContext-owned, not on the runner.
    assert "callbacks" in cfg and "loggers" in cfg
    assert "callbacks" not in cfg.runner
    assert "loggers" not in cfg.runner
    # return_terms is passed directly; no evaluation/reference_energy block.
    assert cfg.runner.return_terms is True
    assert "evaluation" not in cfg
    assert "reference_energy" not in (FIXTURES / fixture).read_text()


class _EventRecorder(Callback):
    """Records every emitted lifecycle event name."""

    def __init__(self) -> None:
        super().__init__(
            triggers=("run_start", "evaluate_start", "samples_collected", "evaluate_end", "run_end")
        )
        self.events: list[str] = []

    def handle(self, event: Event) -> None:
        self.events.append(event.name)


class _RecordingContext(RunContext):
    """Minimal RunContext: holds root callbacks, swallows logs."""

    def __init__(self, callbacks) -> None:
        self.callbacks = list(callbacks)
        self.loggers = []
        self.records: list[tuple[str, dict]] = []

    def log(self, metrics, *, step=None, namespace="run", event=None) -> None:
        self.records.append((namespace, dict(metrics)))


def test_evaluate_emits_lifecycle_events_through_run_context() -> None:
    recorder = _EventRecorder()
    context = _RecordingContext([recorder])
    runner = Evaluate(
        model=build_tiny_spenn(),
        sampler=build_tiny_sampler(),
        hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()],
        return_terms=True,
    )

    result = runner.run(context)

    assert result.status == "completed"
    assert recorder.events == [
        "run_start",
        "evaluate_start",
        "samples_collected",
        "evaluate_end",
        "run_end",
    ]
    # Logged through the context under the eval namespace, intrinsic metrics only.
    eval_records = [m for ns, m in context.records if ns == "eval"]
    assert eval_records
    assert "energy_mean" in eval_records[-1]
    assert "reference_energy" not in eval_records[-1]


def _eval_metrics(run_root: Path) -> dict:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    eval_records = [record["metrics"] for record in records if record.get("namespace") == "eval"]
    assert eval_records, "no eval metric records were logged"
    return eval_records[-1]


@pytest.mark.parametrize(
    ("fixture", "exact_energy"),
    [("exact_singlet.yaml", 2.0), ("exact_triplet.yaml", 1.25)],
)
def test_hooke_eval_runner_matches_exact_energy(tmp_path, fixture: str, exact_energy: float) -> None:
    config_path = FIXTURES / fixture
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    exit_code = run_from_config(cfg, config_path=str(config_path), command="pytest")
    assert exit_code == 0

    metrics = _eval_metrics(tmp_path)
    energy_atol = float(cfg.validation.energy_atol)
    variance_max = float(cfg.validation.variance_max)

    # Evaluate logs intrinsic metrics only -- no reference-energy comparison.
    assert "reference_energy" not in metrics
    assert "abs_energy_error" not in metrics
    assert metrics["n_finite_samples"] == metrics["n_samples"] == 512
    assert metrics["nonfinite_energy_fraction"] == 0.0
    # The exact-energy comparison is done test-side against the intrinsic mean.
    assert abs(metrics["energy_mean"] - exact_energy) < energy_atol
    assert metrics["energy_variance"] < variance_max
    # return_terms: true -> per-term decomposition is logged as terms.<name>_mean
    # and terms.<name>_nonfinite_fraction.
    for term in ("kinetic", "harmonic_trap", "electron_electron"):
        assert f"terms.{term}_mean" in metrics
        assert metrics[f"terms.{term}_nonfinite_fraction"] == 0.0
    # sampler diagnostics are logged with sampler.* keys.
    assert metrics["sampler.n_walkers"] == 512
    assert "sampler.acceptance_rate" in metrics


@pytest.mark.parametrize("fixture", ["exact_singlet.yaml", "exact_triplet.yaml"])
def test_hooke_eval_runner_writes_standard_artifacts(tmp_path, fixture: str) -> None:
    config_path = FIXTURES / fixture
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    assert run_from_config(cfg, config_path=str(config_path), command="pytest") == 0

    run_dirs = list(tmp_path.glob("hooke_exact/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    run_dir = run_dirs[0]
    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "report.md",
        "metrics.jsonl",
        "metrics.csv",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
