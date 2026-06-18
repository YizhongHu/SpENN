"""Tests for the composable Evaluate runner."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import spenn.run as run_module
import spenn.runner as runner_module
import spenn.runner.evaluate as evaluate_runner_module
import spenn.runner.train as train_runner_module
from spenn.artifacts import RunContext
from spenn.callback import Callback, Event
from spenn.checkpoint import RestoreReport
from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.evaluation import (
    EvaluationTask,
    Evaluator,
    HamiltonianTermSummary,
    LocalEnergyCalculator,
    LocalEnergySummary,
    MCMCGenerator,
    ReferenceEnergySummary,
    SamplerStatsSummary,
    WavefunctionCalculator,
)
from spenn.physics.hamiltonian import LocalEnergyResult
from spenn.physics.kinetic import KineticEnergy
from spenn.physics.potential import ElectronElectronInteraction, HarmonicTrap
from spenn.run import run_from_config
from spenn.runner import Evaluate, Train
from spenn.training.state import TrainerState
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn

FIXTURES = Path(__file__).resolve().parents[1] / "artifacts" / "hooke"


def test_evaluate_constructor_uses_evaluator_boundary() -> None:
    params = set(inspect.signature(Evaluate.__init__).parameters)
    assert params == {"self", "model", "load", "evaluator", "construction_seed"}


def test_evaluate_requires_evaluator() -> None:
    with pytest.raises(ValueError, match="requires an evaluator"):
        Evaluate(model=None)


@pytest.mark.parametrize("fixture", ["exact_singlet_eval.yaml", "exact_triplet_eval.yaml"])
def test_evaluate_config_is_root_owned_and_uses_evaluator(fixture: str) -> None:
    cfg = OmegaConf.load(FIXTURES / fixture)
    assert "callbacks" in cfg and "loggers" in cfg
    assert "callbacks" not in cfg.runner
    assert "loggers" not in cfg.runner
    assert cfg.runner.evaluator == "${evaluator}"
    assert cfg.evaluator.namespace == "${evaluation.namespace}"
    assert "exact_energy" not in cfg.system
    assert "phase" not in cfg.evaluation

    raw = OmegaConf.to_container(cfg, resolve=False)
    assert raw["evaluator"]["tasks"] == ["${evaluation_tasks.energy}"]
    assert raw["evaluation_tasks"]["energy"]["generator"]["_target_"] == "spenn.evaluation.generators.MCMCGenerator"
    assert raw["evaluation_tasks"]["energy"]["summaries"][-1]["_target_"] == "spenn.evaluation.summaries.ReferenceEnergySummary"


def test_instantiate_runner_uses_normal_hydra_recursion_for_evaluator(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("_instantiate_runner must not special-case evaluation tasks")

    monkeypatch.setattr(run_module, "_instantiate_sequence", fail_if_called)
    cfg = OmegaConf.create(
        {
            "runner": {
                "_target_": "spenn.runner.Evaluate",
                "model": None,
                "evaluator": {
                    "_target_": "spenn.evaluation.Evaluator",
                    "namespace": "eval",
                    "tasks": [],
                },
            }
        }
    )

    runner = run_module._instantiate_runner(_runner_context(cfg))

    assert isinstance(runner, Evaluate)
    assert isinstance(runner.evaluator, Evaluator)


def test_train_config_with_evaluator_fails_as_normal_constructor_error() -> None:
    cfg = OmegaConf.create(
        {
            "runner": {
                "_target_": "spenn.runner.Train",
                "model": None,
                "sampler": None,
                "hamiltonian_terms": [],
                "optimizer": None,
                "trainer": None,
                "evaluator": {
                    "_target_": "spenn.evaluation.Evaluator",
                    "namespace": "eval",
                    "tasks": [],
                },
            }
        }
    )

    with pytest.raises(Exception, match="evaluator"):
        run_module._instantiate_runner(_runner_context(cfg))


def test_runtime_dtype_rejects_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match="floating point"):
        runner_module._runtime_dtype("int64")


def test_train_asserts_eager_initialization_before_optimizer_construction() -> None:
    class _OptimizerFactory:
        called = False

        def __call__(self, params):
            self.called = True
            raise AssertionError("optimizer should not be constructed")

    optimizer = _OptimizerFactory()
    recorder = _EventRecorder()
    context = _RecordingContext([recorder])
    runner = Train(
        model=nn.LazyLinear(1),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=optimizer,
        trainer=object(),
    )

    with pytest.raises(RuntimeError, match="uninitialized"):
        runner.run(context)

    assert optimizer.called is False
    assert recorder.events == ["run_start"]


def test_evaluate_start_is_emitted_after_model_ready() -> None:
    recorder = _EventRecorder()
    context = _RecordingContext([recorder])
    runner = Evaluate(
        model=nn.LazyLinear(1),
        evaluator=_energy_evaluator(_StaticSampler(torch.zeros(1, 2, 1, dtype=torch.float64)), []),
    )

    with pytest.raises(RuntimeError, match="uninitialized"):
        runner.run(context)

    assert recorder.events == ["run_start"]


def test_train_rejects_model_only_load_mode() -> None:
    runner = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"mode": "model_only", "path": "unused"},
    )

    with pytest.raises(ValueError, match="load.mode.*model_only"):
        runner.run(_RecordingContext([]))


def test_evaluate_rejects_train_resume_load_mode() -> None:
    runner = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(_StaticSampler(torch.zeros(1, 2, 1, dtype=torch.float64)), []),
        load={"mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(ValueError, match="load.mode.*train_resume"):
        runner.run(_RecordingContext([]))


def test_train_train_resume_calls_runner_owned_restore(monkeypatch) -> None:
    calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        calls.append(kwargs)
        return RestoreReport(mode="train_resume", checkpoint_dir="ckpt", step=4)

    monkeypatch.setattr(train_runner_module, "restore_checkpoint_with_events", fake_restore_checkpoint_with_events)
    runner = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"mode": "train_resume", "path": "ckpt"},
    )

    result = runner.run(_RecordingContext([]))

    assert result.status == "completed"
    assert calls and calls[0]["model"] is runner.model
    assert calls[0]["trainer"] is runner.trainer
    assert calls[0]["sampler"] is runner.sampler
    assert calls[0]["emit"] == runner.emit


def test_evaluate_model_only_calls_runner_owned_restore(monkeypatch) -> None:
    calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        calls.append(kwargs)
        return RestoreReport(mode="model_only", checkpoint_dir="ckpt", step=4)

    monkeypatch.setattr(evaluate_runner_module, "restore_checkpoint_with_events", fake_restore_checkpoint_with_events)
    runner = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(
            _StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64)),
            {"constant": _ConstantEnergyTerm([1.0, 1.0])},
        ),
        load={"mode": "model_only", "path": "ckpt"},
    )

    result = runner.run(_RecordingContext([]))

    assert result.status == "completed"
    assert calls and calls[0]["model"] is runner.model
    assert "sampler" not in calls[0]
    assert calls[0]["emit"] == runner.emit


def test_checkpoint_load_mode_none_does_not_call_restore(monkeypatch) -> None:
    def fail_restore(**kwargs):
        raise AssertionError("restore_checkpoint should not be called")

    monkeypatch.setattr(train_runner_module, "restore_checkpoint_with_events", fail_restore)
    monkeypatch.setattr(evaluate_runner_module, "restore_checkpoint_with_events", fail_restore)

    train = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"mode": "none"},
    )
    assert train.run(_RecordingContext([])).status == "completed"

    evaluate = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(
            _StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64)),
            {"constant": _ConstantEnergyTerm([1.0, 1.0])},
        ),
        load={"mode": "none"},
    )
    assert evaluate.run(_RecordingContext([])).status == "completed"


def test_evaluate_emits_lifecycle_events_through_run_context() -> None:
    recorder = _EventRecorder()
    context = _RecordingContext([recorder])
    runner = Evaluate(
        model=build_tiny_spenn(),
        evaluator=_energy_evaluator(
            build_tiny_sampler(),
            [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()],
            return_terms=True,
        ),
    )

    result = runner.run(context)

    assert result.status == "completed"
    assert recorder.events == ["run_start", "evaluate_start", "task_start", "task_end", "evaluate_end", "run_end"]
    energy_records = [m for ns, m in context.records if ns == "eval/energy"]
    assert energy_records
    assert "local_energy_mean" in energy_records[-1]
    assert "reference_energy" not in energy_records[-1]


def test_evaluate_logs_reference_and_term_metrics_from_task() -> None:
    context = _RecordingContext([])
    sampler = _StaticSampler(
        torch.tensor(
            [
                [[0.0], [1.0]],
                [[2.0], [3.0]],
                [[4.0], [5.0]],
            ],
            dtype=torch.float64,
        )
    )
    runner = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(
            sampler,
            {
                "kinetic": _ConstantEnergyTerm([1.0, 2.0, 3.0]),
                "harmonic_trap": _ConstantEnergyTerm([4.0, 5.0, 6.0]),
            },
            return_terms=True,
            reference_energy=7.0,
        ),
    )

    result = runner.run(context)

    assert result.status == "completed"
    assert sampler.calls == 1
    metrics = [m for ns, m in context.records if ns == "eval/energy"][0]
    assert metrics["local_energy_mean"] == pytest.approx(7.0)
    assert metrics["energy_error"] == pytest.approx(0.0)
    assert metrics["energy_abs_error"] == pytest.approx(0.0)
    assert metrics["term/kinetic_mean"] == pytest.approx(2.0)
    assert metrics["term/harmonic_trap_mean"] == pytest.approx(5.0)
    assert metrics["sampler_n_walkers"] == 3


def _runner_context(cfg) -> RunContext:
    context = object.__new__(RunContext)
    context.cfg = cfg
    return context


class _EventRecorder(Callback):
    def __init__(self) -> None:
        super().__init__(
            triggers=("run_start", "evaluate_start", "task_start", "task_end", "task_failed", "evaluate_end", "run_end")
        )
        self.events: list[str] = []

    def handle(self, event: Event) -> None:
        self.events.append(event.name)


class _RecordingContext(RunContext):
    def __init__(self, callbacks) -> None:
        self.callbacks = list(callbacks)
        self.loggers = []
        self.metadata = SimpleNamespace(device="cpu", dtype="float64")
        self.records: list[tuple[str, dict]] = []

    def log(self, metrics, *, step=None, namespace="run", event=None) -> None:
        self.records.append((namespace, dict(metrics)))


class _QuadraticModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        logabs = -flat.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _StaticSampler:
    def __init__(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.calls = 0

    def collect_samples(self, model, *, device: str | torch.device | None = None):
        self.calls += 1
        positions = self.positions.to(device=device)
        return Walkers(positions=positions), {"n_walkers": positions.shape[0], "acceptance_rate": 1.0}


class _NoopTrainer:
    def fit(self, *, model, sampler, hamiltonian_terms, optimizer, context, emit):
        return TrainerState(step=0, model=model, optimizer=optimizer, trainer=self, sampler=sampler)


class _ConstantEnergyTerm:
    def __init__(self, values) -> None:
        self.values = torch.as_tensor(values, dtype=torch.float64)

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        values = self.values.to(device=batch.device, dtype=batch.dtype)
        return LocalEnergyResult(total=values, terms={"internal": values})


def _energy_evaluator(
    sampler,
    terms,
    *,
    return_terms: bool = False,
    reference_energy: float | None = None,
) -> Evaluator:
    summaries = [
        LocalEnergySummary(),
        HamiltonianTermSummary(),
        SamplerStatsSummary(),
    ]
    if reference_energy is not None:
        summaries.append(ReferenceEnergySummary(reference_energy=reference_energy))
    return Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                generator=MCMCGenerator(sampler=sampler),
                calculators=[
                    WavefunctionCalculator(),
                    LocalEnergyCalculator(hamiltonian_terms=terms, return_terms=return_terms),
                ],
                summaries=summaries,
            )
        ],
    )


def _metrics(run_root: Path, namespace: str) -> dict:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    matches = [record["metrics"] for record in records if record.get("namespace") == namespace]
    assert matches, f"no records for namespace {namespace}"
    return matches[-1]


@pytest.mark.parametrize(
    ("fixture", "exact_energy"),
    [("exact_singlet_eval.yaml", 2.0), ("exact_triplet_eval.yaml", 1.25)],
)
def test_hooke_eval_runner_matches_exact_energy(tmp_path, fixture: str, exact_energy: float) -> None:
    config_path = FIXTURES / fixture
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    exit_code = run_from_config(cfg, config_path=str(config_path), command="pytest")
    assert exit_code == 0

    metrics = _metrics(tmp_path, "eval/energy")
    energy_atol = float(cfg.validation.energy_atol)
    variance_max = float(cfg.validation.variance_max)

    assert metrics["reference_energy"] == pytest.approx(exact_energy)
    assert abs(metrics["energy_error"]) < energy_atol
    assert metrics["energy_abs_error"] < energy_atol
    assert metrics["local_energy_n_finite"] == metrics["local_energy_n_total"] == 512
    assert metrics["local_energy_finite_fraction"] == 1.0
    assert metrics["local_energy_nonfinite_count"] == 0
    assert abs(metrics["local_energy_mean"] - exact_energy) < energy_atol
    assert metrics["local_energy_variance"] < variance_max
    for term in ("kinetic", "harmonic_trap", "electron_electron"):
        assert f"term/{term}_mean" in metrics
    assert metrics["sampler_n_walkers"] == 512
    assert "acceptance_rate" in {key.removeprefix("sampler_") for key in metrics}
    assert "wall_time_sec" in _metrics(tmp_path, "eval/perf")
    assert "time_sec" in _metrics(tmp_path, "diagnostics/energy")
    assert any("wall_time_sec" in record for record in _namespace_records(tmp_path, "runtime"))


def _namespace_records(run_root: Path, namespace: str) -> list[dict]:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    return [record["metrics"] for record in records if record.get("namespace") == namespace]


@pytest.mark.parametrize("fixture", ["exact_singlet_eval.yaml", "exact_triplet_eval.yaml"])
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
        "metrics.jsonl",
        "metrics.csv",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"


def test_hooke_exact_evaluation_stack_runs_from_yaml_fixture(tmp_path) -> None:
    """The full Generator->Calculator->Summary stack runs on exact Hooke from YAML.

    This is a correctness test for the evaluation implementation: the same
    deterministic task stack used for learned-model validation is exercised on
    the analytic Hooke singlet, where every diagnostic has a known answer.
    """

    config_path = FIXTURES / "hooke_exact_evaluation.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    exit_code = run_from_config(cfg, config_path=str(config_path), command="pytest")
    assert exit_code == 0

    # Exact local energy E_L = 2.0 everywhere finite, for every geometry task.
    # The exact singlet is nodeless (sign=+1) so every finite configuration has
    # a finite local energy. The variance tolerance is looser for the cusp task:
    # at r12=1e-5 the autograd Laplacian sums 1/r12 terms that cancel to give
    # E_L=2.0, and float64 catastrophic cancellation leaves a small residual.
    variance_tol = {"cusp": 1.0e-6, "tail": 1.0e-8, "stratified_geometry": 1.0e-8, "energy": 1.0e-8}
    for task in ("cusp", "tail", "stratified_geometry", "energy"):
        metrics = _metrics(tmp_path, f"hooke_exact/{task}")
        assert metrics["local_energy_finite_fraction"] == 1.0, task
        assert metrics["local_energy_nonfinite_count"] == 0, task
        assert metrics["local_energy_mean"] == pytest.approx(2.0, abs=1.0e-3), task
        assert metrics["local_energy_variance"] < variance_tol[task], task

    # Opposite-spin cusp: even slope -> 1/2; near-coalescence C_{-1} -> 0.
    cusp = _metrics(tmp_path, "hooke_exact/cusp")
    assert cusp["cusp_even_slope_mean"] == pytest.approx(0.5, abs=1.0e-3)
    assert cusp["cusp_even_slope_abs_error"] < 1.0e-3
    assert cusp["c_minus_1_abs_max"] < 1.0e-3

    # Reference energy comparison (eval-only summary, no phase gate).
    energy = _metrics(tmp_path, "hooke_exact/energy")
    assert energy["reference_energy"] == pytest.approx(2.0)
    assert energy["energy_abs_error"] < 1.0e-4

    # Each task writes under its resolved task output directory.
    run_dirs = list(tmp_path.glob("hooke_exact_stack/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    run_dir = run_dirs[0]
    for task in ("cusp", "tail", "stratified_geometry", "energy"):
        assert (run_dir / task).is_dir(), f"missing task output dir: {task}"
