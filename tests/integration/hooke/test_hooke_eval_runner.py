"""Tests for the Evaluate diagnostic runner."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

import spenn.runner as runner_module
import spenn.runner.evaluate as evaluate_runner_module
import spenn.runner.train as train_runner_module
import spenn.run as run_module
from spenn.checkpoint import RestoreReport
from spenn.artifacts import RunContext
from spenn.callback import Callback, Event
from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.diagnostics import EnergyEvaluation, EvaluationContext
from spenn.physics.hamiltonian import LocalEnergyResult
from spenn.physics.kinetic import KineticEnergy
from spenn.physics.potential import ElectronElectronInteraction, HarmonicTrap
from spenn.run import run_from_config
from spenn.runner import Evaluate, Train
from spenn.training.state import TrainerState
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn

FIXTURES = Path(__file__).resolve().parents[1] / "artifacts" / "hooke"


def test_evaluate_accepts_only_minimal_constructor_args() -> None:
    params = set(inspect.signature(Evaluate.__init__).parameters)
    assert params == {
        "self",
        "model",
        "sampler",
        "hamiltonian_terms",
        "diagnostics",
        "return_terms",
        "load",
    }


def test_evaluate_rejects_reference_energy_api() -> None:
    with pytest.raises(TypeError):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], evaluation={"reference_energy": 2.0})


@pytest.mark.parametrize("diagnostics", [None, []])
def test_evaluate_requires_at_least_one_diagnostic(diagnostics) -> None:
    with pytest.raises(ValueError, match="requires at least one diagnostic"):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], diagnostics=diagnostics)


@pytest.mark.parametrize("raw", [{"name": "energy"}, OmegaConf.create({"name": "energy"})])
def test_evaluate_rejects_raw_diagnostic_configs(raw) -> None:
    with pytest.raises(TypeError, match="instantiated diagnostic object.*Hydra"):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], diagnostics=[raw])


def test_evaluate_rejects_callbacks_and_loggers() -> None:
    with pytest.raises(TypeError):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], callbacks=[])
    with pytest.raises(TypeError):
        Evaluate(model=None, sampler=None, hamiltonian_terms=[], loggers=[])


@pytest.mark.parametrize("fixture", ["exact_singlet.yaml", "exact_triplet.yaml"])
def test_evaluate_config_is_root_owned_and_uses_diagnostics(fixture: str) -> None:
    cfg = OmegaConf.load(FIXTURES / fixture)
    # Callbacks and loggers are config-root / RunContext-owned, not on the runner.
    assert "callbacks" in cfg and "loggers" in cfg
    assert "callbacks" not in cfg.runner
    assert "loggers" not in cfg.runner
    # Diagnostics are runner-owned; system metadata stays reference-blind.
    assert "diagnostics" not in cfg
    assert "references" in cfg
    assert "exact_energy" not in cfg.system
    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=False)
    diagnostics_cfg = runner_cfg["diagnostics"]
    assert cfg.runner.return_terms is True
    assert "evaluation" not in cfg
    assert diagnostics_cfg[0]["_target_"] == "spenn.diagnostics.EnergyEvaluation"
    assert diagnostics_cfg[0]["reference_energy"] == "${references.exact_energy}"


def test_instantiate_runner_uses_normal_hydra_recursion_for_diagnostics(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("_instantiate_runner must not special-case diagnostics")

    monkeypatch.setattr(run_module, "_instantiate_sequence", fail_if_called)
    cfg = OmegaConf.create(
        {
            "runner": {
                "_target_": "spenn.runner.Evaluate",
                "model": None,
                "sampler": None,
                "hamiltonian_terms": [],
                "diagnostics": [
                    {
                        "_target_": "spenn.diagnostics.EnergyEvaluation",
                        "name": "energy",
                    }
                ],
            }
        }
    )

    runner = run_module._instantiate_runner(_runner_context(cfg))

    assert isinstance(runner, Evaluate)
    assert len(runner.diagnostics) == 1
    assert isinstance(runner.diagnostics[0], EnergyEvaluation)


def test_train_config_with_diagnostics_fails_as_normal_constructor_error() -> None:
    cfg = OmegaConf.create(
        {
            "runner": {
                "_target_": "spenn.runner.Train",
                "model": None,
                "sampler": None,
                "hamiltonian_terms": [],
                "optimizer": None,
                "trainer": None,
                "diagnostics": [
                    {
                        "_target_": "spenn.diagnostics.EnergyEvaluation",
                        "name": "energy",
                    }
                ],
            }
        }
    )

    with pytest.raises(Exception, match="diagnostics"):
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
        sampler=_StaticSampler(torch.zeros(1, 2, 1, dtype=torch.float64)),
        hamiltonian_terms=[],
        diagnostics=[EnergyEvaluation()],
    )

    with pytest.raises(RuntimeError, match="uninitialized"):
        runner.run(context)

    assert recorder.events == ["run_start"]


def test_train_rejects_model_only_restore_mode() -> None:
    runner = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"restore_mode": "model_only", "path": "unused"},
    )

    with pytest.raises(ValueError, match="load.restore_mode.*model_only"):
        runner.run(_RecordingContext([]))


def test_evaluate_rejects_train_resume_restore_mode() -> None:
    runner = Evaluate(
        model=_QuadraticModel(),
        sampler=_StaticSampler(torch.zeros(1, 2, 1, dtype=torch.float64)),
        hamiltonian_terms=[],
        diagnostics=[EnergyEvaluation()],
        load={"restore_mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(ValueError, match="load.restore_mode.*train_resume"):
        runner.run(_RecordingContext([]))


def test_train_train_resume_calls_runner_owned_restore(monkeypatch) -> None:
    calls = []

    def fake_restore_checkpoint(**kwargs):
        calls.append(kwargs)
        return RestoreReport(restore_mode="train_resume", checkpoint_dir="ckpt", step=4)

    monkeypatch.setattr(train_runner_module, "restore_checkpoint", fake_restore_checkpoint)
    runner = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"restore_mode": "train_resume", "path": "ckpt"},
    )

    result = runner.run(_RecordingContext([]))

    assert result.status == "completed"
    assert calls and calls[0]["model"] is runner.model
    assert calls[0]["trainer"] is runner.trainer
    assert calls[0]["sampler"] is runner.sampler


def test_evaluate_model_only_calls_runner_owned_restore(monkeypatch) -> None:
    calls = []

    def fake_restore_checkpoint(**kwargs):
        calls.append(kwargs)
        return RestoreReport(restore_mode="model_only", checkpoint_dir="ckpt", step=4)

    monkeypatch.setattr(evaluate_runner_module, "restore_checkpoint", fake_restore_checkpoint)
    runner = Evaluate(
        model=_QuadraticModel(),
        sampler=_StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64)),
        hamiltonian_terms={"constant": _ConstantEnergyTerm([1.0, 1.0])},
        diagnostics=[EnergyEvaluation()],
        load={"restore_mode": "model_only", "path": "ckpt"},
    )

    result = runner.run(_RecordingContext([]))

    assert result.status == "completed"
    assert calls and calls[0]["model"] is runner.model
    assert calls[0]["sampler"] is runner.sampler


def test_checkpoint_restore_mode_none_does_not_call_restore(monkeypatch) -> None:
    def fail_restore(**kwargs):
        raise AssertionError("restore_checkpoint should not be called")

    monkeypatch.setattr(train_runner_module, "restore_checkpoint", fail_restore)
    monkeypatch.setattr(evaluate_runner_module, "restore_checkpoint", fail_restore)

    train = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"restore_mode": "none"},
    )
    assert train.run(_RecordingContext([])).status == "completed"

    evaluate = Evaluate(
        model=_QuadraticModel(),
        sampler=_StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64)),
        hamiltonian_terms={"constant": _ConstantEnergyTerm([1.0, 1.0])},
        diagnostics=[EnergyEvaluation()],
        load={"restore_mode": "none"},
    )
    assert evaluate.run(_RecordingContext([])).status == "completed"


def _runner_context(cfg) -> RunContext:
    """Return a minimal real RunContext instance for private runner-instantiation tests."""

    context = object.__new__(RunContext)
    context.cfg = cfg
    return context


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
        self.metadata = SimpleNamespace(device="cpu", dtype="float64")
        self.records: list[tuple[str, dict]] = []

    def log(self, metrics, *, step=None, namespace="run", event=None) -> None:
        self.records.append((namespace, dict(metrics)))


class _QuadraticModel(nn.Module):
    """Simple wavefunction model for fast runner-level diagnostic tests."""

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        logabs = -flat.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _StaticSampler:
    """Sampler stub that returns one fixed walker batch and counts calls."""

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
    """Hamiltonian term returning fixed local-energy samples."""

    def __init__(self, values) -> None:
        self.values = torch.as_tensor(values, dtype=torch.float64)

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        values = self.values.to(device=batch.device, dtype=batch.dtype)
        return LocalEnergyResult(total=values, terms={"internal": values})


class _SharedContextProbe:
    """Diagnostic stub proving Evaluate passes shared state instead of resampling."""

    name = "probe"

    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[EvaluationContext] = []

    def evaluate(self, context: EvaluationContext) -> dict[str, int | float]:
        self.calls += 1
        self.contexts.append(context)
        assert context.local_energy_terms is not None
        assert context.local_energy.shape == context.wavefunction_output.logabs.shape
        assert list(context.hamiltonian_terms) == ["kinetic", "harmonic_trap"]
        return {
            "probe_batch_size": int(context.local_energy.numel()),
            "probe_logabs_sum": float(context.wavefunction_output.logabs.sum().item()),
        }


def test_evaluate_emits_lifecycle_events_through_run_context() -> None:
    recorder = _EventRecorder()
    context = _RecordingContext([recorder])
    runner = Evaluate(
        model=build_tiny_spenn(),
        sampler=build_tiny_sampler(),
        hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()],
        diagnostics=[EnergyEvaluation()],
        return_terms=True,
    )

    result = runner.run(context)

    assert result.status == "completed"
    assert recorder.events == [
        "run_start",
        "evaluate_start",
        "samples_collected",
        "diagnostic_start",
        "diagnostic_end",
        "evaluate_end",
        "run_end",
    ]
    # Energy metrics are emitted by the configured diagnostic.
    eval_records = [m for ns, m in context.records if ns == "eval"]
    assert eval_records
    assert "energy" in eval_records[-1]
    assert "energy_mean" not in eval_records[-1]
    assert "reference_energy" not in eval_records[-1]
    sampler_records = [m for ns, m in context.records if ns == "eval/sampler"]
    assert sampler_records
    assert "n_walkers" in sampler_records[-1]


def test_evaluate_runs_energy_diagnostics_from_shared_context_once() -> None:
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
    probe = _SharedContextProbe()
    runner = Evaluate(
        model=_QuadraticModel(),
        sampler=sampler,
        hamiltonian_terms={
            "kinetic": _ConstantEnergyTerm([1.0, 2.0, 3.0]),
            "harmonic_trap": _ConstantEnergyTerm([4.0, 5.0, 6.0]),
        },
        diagnostics=[probe, EnergyEvaluation(reference_energy=7.0, include_terms=True)],
        return_terms=True,
    )

    result = runner.run(context)

    assert result.status == "completed"
    assert sampler.calls == 1
    assert probe.calls == 1
    eval_records = [m for ns, m in context.records if ns == "eval"]
    assert len(eval_records) == 1
    metrics = eval_records[0]
    assert metrics["energy"] == pytest.approx(7.0)
    assert metrics["energy_error"] == pytest.approx(0.0)
    assert metrics["energy_abs_error"] == pytest.approx(0.0)
    assert metrics["energy_term_kinetic"] == pytest.approx(2.0)
    assert metrics["energy_term_harmonic_trap"] == pytest.approx(5.0)
    assert metrics["probe_batch_size"] == 3
    sampler_records = [m for ns, m in context.records if ns == "eval/sampler"]
    assert len(sampler_records) == 1
    assert sampler_records[0]["n_walkers"] == 3
    assert "energy_mean" not in metrics
    assert not any(key.startswith("sampler.") for key in metrics)


def test_energy_evaluation_fails_when_terms_were_not_returned() -> None:
    context = _RecordingContext([])
    sampler = _StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64))
    runner = Evaluate(
        model=_QuadraticModel(),
        sampler=sampler,
        hamiltonian_terms={"kinetic": _ConstantEnergyTerm([1.0, 2.0])},
        diagnostics=[EnergyEvaluation(include_terms=True)],
        return_terms=False,
    )

    with pytest.raises(ValueError, match="local_energy_terms"):
        runner.run(context)


def test_energy_evaluation_fails_loudly_when_local_energy_has_no_finite_samples() -> None:
    context = _RecordingContext([])
    sampler = _StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64))
    runner = Evaluate(
        model=_QuadraticModel(),
        sampler=sampler,
        hamiltonian_terms={"kinetic": _ConstantEnergyTerm([float("nan"), float("inf")])},
        diagnostics=[EnergyEvaluation()],
    )

    with pytest.raises(ValueError, match="no finite local-energy samples"):
        runner.run(context)


def _eval_metrics(run_root: Path) -> dict:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    eval_records = [record["metrics"] for record in records if record.get("namespace") == "eval"]
    assert eval_records, "no eval metric records were logged"
    return eval_records[-1]


def _namespace_records(run_root: Path, namespace: str) -> list[dict]:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    return [record["metrics"] for record in records if record.get("namespace") == namespace]


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

    # Reference-energy comparison is owned by the evaluation diagnostic.
    assert "reference_energy" not in metrics
    assert "abs_energy_error" not in metrics
    assert abs(metrics["energy_error"]) < energy_atol
    assert metrics["energy_abs_error"] < energy_atol
    assert metrics["local_energy_n_finite"] == metrics["local_energy_n_total"] == 512
    assert metrics["local_energy_finite_fraction"] == 1.0
    assert metrics["local_energy_nonfinite_count"] == 0
    assert abs(metrics["energy"] - exact_energy) < energy_atol
    assert metrics["energy_variance"] < variance_max
    # return_terms: true -> per-term decomposition is logged with configured names.
    for term in ("kinetic", "harmonic_trap", "electron_electron"):
        prefix = f"energy_term_{term}"
        assert prefix in metrics
        assert metrics[f"{prefix}_nonfinite_count"] == 0
    assert not any(key.startswith("sampler.") for key in metrics)
    # Sampler diagnostics own the eval/sampler namespace.
    sampler_metrics = _namespace_records(tmp_path, "eval/sampler")[-1]
    assert sampler_metrics["n_walkers"] == 512
    assert "acceptance_rate" in sampler_metrics
    eval_perf = _namespace_records(tmp_path, "eval/perf")[-1]
    assert "wall_time_sec" in eval_perf
    diagnostic_timing = _namespace_records(tmp_path, "diagnostics/energy")[-1]
    assert "time_sec" in diagnostic_timing
    runtime_metrics = _namespace_records(tmp_path, "runtime")
    assert any("wall_time_sec" in record for record in runtime_metrics)


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
        "metrics.jsonl",
        "metrics.csv",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
