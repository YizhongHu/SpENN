"""Tests for the generic Hooke scaffold run plumbing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from spenn.callback import Callback, ConfigSnapshot, Event, Metadata, ReportSkeleton, Status
from spenn.logging import CSV, JSONL, LogRecord, Logger
from spenn.runner import Runner, Scaffold
from spenn.training.artifacts import REQUIRED_RUN_DIRS, RunContext, RunResult
from spenn.training.run import load_config, prepare_run_context, run_from_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "hooke" / "configs" / "smoke" / "scaffold.yaml"


class RecordingCallback(Callback):
    """Record handled event names for callback registry tests."""

    def __init__(self, triggers: list[str]) -> None:
        super().__init__(triggers)
        self.events: list[str] = []

    def on_run_start(self, event: Event) -> None:
        """Record run start."""

        self.events.append(event.name)

    def on_run_end(self, event: Event) -> None:
        """Record run end."""

        self.events.append(event.name)


class FailingRunner(Runner):
    """Runner target that fails after run start for exception-path tests."""

    def run(self, context: RunContext) -> RunResult:
        """Emit run start and fail deliberately."""

        self.emit("run_start", context)
        raise RuntimeError("intentional scaffold failure")


class FinishMarker(Logger):
    """Logger that writes a marker file when finished."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, record: LogRecord) -> None:
        """Ignore metric records."""

        del record

    def finish(self) -> None:
        """Write the finish marker."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("finished\n", encoding="utf-8")


def test_smoke_config_prepares_artifact_context(tmp_path: Path) -> None:
    """The smoke config resolves only generic run-management targets."""

    cfg = load_config(str(CONFIG), [f"run.root={tmp_path}"])
    for forbidden in ("system", "model", "hamiltonian", "sampler", "trainer", "diagnostics"):
        assert forbidden not in cfg

    context = prepare_run_context(cfg, config_path=str(CONFIG), command="pytest scaffold")

    assert context.run_dir.parent == tmp_path / "hooke_scaffold" / "scaffold"
    assert context.cfg.run.run_id
    assert context.cfg.run.dir == str(context.run_dir)
    assert {path.name for path in context.run_dir.iterdir()} == set(REQUIRED_RUN_DIRS)
    assert [type(logger) for logger in context.loggers] == [CSV, JSONL]
    assert [type(callback) for callback in context.callbacks] == [
        ConfigSnapshot,
        Metadata,
        Status,
        ReportSkeleton,
    ]


def test_flat_public_targets_instantiate() -> None:
    """Hydra can instantiate every flat public scaffold target."""

    runner = instantiate(
        OmegaConf.create(
            {
                "_target_": "spenn.runner.Scaffold",
                "callbacks": [],
                "loggers": [],
            }
        )
    )
    callback = instantiate(
        OmegaConf.create(
            {
                "_target_": "spenn.callback.Status",
                "triggers": ["run_start"],
                "output_path": "status.json",
            }
        )
    )
    csv = instantiate(OmegaConf.create({"_target_": "spenn.logging.CSV", "path": "metrics.csv"}))
    jsonl = instantiate(OmegaConf.create({"_target_": "spenn.logging.JSONL", "path": "metrics.jsonl"}))

    assert isinstance(runner, Scaffold)
    assert isinstance(callback, Status)
    assert isinstance(csv, CSV)
    assert isinstance(jsonl, JSONL)


def test_runner_callback_registry_dispatches_matching_events(tmp_path: Path) -> None:
    """Runner callback dispatch uses subscribed event names."""

    callback = RecordingCallback(["run_start"])
    runner = Scaffold(callbacks=[callback], loggers=[])
    cfg = load_config(str(CONFIG), [f"run.root={tmp_path}"])
    context = prepare_run_context(cfg, config_path=str(CONFIG), command="pytest registry")

    runner.emit("run_start", context)
    runner.emit("run_end", context)

    assert callback.events == ["run_start"]


def test_scaffold_run_writes_lifecycle_artifacts(tmp_path: Path) -> None:
    """The scaffold runner writes the required artifact skeleton."""

    code = run_from_config(load_config(str(CONFIG), [f"run.root={tmp_path}"]), config_path=str(CONFIG), command="pytest run")

    assert code == 0
    run_dir = _single_run_dir(tmp_path)
    assert {path.name for path in run_dir.iterdir()} >= {
        "checkpoints",
        "traces",
        "diagnostics",
        "figures",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.csv",
        "metrics.jsonl",
        "report.md",
    }

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metric_rows = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    metric_records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    resolved = OmegaConf.load(run_dir / "resolved_config.yaml")
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert status["status"] == "completed"
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == run_dir.name
    assert metadata["run_dir"] == str(run_dir)
    assert metadata["config_path"] == str(CONFIG)
    assert metric_rows == [
        "step,namespace,key,value",
        "0,scaffold,scaffold_completed,true",
    ]
    assert metric_records == [
        {
            "event": None,
            "metrics": {"scaffold_completed": True},
            "namespace": "scaffold",
            "step": 0,
        }
    ]
    assert resolved.run.dir == str(run_dir)
    assert resolved.run.run_id == run_dir.name
    assert "No Hooke physics" in report


def test_train_cli_runs_scaffold_config(tmp_path: Path) -> None:
    """The public train.py command delegates to the generic launcher."""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            str(CONFIG),
            f"run.root={tmp_path}",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = _single_run_dir(tmp_path)
    assert (run_dir / "status.json").exists()
    assert (run_dir / "metrics.jsonl").exists()


def test_exception_path_records_failure_and_finishes_loggers(tmp_path: Path) -> None:
    """Run failures write failed status and still close loggers."""

    cfg = OmegaConf.create(
        {
            "experiment": {
                "name": "hooke_scaffold",
                "sector": "scaffold",
                "run_name": "hooke_scaffold",
            },
            "run": {
                "root": str(tmp_path),
                "run_id": None,
                "dir": None,
            },
            "runtime": {
                "device": "cpu",
                "dtype": "float64",
            },
            "runner": {
                "_target_": "tests.hooke.test_scaffold.FailingRunner",
                "callbacks": "${callbacks}",
                "loggers": "${loggers}",
            },
            "loggers": [
                {
                    "_target_": "tests.hooke.test_scaffold.FinishMarker",
                    "path": "${run.dir}/finished.txt",
                }
            ],
            "callbacks": [
                {
                    "_target_": "spenn.callback.Metadata",
                    "triggers": ["run_start", "run_end", "exception"],
                    "output_path": "${run.dir}/metadata.json",
                },
                {
                    "_target_": "spenn.callback.Status",
                    "triggers": ["run_start", "run_end", "exception"],
                    "output_path": "${run.dir}/status.json",
                },
            ],
        }
    )

    code = run_from_config(cfg, config_path="inline", command="pytest failure")

    assert code == 1
    run_dir = _single_run_dir(tmp_path)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["exception_type"] == "RuntimeError"
    assert status["exception_message"] == "intentional scaffold failure"
    assert metadata["status"] == "failed"
    assert (run_dir / "finished.txt").read_text(encoding="utf-8") == "finished\n"


def _single_run_dir(root: Path) -> Path:
    run_dirs = sorted((root / "hooke_scaffold" / "scaffold").iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]
