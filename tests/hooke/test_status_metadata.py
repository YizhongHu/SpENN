"""Status and metadata tests for configured runs."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from spenn.artifacts import RunContext, RunResult
from spenn.logging import LogRecord, Logger
from spenn.runner import Runner
from spenn.run import run_from_config


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
                "_target_": "tests.hooke.test_status_metadata.FailingRunner",
                "callbacks": "${callbacks}",
                "loggers": "${loggers}",
            },
            "loggers": [
                {
                    "_target_": "tests.hooke.test_status_metadata.FinishMarker",
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
