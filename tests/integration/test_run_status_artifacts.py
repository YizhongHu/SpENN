"""End-to-end artifacts written from the typed run lifecycle.

WHY THESE ARE INTEGRATION TESTS AND NOT UNIT ONES. ``status.json`` and the
empty-suite ``diagnostics/index.json`` are written by `tpen.callback.Status` and
`tpen.callback.ArtifactIndex` from a subscription group declaring
`tpen.callback.cadence.SubscriptionGroup.stateless`, and those two are the first
consumers of that mechanism anywhere. Three separate things have to hold for a
byte to reach disk: `tpen.run.run_from_config` has to emit the boundary,
`tpen.artifacts.RunContext._dispatch_occurrence` has to route a state-free
occurrence to a `StatefulCallback` rather than skip it, and the callback has to
answer on ``handle_stateless_occurrence_impl``. A unit test that hands an
occurrence to the callback by hand proves only the third, and the first two are
exactly the ones that fail SILENTLY -- a missing emit and a skipped dispatch both
look like a passing suite and an absent file.

The failure cases matter most. Runs fail on a cluster far more often than a
green test suite is read, and before these landed nothing at any level asserted
that ``status.json`` records a failed run at all: the unit tests drove only the
success boundaries, and ``test_run_fail_loudness.py`` asserts ``error.json`` and
``events.jsonl`` but never ``status.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from tpen.run import run_from_config

_STATUS_CALLBACK = {
    "_target_": "tpen.callback.Status",
    "output_path": "${run.dir}/status.json",
    # Off so the status boxes do not flood the test log; the artifact is what is
    # under test, and the terminal line has its own unit coverage.
    "terminal": False,
}

_ARTIFACT_INDEX_CALLBACK = {"_target_": "tpen.callback.ArtifactIndex"}


def _cfg(tmp_path: Path, **extra) -> DictConfig:
    base = {
        "experiment": {"name": "s", "sector": "s", "run_name": "s"},
        "run": {"root": str(tmp_path), "run_id": None, "dir": None},
        "runtime": {"seed": 0},
        "terminal": {"enabled": False},
        "callbacks": [_STATUS_CALLBACK, _ARTIFACT_INDEX_CALLBACK],
        "evaluator": {
            "_target_": "tpen.evaluation.Evaluator",
            "namespace": "eval",
            "tasks": [],
        },
        "runner": {
            "_target_": "tpen.runner.Evaluate",
            "model": {"_target_": "torch.nn.Identity"},
            "evaluator": "${evaluator}",
        },
    }
    base.update(extra)
    return OmegaConf.create(base)


def _run_dir(tmp_path: Path) -> Path:
    run_dirs = list(tmp_path.glob("s/s/*"))
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


def test_a_successful_run_records_completed_and_an_empty_index(tmp_path: Path) -> None:
    """The success path, through the real harness, for both callbacks at once.

    The empty suite is the whole reason `ArtifactIndex` observes a run boundary:
    every run with at least one task rewrites the index from the task boundary,
    so this is the only shape where the run-level subscription is what puts the
    file on disk.
    """

    assert run_from_config(_cfg(tmp_path), config_path="s.yaml", command="pytest") == 0

    run_dir = _run_dir(tmp_path)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["current_event"] == "run_end"
    assert status["start_time"] is not None
    assert status["end_time"] is not None
    assert status["exception_type"] is None
    assert status["exception_message"] is None

    assert json.loads((run_dir / "diagnostics" / "index.json").read_text()) == {"tasks": []}


def test_a_run_that_fails_inside_the_runner_records_failed(tmp_path: Path) -> None:
    """A failed run must still say so on disk, with the failure's identity.

    ``run_end`` and ``exception`` were two separate legacy strings, and mapping
    both onto `tpen.run_events.RunCompleted` would drop this case while leaving
    every success-path test green.
    """

    missing = tmp_path / "missing" / "latest.json"
    cfg = _cfg(
        tmp_path,
        load={
            "path": str(missing),
            "mode": "model_only",
            "strict": True,
            "allow_protocol_mismatch": False,
        },
        runner={
            "_target_": "tpen.runner.Evaluate",
            "model": None,
            "load": "${load}",
            "evaluator": "${evaluator}",
        },
    )

    assert run_from_config(cfg, config_path="s.yaml", command="pytest") == 1

    run_dir = _run_dir(tmp_path)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["current_event"] == "exception"
    # Not merely non-null: the identity is what makes the artifact worth reading,
    # and it now travels as typed fields on `RunFailed` rather than as a live
    # exception object in an untyped payload.
    assert status["exception_type"] == "FileNotFoundError"
    assert str(missing) in status["exception_message"]
    assert status["start_time"] is not None
    assert status["end_time"] is not None

    # The runner never returned, so the completion boundary never happened and
    # the index was never written -- the same as before this migration, where
    # ``run_end`` was the last statement of ``Evaluate.run`` rather than a
    # ``finally``.
    assert not (run_dir / "diagnostics" / "index.json").exists()


def test_a_run_that_fails_before_the_runner_exists_still_records_failed(
    tmp_path: Path,
) -> None:
    """`RunStarted` fires before the runner is built, and `RunFailed` after.

    `tpen.run.run_from_config` emits `RunStarted` and only then instantiates the
    runner, so a configuration error lands between the two boundaries: the run
    is recorded as ``running`` and then as ``failed``, and never reaches a
    completion boundary at all.
    """

    cfg = _cfg(tmp_path, runner={"_target_": "tpen.runner.Evaluate", "model": None})

    assert run_from_config(cfg, config_path="s.yaml", command="pytest") == 1

    status = json.loads((_run_dir(tmp_path) / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["exception_type"] == "InstantiationException"
    assert "evaluator" in status["exception_message"]
