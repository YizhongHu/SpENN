"""Unit tests for the optional Parsl allocation-attach executor."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from experiments.toolkit.dispatch import AllocationContext, DispatchExecutor, DispatchSpec
from experiments.toolkit.parsl_attach import ParslAttachExecutor, _parsl_app_runner, _run_dispatch_payload
from experiments.toolkit.specs import CompletionSpec


def _dispatch(tmp_path: Path, *, completion: CompletionSpec | None = None) -> DispatchSpec:
    return DispatchSpec(
        logical_task_id="logical-1",
        admission_id="admission-1",
        attempt_id="attempt-1",
        stage="train",
        run_id="run-1",
        argv=(sys.executable, "-c", "print('ok')"),
        result_dir=str(tmp_path / "result"),
        runtime="test",
        cwd=str(tmp_path),
        environment={"EXTRA": "one"},
        completion=completion or CompletionSpec(policy="none"),
    )


def _context(tmp_path: Path, **overrides: Any) -> AllocationContext:
    values: dict[str, Any] = {
        "allocation_id": "allocation-1",
        "visibility_variable": "CUDA_VISIBLE_DEVICES",
        "visibility_values": ("0",),
        "run_root": str(tmp_path / "launch"),
    }
    values.update(overrides)
    return AllocationContext(**values)


def test_protocol_records_verbatim_argv_and_worker_layout(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True)
        status = output / "attempt_status.json"
        status.write_text(json.dumps({"status": "success"}) + "\n")
        (output / "stdout.log").write_text("")
        (output / "stderr.log").write_text("")
        return {"returncode": 0, "attempt_status_path": str(status)}

    spec = _dispatch(tmp_path)
    executor = ParslAttachExecutor(app_runner=fake_runner)
    assert isinstance(executor, DispatchExecutor)
    records = executor.dispatch((spec,), context=_context(tmp_path))

    assert records[0].submitted_command == spec.argv
    assert spec.argv == (sys.executable, "-c", "print('ok')")
    assert records[0].backend == "parsl_attach"
    assert records[0].launcher_job_id == "allocation-1"
    assert calls[0]["argv"] == spec.argv
    assert calls[0]["environment"]["EXTRA"] == "one"
    assert "CUDA_VISIBLE_DEVICES" not in calls[0]["environment"]
    assert calls[0]["visibility_value"] == "0"
    assert Path(calls[0]["output_directory"]) == tmp_path / "launch" / "dispatch" / spec.attempt_id


def test_completion_rechecked_after_success(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    spec = _dispatch(tmp_path, completion=CompletionSpec(policy="file_exists", output_paths=(str(missing),)))

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True)
        status = output / "attempt_status.json"
        status.write_text(json.dumps({"status": "success"}) + "\n")
        return {"returncode": 0, "attempt_status_path": str(status)}

    with pytest.raises(RuntimeError, match="completion predicate"):
        ParslAttachExecutor(app_runner=fake_runner).dispatch((spec,), context=_context(tmp_path))
    status = json.loads((tmp_path / "launch" / "dispatch" / spec.attempt_id / "attempt_status.json").read_text())
    assert status["status"] == "failed"
    assert "completion_error" in status


def test_deadline_guard_refuses_new_dispatches(tmp_path: Path) -> None:
    spec = _dispatch(tmp_path)
    context = _context(tmp_path, deadline=time.time() + 1, deadline_guard_min=1)
    with pytest.raises(RuntimeError, match="deadline guard"):
        ParslAttachExecutor(app_runner=lambda **kwargs: pytest.fail("must not run")).dispatch(
            (spec,), context=context
        )


def test_stdlib_worker_writes_output_layout(tmp_path: Path) -> None:
    output = tmp_path / "dispatch" / "attempt-1"
    payload = _run_dispatch_payload(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ['EXTRA']); print(os.environ['CUDA_VISIBLE_DEVICES'])",
        ),
        str(tmp_path),
        {"EXTRA": "one"},
        "CUDA_VISIBLE_DEVICES",
        "0",
        str(output),
        "attempt-1",
    )
    assert payload["returncode"] == 0
    assert (output / "stdout.log").read_text() == "one\n0\n"
    assert (output / "stderr.log").is_file()
    assert json.loads((output / "attempt_status.json").read_text())["visibility_value"] == "0"


def test_parsl_config_has_no_retries_and_accelerator_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parsl = pytest.importorskip("parsl")
    captured: dict[str, Any] = {}

    def fake_load(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(parsl, "load", fake_load)
    _parsl_app_runner(_context(tmp_path), tmp_path / "launch")
    config = captured["config"]
    executor = config.executors[0]
    assert config.retries == 0
    assert config.run_dir == str(tmp_path / "launch" / "parsl")
    assert config.usage_tracking is False
    assert executor.max_workers_per_node == 1
    assert executor.available_accelerators == ["0"]

    _parsl_app_runner(
        _context(tmp_path, visibility_values=()),
        tmp_path / "inherit-launch",
    )
    inherit_executor = captured["config"].executors[0]
    assert inherit_executor.max_workers_per_node == 1
    assert inherit_executor.available_accelerators == []
