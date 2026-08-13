"""Tests for the scheduler-neutral pair-v1 allocation launcher."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.toolkit import (
    AllocationPoolExecutor,
    CompletionSpec,
    ExecutionRecord,
    ResourceSpec,
    StagePlan,
    SubmissionRequest,
    TaskSpec,
    task_id_from_parts,
)


_LAUNCH_PATH = Path(__file__).with_name("launch.py")
_SPEC = importlib.util.spec_from_file_location("tpen_pair_v1_launch", _LAUNCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
launch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launch)


def _args(tmp_path: Path, **overrides: object):
    values = {
        "python": sys.executable,
        "results_root": str(tmp_path / "results"),
        "run_id": "smoke-001",
        "device": "cuda",
        "visibility_variable": "CUDA_VISIBLE_DEVICES",
        "visibility_values": ["0"],
        "deadline": None,
        "deadline_env_var": None,
        "pass_id": "pass-test",
        "allocation_id": "allocation-test",
        "dry_run": False,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_dry_run_has_config_and_device_overrides_for_cuda_and_xpu(tmp_path: Path) -> None:
    for device in ("cuda", "xpu"):
        plan, request = launch.build_plan(_args(tmp_path, device=device), checkout=tmp_path / "checkout")
        command = plan.tasks[0].command
        assert command == (
            sys.executable,
            "run.py",
            "--config",
            launch.CONFIG_PATH,
            f"run.root={Path(tmp_path / 'results').resolve()}",
            "run.run_id=smoke-001",
            f"runtime.device={device}",
        )
        assert request.submitted_commands == (command,)
        expected_run_dir = (
            Path(tmp_path / "results").resolve()
            / "tpen_pair_v1"
            / "pair"
            / "smoke-001"
        )
        assert plan.tasks[0].result_dir == str(expected_run_dir)
        assert plan.tasks[0].logs == (str(expected_run_dir / "launcher-status.json"),)
        assert plan.tasks[0].completion == CompletionSpec(
            policy="status_completed",
            status_path=str(expected_run_dir / "status.json"),
        )
        assert plan.results_root == str(Path(tmp_path / "results").resolve())


def test_dry_run_prints_the_submitted_argv_without_executing_it(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(launch, "AllocationPoolExecutor", lambda **_: pytest.fail("executed"))
    assert launch.main(
        [
            "--python",
            "/overlay/bin/python",
            "--results-root",
            str(tmp_path / "results"),
            "--run-id",
            "dry-run-001",
            "--device",
            "xpu",
            "--visibility-variable",
            "ZE_AFFINITY_MASK",
            "--visibility-values",
            "0",
            "--allocation-id",
            "allocation-dry-run",
            "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "/overlay/bin/python run.py --config " + launch.CONFIG_PATH in output
    assert "run.run_id=dry-run-001" in output
    assert "runtime.device=xpu" in output


def test_results_root_and_worker_bindings_are_validated(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    with pytest.raises(ValueError, match="outside repository"):
        launch.build_plan(_args(tmp_path, results_root=str(checkout / "results")), checkout=checkout)
    with pytest.raises(ValueError, match="single task on a single device"):
        launch.build_plan(_args(tmp_path, visibility_values=["0", "1"]), checkout=checkout)
    with pytest.raises(ValueError, match="outside repository"):
        launch.build_plan(_args(tmp_path, results_root="."), checkout=Path.cwd())


def test_deadline_arguments_are_forwarded_to_allocation_executor(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def submit(self, plan, tasks, request):
            captured["plan"] = plan
            captured["tasks"] = tasks
            captured["request"] = request
            return ()

    monkeypatch.setattr(launch, "AllocationPoolExecutor", FakeExecutor)
    assert launch.main(
        [
            "--python",
            sys.executable,
            "--results-root",
            str(tmp_path / "results"),
            "--run-id",
            "deadline-smoke",
            "--device",
            "cuda",
            "--visibility-variable",
            "CUDA_VISIBLE_DEVICES",
            "--visibility-values",
            "0",
            "--allocation-id",
            "allocation-deadline",
            "--deadline",
            "2030-01-01T00:00:00Z",
            "--deadline-env-var",
            "PBS_WALLTIME",
        ]
    ) == 0
    assert captured["deadline"] == "2030-01-01T00:00:00Z"
    assert captured["deadline_env_var"] == "PBS_WALLTIME"
    assert captured["n_workers"] == 1
    assert captured["visibility_values"] == ("0",)
    assert captured["allocation_id"] == "allocation-deadline"


@pytest.mark.parametrize("returncode, expected", [(7, 1), (0, 0), (None, 1)])
def test_execution_record_returncode_controls_launcher_exit(
    tmp_path: Path, monkeypatch, returncode: int | None, expected: int
) -> None:
    args = _args(tmp_path, run_id="failed-task", pass_id="pass-1")
    plan, request = launch.build_plan(args, checkout=tmp_path / "checkout")
    task = plan.tasks[0]
    record = ExecutionRecord(
        task_id=task.task_id,
        run_id=task.run_id,
        stage=task.stage,
        attempt_id=task.attempt_id,
        backend="allocation_pool",
        launcher_job_id="allocation-test",
        submitted_command=task.command,
        metadata={"returncode": returncode},
    )

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, submitted_plan, tasks, submitted_request):
            assert submitted_plan == plan
            assert tasks == (task,)
            assert submitted_request == request
            return (record,)

    monkeypatch.setattr(launch, "AllocationPoolExecutor", FakeExecutor)
    assert launch.main(
        [
            "--python",
            sys.executable,
            "--results-root",
            str(tmp_path / "results"),
            "--run-id",
            "failed-task",
            "--device",
            "cuda",
            "--visibility-variable",
            "CUDA_VISIBLE_DEVICES",
            "--visibility-values",
            "0",
            "--allocation-id",
            "allocation-test",
        ]
    ) == expected


@pytest.mark.parametrize(
    "shape",
    [
        "non-iterable",
        "not-an-execution-record",
        "missing-metadata",
        "none-metadata",
        "non-mapping-metadata",
        "missing-returncode",
        "bool-returncode",
        "float-returncode",
        "nonzero-returncode",
        "mixed-valid-invalid",
        "mixed-success-failure",
    ],
)
def test_malformed_execution_receipts_fail_closed(
    tmp_path: Path, monkeypatch, shape: str
) -> None:
    args = _args(tmp_path, run_id=f"malformed-{shape}", pass_id="pass-1")
    plan, request = launch.build_plan(args, checkout=tmp_path / "checkout")
    task = plan.tasks[0]
    valid = ExecutionRecord(
        task_id=task.task_id,
        run_id=task.run_id,
        stage=task.stage,
        attempt_id=task.attempt_id,
        backend="allocation_pool",
        launcher_job_id="allocation-test",
        submitted_command=task.command,
        metadata={"returncode": 0},
    )
    if shape == "non-iterable":
        records = 1
    elif shape == "not-an-execution-record":
        records = (object(),)
    elif shape == "missing-metadata":
        records = (
            ExecutionRecord(
                task_id=task.task_id,
                run_id=task.run_id,
                stage=task.stage,
                attempt_id=task.attempt_id,
                backend="allocation_pool",
                launcher_job_id="allocation-test",
                submitted_command=task.command,
            ),
        )
    elif shape == "none-metadata":
        records = (valid.__class__(**{**valid.__dict__, "metadata": None}),)
    elif shape == "non-mapping-metadata":
        records = (valid.__class__(**{**valid.__dict__, "metadata": ["returncode", 0]}),)
    elif shape == "missing-returncode":
        records = (valid.__class__(**{**valid.__dict__, "metadata": {}}),)
    elif shape == "bool-returncode":
        records = (valid.__class__(**{**valid.__dict__, "metadata": {"returncode": False}}),)
    elif shape == "float-returncode":
        records = (valid.__class__(**{**valid.__dict__, "metadata": {"returncode": 0.0}}),)
    elif shape == "nonzero-returncode":
        records = (valid.__class__(**{**valid.__dict__, "metadata": {"returncode": 2}}),)
    elif shape == "mixed-valid-invalid":
        records = (valid, object())
    else:
        failed = valid.__class__(**{**valid.__dict__, "metadata": {"returncode": 2}})
        records = (valid, failed)

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, submitted_plan, tasks, submitted_request):
            assert submitted_plan == plan
            assert tasks == (task,)
            assert submitted_request == request
            return records

    monkeypatch.setattr(launch, "AllocationPoolExecutor", FakeExecutor)
    assert launch.main(
        [
            "--python", sys.executable,
            "--results-root", str(tmp_path / "results"),
            "--run-id", f"malformed-{shape}",
            "--device", "cuda",
            "--visibility-variable", "CUDA_VISIBLE_DEVICES",
            "--visibility-values", "0",
            "--allocation-id", "allocation-test",
        ]
    ) == 1


def test_empty_and_exact_zero_execution_receipts_succeed(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path, run_id="empty-or-zero")
    plan, request = launch.build_plan(args, checkout=tmp_path / "checkout")
    task = plan.tasks[0]
    record = ExecutionRecord(
        task_id=task.task_id,
        run_id=task.run_id,
        stage=task.stage,
        attempt_id=task.attempt_id,
        backend="allocation_pool",
        launcher_job_id="allocation-test",
        submitted_command=task.command,
        metadata={"returncode": 0},
    )
    batches = iter(((), (record,)))

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, submitted_plan, tasks, submitted_request):
            return next(batches)

    monkeypatch.setattr(launch, "AllocationPoolExecutor", FakeExecutor)
    common = [
        "--python", sys.executable, "--results-root", str(tmp_path / "results"),
        "--run-id", "empty-or-zero", "--device", "cuda",
        "--visibility-variable", "CUDA_VISIBLE_DEVICES", "--visibility-values", "0",
        "--allocation-id", "allocation-test",
    ]
    assert launch.main(common) == 0
    assert launch.main(common) == 0


def test_single_visibility_value_binds_one_deterministic_worker(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def submit(self, plan, tasks, request):
            return ()

    monkeypatch.setattr(launch, "AllocationPoolExecutor", FakeExecutor)
    launch.main(
        [
            "--python",
            sys.executable,
            "--results-root",
            str(tmp_path / "results"),
            "--run-id",
            "one-device",
            "--device",
            "cuda",
            "--visibility-variable",
            "CUDA_VISIBLE_DEVICES",
            "--visibility-values",
            "3",
            "--allocation-id",
            "allocation-one-device",
        ]
    )
    assert captured["n_workers"] == 1
    assert captured["visibility_values"] == ("3",)
    assert captured["allocation_id"] == "allocation-one-device"


def test_interpreter_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        launch.build_plan(_args(tmp_path, python="python"), checkout=tmp_path / "checkout")


def _pool_task(tmp_path: Path, run_id: str, script: Path, output: Path) -> TaskSpec:
    stage = "01_train"
    attempt = "A1"
    command = (sys.executable, str(script), str(output))
    return TaskSpec(
        task_id=task_id_from_parts(stage=stage, run_id=run_id, attempt_id=attempt),
        stage=stage,
        attempt_id=attempt,
        run_id=run_id,
        command=command,
        result_dir=str(tmp_path / run_id),
        logs=(str(tmp_path / f"{run_id}.json"),),
        resources=ResourceSpec(profile="gpu", device="cuda", gpus=1),
        completion=CompletionSpec(policy="none"),
    )


def test_real_pool_propagates_each_worker_visibility_value(tmp_path: Path) -> None:
    script = tmp_path / "fake.py"
    script.write_text(
        "import os, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(os.environ['TEST_VISIBLE'])\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    tasks = tuple(
        _pool_task(tmp_path, f"run-{index}", script, tmp_path / f"out-{index}")
        for index in range(2)
    )
    plan = StagePlan(
        study="launcher-test",
        stage="01_train",
        attempt_id="A1",
        results_root=str(tmp_path / "results"),
        tasks=tasks,
    )
    request = SubmissionRequest(
        command_sets={"allocation": tuple(task.command for task in tasks)},
        submitted_commands=tuple(task.command for task in tasks),
    )
    records = AllocationPoolExecutor(
        pass_id="visibility-pass",
        n_workers=2,
        visibility_variable="TEST_VISIBLE",
        visibility_values=("worker-a", "worker-b"),
        run_root=tmp_path / "claims",
        deadline_guard_min=0,
    ).submit(plan, tasks, request)
    assert len(records) == 2
    assert {path.read_text() for path in tmp_path.glob("out-*")} == {"worker-a", "worker-b"}


def test_launcher_has_no_scheduler_invocation(monkeypatch, tmp_path: Path) -> None:
    seen = []

    def fail_scheduler(*args, **kwargs):
        seen.append((args, kwargs))
        raise AssertionError("scheduler command invoked")

    monkeypatch.setattr(launch, "AllocationPoolExecutor", fail_scheduler)
    assert launch.main(
        [
            "--python",
            sys.executable,
            "--results-root",
            str(tmp_path / "results"),
            "--run-id",
            "dry-run",
            "--device",
            "xpu",
            "--visibility-variable",
            "ZE_AFFINITY_MASK",
            "--visibility-values",
            "0",
            "--allocation-id",
            "allocation-dry-run",
            "--dry-run",
        ]
    ) == 0
    assert seen == []
