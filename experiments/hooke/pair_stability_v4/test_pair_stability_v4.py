"""Focused V4-0 dispatcher and artifact-contract tests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]

while str(STUDY_DIR) in sys.path:
    sys.path.remove(str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR))
for module_name in ("audit", "dispatch"):
    sys.modules.pop(module_name, None)

import dispatch  # noqa: E402
import fanout_audit  # noqa: E402
import roots  # noqa: E402
import routes  # noqa: E402


def test_every_role_prepares_from_real_shaped_upstream_artifacts(
    tmp_path: Path,
) -> None:
    """All ten routes independently resolve their exact recorded prerequisites."""

    lineage = "lineage-a"
    root = roots.initialize_root(
        (tmp_path / "candidate").absolute(),
        lineage_id=lineage,
    )
    _write_upstream_fixture(root, lineage)

    for role, route in routes.load_routes().items():
        inputs = {name: lineage for name in route.required_input_attempts}
        prepared_root, prepared_inputs, configs = dispatch._prepare_dispatch(
            route,
            results_root=root,
            output_attempt=lineage,
            input_attempts=inputs,
            purpose=roots.PURPOSE_EXPERIMENT,
        )
        argv = routes.render_legacy_argv(
            route,
            results_root=prepared_root,
            output_attempt=lineage,
            input_attempts=prepared_inputs,
            config_paths=configs,
            repo_root=REPO_ROOT,
        )

        assert argv[2] == str(REPO_ROOT / route.legacy_script)
        assert all(path.is_file() for path in configs.values())


def test_gpu_test_cap_is_checked_before_every_fanout_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile above two gpu_test array tasks cannot reach route preparation."""

    assert fanout_audit.gpu_test_array_task_count(population=64, chunk_size=32) == 2
    assert fanout_audit.gpu_test_array_task_count(population=8, chunk_size=8) == 1
    loaded = dict(routes.load_routes())
    route = loaded["screen_train"]
    arguments = list(route.arguments)
    arguments[arguments.index("--chunk-size") + 1] = "16"
    loaded["screen_train"] = replace(route, arguments=tuple(arguments))
    monkeypatch.setattr(fanout_audit, "load_routes", lambda: loaded)
    errors = fanout_audit.audit_gpu_test_fanout_profile()
    assert any("submits 4 gpu_test array tasks, cap is 2" in error for error in errors)

    lineage = "lineage-a"
    root = roots.initialize_root(
        (tmp_path / "candidate").absolute(),
        lineage_id=lineage,
    )
    _write_upstream_fixture(root, lineage)
    monkeypatch.setattr(dispatch.audit, "audit_gpu_test_fanout_profile", lambda: errors)
    with pytest.raises(ValueError, match="before fan-out submission"):
        dispatch._prepare_dispatch(
            routes.load_routes()["screen_train"],
            results_root=root,
            output_attempt=lineage,
            input_attempts={"grid": lineage},
            purpose=roots.PURPOSE_EXPERIMENT,
        )


def test_report_requires_real_final_collect_manifest_yaml(tmp_path: Path) -> None:
    """The report route follows the v3 manifest.yaml contract exactly."""

    lineage = "lineage-a"
    root = roots.initialize_root(
        (tmp_path / "candidate").absolute(),
        lineage_id=lineage,
    )
    final_collect = root / "08_final_collect" / lineage
    final_collect.mkdir(parents=True)
    (final_collect / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="final_collect upstream"):
        dispatch._prepare_dispatch(
            routes.load_routes()["report"],
            results_root=root,
            output_attempt=lineage,
            input_attempts={"final_collect": lineage},
            purpose=roots.PURPOSE_EXPERIMENT,
        )

    (final_collect / "manifest.yaml").write_text(
        "study: pair_stability_v4\nstage: 08_final_collect\n"
    )
    _, inputs, configs = dispatch._prepare_dispatch(
        routes.load_routes()["report"],
        results_root=root,
        output_attempt=lineage,
        input_attempts={"final_collect": lineage},
        purpose=roots.PURPOSE_EXPERIMENT,
    )
    assert inputs == {"final_collect": lineage}
    assert configs == {}


def test_dispatch_receipts_record_success_source_identities_and_exact_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success writes immutable request/result receipts around the subprocess."""

    lineage = "lineage-a"
    root = roots.initialize_root(
        (tmp_path / "candidate").absolute(),
        lineage_id=lineage,
    )
    captured: list[tuple[list[str], Path, dict[str, str]]] = []
    _stub_receipt_sources(monkeypatch)

    def completed(
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> SimpleNamespace:
        captured.append((argv, cwd, env))
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatch.subprocess, "run", completed)

    assert dispatch.dispatch_stage(
        "screen_plan",
        results_root=root,
        output_attempt=lineage,
        input_attempts={},
    ) == 0

    receipt_dir = _only_receipt(root, lineage, "screen_plan")
    request = json.loads((receipt_dir / "request.json").read_text())
    result = json.loads((receipt_dir / "result.json").read_text())
    assert request["argv"] == captured[0][0]
    assert captured[0][1] == REPO_ROOT
    assert captured[0][2]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert result["status"] == "completed"
    assert result["returncode"] == 0
    assert result["argv"] == request["argv"]
    assert result["cwd"] == request["cwd"]
    assert result["legacy_source"] == {"closure_sha256": "legacy"}
    assert result["runtime_source"] == {"closure_sha256": "runtime"}
    assert result["config_source"] == {"closure_sha256": "config"}


@pytest.mark.parametrize(
    ("returncode", "status", "signal"),
    [((7), "failed", None), ((-15), "failed", 15)],
)
def test_dispatch_receipt_preserves_nonzero_and_signal_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    status: str,
    signal: int | None,
) -> None:
    """Terminal child failures are recorded without output repair."""

    lineage = "lineage-a"
    root = roots.initialize_root(
        (tmp_path / f"candidate-{returncode}").absolute(),
        lineage_id=lineage,
    )
    _stub_receipt_sources(monkeypatch)
    monkeypatch.setattr(
        dispatch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode),
    )

    assert dispatch.dispatch_stage(
        "screen_plan",
        results_root=root,
        output_attempt=lineage,
        input_attempts={},
    ) == returncode
    result = json.loads(
        (
            _only_receipt(root, lineage, "screen_plan")
            / "result.json"
        ).read_text()
    )
    assert result["status"] == status
    assert result["returncode"] == returncode
    assert result["signal"] == signal


def test_dispatch_interruption_leaves_request_and_interrupted_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controller interruption never becomes a successful dispatch receipt."""

    lineage = "lineage-a"
    root = roots.initialize_root(
        (tmp_path / "candidate").absolute(),
        lineage_id=lineage,
    )
    _stub_receipt_sources(monkeypatch)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(dispatch.subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        dispatch.dispatch_stage(
            "screen_plan",
            results_root=root,
            output_attempt=lineage,
            input_attempts={},
        )

    receipt_dir = _only_receipt(root, lineage, "screen_plan")
    assert (receipt_dir / "request.json").is_file()
    result = json.loads((receipt_dir / "result.json").read_text())
    assert result["status"] == "interrupted"
    assert result["returncode"] is None


def test_source_mismatch_stops_before_subprocess_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed pinned v3 closure cannot reach subprocess launch."""

    lineage = "lineage-a"
    root = roots.initialize_root(
        (tmp_path / "candidate").absolute(),
        lineage_id=lineage,
    )
    launched = False

    def should_not_launch(*args: object, **kwargs: object) -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr(
        dispatch,
        "verify_legacy_source_manifest",
        lambda repo_root: ("digest mismatch",),
    )
    monkeypatch.setattr(dispatch.subprocess, "run", should_not_launch)

    with pytest.raises(ValueError, match="digest mismatch"):
        dispatch.dispatch_stage(
            "screen_plan",
            results_root=root,
            output_attempt=lineage,
            input_attempts={},
        )
    assert launched is False
    assert not (root / "_v4" / "dispatch").exists()


def test_ownership_audit_preflights_source_before_creating_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad pinned source/config boundary cannot create audit artifacts."""

    destination = (tmp_path / "ownership").absolute()
    monkeypatch.setattr(
        dispatch,
        "verify_legacy_source_manifest",
        lambda _root: ("digest mismatch",),
    )

    with pytest.raises(ValueError, match="source preflight"):
        dispatch._run_ownership_audit(
            destination,
            lineage_id="lineage-a",
        )

    assert not destination.exists()


def test_ownership_train_audit_checks_worker_v4_identity(
    tmp_path: Path,
) -> None:
    """The disposable CPU worker, not only its grid, must identify as V4."""

    lineage = "lineage-a"
    run_id = "scan-run"
    root = roots.initialize_root(
        (tmp_path / "ownership").absolute(),
        lineage_id=lineage,
        purpose=roots.PURPOSE_OWNERSHIP_AUDIT,
    )
    grid = root / "00_grid" / lineage
    grid.mkdir(parents=True)
    manifest = grid / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "study": "pair_stability_v4",
                "attempt_id": lineage,
                "jobs": [{"run_id": run_id}],
            }
        )
    )
    result = root / "01_train" / run_id / lineage
    result.mkdir(parents=True)
    command = (
        "run.py --config config.yaml "
        f"run.root={root / '01_train'} "
        f"run.run_id={run_id}/{lineage} "
        "study.name=pair_stability_v4 runtime.device=cpu"
    )
    submitted = f"uv run {command}"
    records = {
        "source_grid_attempt.json": {
            "run_id": run_id,
            "grid_attempt_id": lineage,
            "grid_attempt_dir": str(grid),
            "manifest_path": str(manifest),
        },
        "submission.json": {
            "run_id": run_id,
            "grid_attempt_id": lineage,
            "launcher": "local",
            "launcher_job_id": "local-0",
            "command": command,
            "submitted_command": submitted,
        },
        "status.json": {
            "status": "completed",
            "current_event": "run_end",
            "exception_type": None,
            "exception_message": None,
        },
        "run_start.json": {
            "run_id": f"{run_id}/{lineage}",
            "run_dir": str(result),
            "study": {
                "name": "pair_stability_v4",
                "config_id": None,
            },
            "command": command,
        },
        "metadata.json": {
            "run_id": f"{run_id}/{lineage}",
            "run_dir": str(result),
            "command": command,
            "status": "completed",
            "runtime": {"device": "cpu"},
        },
    }
    for name, value in records.items():
        (result / name).write_text(json.dumps(value))
    (result / "command.txt").write_text(submitted + "\n")
    checkpoint = result / "checkpoints" / "step_000000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("complete\n")
    (checkpoint / "manifest.json").write_text("{}")
    (checkpoint.parent / "latest.json").write_text(
        json.dumps({"checkpoint_dir": checkpoint.name})
    )

    assert dispatch._audit_ownership_train_result(
        root,
        lineage_id=lineage,
    ) == ()

    run_start_path = result / "run_start.json"
    bad_start = records["run_start.json"]
    bad_start["study"] = {
        "name": "pair_stability_v3",
        "config_id": None,
    }
    run_start_path.write_text(json.dumps(bad_start))
    assert any(
        "study identity mismatch" in error
        for error in dispatch._audit_ownership_train_result(
            root,
            lineage_id=lineage,
        )
    )


def test_dispatch_cli_rejects_arbitrary_passthrough() -> None:
    """Argparse exposes typed V4 inputs, not trailing legacy arguments."""

    with pytest.raises(SystemExit) as exc_info:
        dispatch.main(
            [
                "run",
                "screen_plan",
                "--results-root",
                "/tmp/unused",
                "--output-attempt",
                "lineage",
                "--legacy-argv",
                "--smoke",
            ]
        )
    assert exc_info.value.code == 2


def test_contract_sidecar_dispatch_commands_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller can invoke finalization and receipt-aware verification."""

    calls: list[tuple[str, Path, str, bool | None]] = []

    def finalize(root: Path, *, lineage_id: str) -> Path:
        calls.append(("finalize", root, lineage_id, None))
        return root / "receipt.json"

    def verify(root: Path, *, lineage_id: str, require_receipt: bool) -> dict[str, str]:
        calls.append(("verify", root, lineage_id, require_receipt))
        return {"manifest_sha256": "a" * 64}

    monkeypatch.setattr(dispatch.contract_sidecars, "finalize_contract_sidecars", finalize)
    monkeypatch.setattr(dispatch.contract_sidecars, "verify_contract_sidecars", verify)
    root = (tmp_path / "candidate").absolute()

    assert dispatch.main(
        [
            "contract-sidecars-finalize",
            "--results-root",
            str(root),
            "--lineage-id",
            "lineage-a",
        ]
    ) == 0
    assert dispatch.main(
        [
            "contract-sidecars-verify",
            "--results-root",
            str(root),
            "--lineage-id",
            "lineage-a",
            "--no-receipt",
        ]
    ) == 0
    assert calls == [
        ("finalize", root, "lineage-a", None),
        ("verify", root, "lineage-a", False),
    ]


def test_smoke_controller_syntax_usage_order_waits_and_count_guards(
    tmp_path: Path,
) -> None:
    """The shell controller preserves the complete barriered smoke workflow."""

    script = STUDY_DIR / "submit_stack.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    marker = tmp_path / "sbatch-called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(f"#!/usr/bin/env bash\ntouch {marker}\n")
    fake_sbatch.chmod(0o755)
    environment = {"PATH": f"{fake_bin}:/usr/bin:/bin"}
    usage = subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert usage.returncode == 2
    assert "usage:" in usage.stdout
    assert not marker.exists()

    text = script.read_text()
    roles = re.findall(
        r"^(?:local_dispatch|fanout_dispatch) run ([a-z_]+) \\$",
        text,
        flags=re.MULTILINE,
    )
    assert roles == [
        "screen_plan",
        "screen_train",
        "screen_eval",
        "screen_collect",
        "select",
        "confirm_plan",
        "confirm_train",
        "confirm_eval",
        "confirm_collect",
        "report",
    ]
    assert len(roles) == len(set(roles)) == 10
    assert text.count("wait_stage screen_train ") == 1
    assert text.count("wait_stage screen_eval ") == 1
    assert text.count("wait_stage confirm_train ") == 1
    assert text.count("wait_stage confirm_eval ") == 1
    assert '"$EXPECTED" == "64"' in text
    assert '"$COLLECTED" == "64"' in text
    assert '"$FINAL_EXPECTED" == "8"' in text
    assert '"$FINAL_COLLECTED" == "8"' in text
    assert 'CONTROLLER_PARTITION="${STACK_CONTROLLER_PARTITION:-sapphire}"' in text
    assert 'CONTROLLER_TIME="${STACK_CONTROLLER_TIME:-3-00:00:00}"' in text
    assert "profile-check" in text
    assert text.index("profile-check") < text.index("OUTPUT=$(sbatch")
    assert text.index("controller-submission") > text.index("[[ -n \"${SLURM_JOB_ID:-}\" ]]")
    assert text.index("controller-submission") > text.index("[[ $# == 6 ]] ||")
    assert "--effective-cpus-per-task" in text
    assert "--effective-mem-per-cpu-mb" in text
    assert "--effective-time-limit" in text
    assert "scientific_partition=gpu_test" in text
    assert "chunk=32,cpus=4,mem_per_cpu_gb=8,partition=gpu_test" in text
    assert "chunk=8,cpus=4,mem_per_cpu_gb=8,partition=gpu_test" in text
    assert ".venv-submitit" in text
    assert "stack-inventory" in text
    assert "contract-sidecars-finalize" in text
    assert text.index("dispatch-manifest") < text.index("contract-sidecars-finalize")
    assert text.index("contract-sidecars-finalize") < text.index("controller-result")
    assert "contract_sidecars_finalize_failed" in text
    assert "--wait-job" not in text
    assert "pair_stability_v3/" not in text
    for flag in (
        "--grid-attempt",
        "--train-attempt",
        "--collection-attempt",
        "--selection-attempt",
        "--final-grid-attempt",
        "--final-train-attempt",
        "--final-eval-attempt",
        "--final-collect-attempt",
    ):
        assert flag in text


def _write_upstream_fixture(root: Path, lineage: str) -> None:
    grid_dir = root / "00_grid" / lineage
    grid_dir.mkdir(parents=True)
    train_config = grid_dir / "train_config.yaml"
    validation_config = grid_dir / "validation_config.yaml"
    train_config.write_text("study:\n  name: pair_stability_v4\n")
    validation_config.write_text("study:\n  name: pair_stability_v4\n")
    (grid_dir / "manifest.json").write_text(
        json.dumps(
            {
                "study": "pair_stability_v4",
                "attempt_id": lineage,
                "config_snapshots": {
                    "train": train_config.name,
                    "validation": validation_config.name,
                },
                "jobs": [{"run_id": "scan-run"}],
            }
        )
    )
    (root / "01_train" / "scan-run" / lineage).mkdir(parents=True)

    collection = root / "03_collect" / lineage
    collection.mkdir(parents=True)
    (collection / "collection_report.json").write_text("{}")
    selection = root / "04_select" / lineage
    selection.mkdir(parents=True)
    (selection / "selection_report.json").write_text("{}")

    final_grid = root / "05_final_grid" / lineage
    (final_grid / "jobs").mkdir(parents=True)
    (final_grid / "manifest.json").write_text(
        json.dumps(
            {
                "study": "pair_stability_v4",
                "attempt_id": lineage,
                "train_config": str(train_config),
                "eval_config": str(validation_config),
            }
        )
    )
    (final_grid / "jobs" / "final-run.json").write_text(
        json.dumps({"final_run_id": "final-run"})
    )
    (root / "06_final_train" / "final-run" / lineage).mkdir(parents=True)
    (root / "07_final_eval" / "final-run" / lineage).mkdir(parents=True)
    final_collect = root / "08_final_collect" / lineage
    final_collect.mkdir(parents=True)
    (final_collect / "manifest.yaml").write_text(
        "study: pair_stability_v4\nstage: 08_final_collect\n"
    )


def _stub_receipt_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dispatch,
        "verify_legacy_source_manifest",
        lambda repo_root: (),
    )
    monkeypatch.setattr(
        dispatch,
        "legacy_source_receipt",
        lambda repo_root: {"closure_sha256": "legacy"},
    )
    monkeypatch.setattr(
        dispatch,
        "runtime_source_receipt",
        lambda repo_root: {"closure_sha256": "runtime"},
    )
    monkeypatch.setattr(
        dispatch,
        "config_source_receipt",
        lambda repo_root: {"closure_sha256": "config"},
    )


def _only_receipt(root: Path, lineage: str, role: str) -> Path:
    receipts = list((root / "_v4" / "dispatch" / lineage / role).iterdir())
    assert len(receipts) == 1
    return receipts[0]
