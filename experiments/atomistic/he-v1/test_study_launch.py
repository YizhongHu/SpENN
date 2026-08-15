"""Launch-stage behavior: the pin is mandatory and the script is a file.

The failures guarded here have all cost this program time:

unpinned GPU rows
    `seas_gpu` mixes H200 and A100-80GB. An unconstrained submission already
    destroyed timing comparability between two of this program's own baseline
    runs, so every emitted script must carry ``--constraint`` and it must be
    the stratum's own feature -- re-derived here, not copied from a manifest
    that could have been hand-edited.

bare ``uv``
    ``uv`` is on PATH in no shell on Cannon. A job that calls it dies
    ``ExitCode 127:0`` at ``Elapsed 00:00:00``, which reads like an outage and
    is a script defect.

silent resume
    ``--no-requeue`` is not optional: a requeued row would restart from scratch
    and quietly consume the allocation it was sized against.

a dry run is not a submission
    Without ``--submit`` the row records ``submitted: false`` and a null job
    id. An empty job id string would later read as "submitted, id unknown".
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

STUDY_DIR = Path(__file__).resolve().parent


def _load_study_module(name: str) -> ModuleType:
    """Load one study module by path (the study directory is not a package)."""

    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan = _load_study_module("plan")
launch = _load_study_module("launch")
strata = _load_study_module("strata")

UV_BIN = "/opt/uv/bin/uv"

GRID: dict[str, Any] = {
    "study": "tpen_he_v1",
    "train_config": "experiments/atomistic/he-v1/configs/train.yaml",
    "eval_config": "experiments/atomistic/he-v1/configs/eval.yaml",
    "seeds": [0, 1],
    "checkpoint_steps": [10, 20],
    "eval_chains": 2,
    "eval_chain_seed_base": 900000,
    "train_resources": {
        "partition": "kozinsky_gpu",
        "stratum": "a100",
        "timeout_min": 600,
        "cpus": 16,
        "mem_gb": 128,
        "gpus": 1,
    },
    "eval_resources": {
        "partition": "seas_gpu",
        "stratum": "h200",
        "timeout_min": 120,
        "cpus": 8,
        "mem_gb": 64,
        "gpus": 1,
    },
    "gate_spec": {},
}


@pytest.fixture()
def planned(tmp_path: Path) -> dict[str, Any]:
    """Write one plan attempt and return its manifest."""

    config = plan.validate_grid_config(copy.deepcopy(GRID))
    manifest = plan.build_manifest(
        config=config,
        rows=plan.expand_rows(config),
        attempt_id="20260815T120000",
        results_root=str(tmp_path),
        grid_config_path=None,
        grid_config_sha256=None,
        created_at="2026-08-15T12:00:00-04:00",
    )
    plan.write_plan(manifest, results_root=tmp_path)
    return manifest


def _launch(
    planned: dict[str, Any],
    tmp_path: Path,
    *,
    rows: Any = None,
    submit: bool = False,
) -> dict[str, Any]:
    return launch.launch(
        manifest=planned,
        results_root=tmp_path,
        repo_root=tmp_path / "checkout",
        rows=planned["rows"] if rows is None else rows,
        launch_attempt_id="20260815T130000",
        uv_bin=UV_BIN,
        uv_extras=["cu128"],
        uv_project_environment="/work/env",
        uv_cache_root="/work/uv-cache",
        account="kozinsky_lab",
        submit=submit,
    )


def test_every_gpu_row_is_pinned_to_its_stratum(planned: dict[str, Any], tmp_path: Path) -> None:
    """No row reaches sbatch without a stratum constraint."""

    summary = _launch(planned, tmp_path)
    for record in summary["rows"]:
        script = Path(record["script_path"]).read_text(encoding="utf-8")
        expected = strata.constraint_for(record["requested_stratum"])
        assert f"#SBATCH --constraint={expected}" in script
        assert record["requested_constraint"] == expected


def test_manifest_constraint_cannot_contradict_the_stratum(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    """A hand-edited manifest cannot smuggle an A100 script under an H200 row."""

    row = copy.deepcopy(planned["rows"][1])
    row["resources"]["constraint"] = "a100"
    with pytest.raises(launch.LaunchError, match="pins"):
        launch.sbatch_directives(row, job_name="x", log_dir=tmp_path, account=None, dependency=None)


def test_scripts_carry_no_requeue(planned: dict[str, Any], tmp_path: Path) -> None:
    """Rows are sized to finish; a requeue would silently restart one."""

    summary = _launch(planned, tmp_path)
    for record in summary["rows"]:
        assert "#SBATCH --no-requeue" in Path(record["script_path"]).read_text(encoding="utf-8")
        assert record["requeue"] is False
        assert record["resume"] is False


def test_bare_uv_is_refused(planned: dict[str, Any], tmp_path: Path) -> None:
    """The 127 trap is a script defect, so the script is never allowed to have it."""

    with pytest.raises(launch.LaunchError, match="not an absolute path"):
        launch.launch(
            manifest=planned,
            results_root=tmp_path,
            repo_root=tmp_path,
            rows=planned["rows"][:1],
            launch_attempt_id="a",
            uv_bin="uv",
            uv_extras=["cu128"],
            uv_project_environment="/work/env",
            uv_cache_root="/work/uv-cache",
            account=None,
            submit=False,
        )


def test_uv_is_invoked_by_absolute_path(planned: dict[str, Any], tmp_path: Path) -> None:
    """The emitted command line calls the binary, never a PATH lookup."""

    summary = _launch(planned, tmp_path)
    script = Path(summary["rows"][0]["script_path"]).read_text(encoding="utf-8")
    assert UV_BIN in script
    assert "\nuv run" not in script


def test_torch_build_extra_is_explicit(planned: dict[str, Any], tmp_path: Path) -> None:
    """No default extra: the torch build a row ran under is recorded, not assumed."""

    with pytest.raises(launch.LaunchError, match="uv extra is required"):
        launch.launch(
            manifest=planned,
            results_root=tmp_path,
            repo_root=tmp_path,
            rows=planned["rows"][:1],
            launch_attempt_id="a",
            uv_bin=UV_BIN,
            uv_extras=[],
            uv_project_environment="/work/env",
            uv_cache_root="/work/uv-cache",
            account=None,
            submit=False,
        )


def test_uv_cache_is_job_local(planned: dict[str, Any], tmp_path: Path) -> None:
    """A shared NFS uv cache caused a .nfs rename race; the cache is per job."""

    summary = _launch(planned, tmp_path)
    script = Path(summary["rows"][0]["script_path"]).read_text(encoding="utf-8")
    assert 'export UV_CACHE_DIR=' in script
    assert '"/${SLURM_JOB_ID}"' in script


def test_script_is_written_to_disk_and_submitted_by_path(
    planned: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sbatch runs the file that was written; nothing is re-expanded by a shell."""

    seen: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = "424242\n"
        stderr = ""

    def _fake_run(command: list[str], **kwargs: Any) -> Any:
        seen.append(list(command))
        return _Completed()

    monkeypatch.setattr(launch.subprocess, "run", _fake_run)
    summary = _launch(planned, tmp_path, rows=planned["rows"][:1], submit=True)
    record = summary["rows"][0]
    assert seen == [["sbatch", "--parsable", record["script_path"]]]
    assert Path(record["script_path"]).is_file()
    assert record["job_id"] == "424242"
    assert record["submitted"] is True


def test_dry_run_records_absence_not_an_empty_job_id(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    """``submitted: false`` with a null id, never an empty string that reads as an id."""

    summary = _launch(planned, tmp_path)
    assert summary["submitted"] is False
    assert summary["n_submitted"] == 0
    for record in summary["rows"]:
        assert record["job_id"] is None
        assert record["submitted"] is False


def test_evaluation_rows_chain_on_their_training_row(
    planned: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An eval row submitted with its training row waits for it with afterok."""

    job_ids = iter(["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"])

    class _Completed:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        launch.subprocess, "run", lambda command, **kwargs: _Completed(next(job_ids))
    )
    summary = _launch(planned, tmp_path, submit=True)
    train_record = next(record for record in summary["rows"] if record["kind"] == "train")
    eval_record = next(
        record
        for record in summary["rows"]
        if record["kind"] == "eval" and record["row_id"].startswith("eval-seed0000")
    )
    assert eval_record["dependency"] == f"afterok:{train_record['job_id']}"


def test_evaluation_row_without_its_training_row_needs_the_checkpoint(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    """Launching an eval row alone fails unless the checkpoint it restores exists."""

    eval_rows = [row for row in planned["rows"] if row["kind"] == "eval"][:1]
    with pytest.raises(launch.LaunchError, match="does not exist"):
        _launch(planned, tmp_path, rows=eval_rows)

    checkpoint_dir = launch.checkpoint_dir_for_eval_row(
        tmp_path, eval_rows[0], planned["attempt_id"], manifest=planned
    )
    checkpoint_dir.mkdir(parents=True)
    summary = _launch(planned, tmp_path, rows=eval_rows)
    assert summary["rows"][0]["dependency"] is None


def test_submission_records_are_written_per_row(planned: dict[str, Any], tmp_path: Path) -> None:
    """Every row leaves a durable record of what was asked for."""

    summary = _launch(planned, tmp_path)
    for record in summary["rows"]:
        payload = json.loads(
            (Path(record["script_path"]).parent / "submission.json").read_text(encoding="utf-8")
        )
        assert payload["requested_constraint"] == record["requested_constraint"]
        assert payload["plan_attempt_id"] == planned["attempt_id"]
        assert payload["run_dir"].endswith(record["row_id"])


def test_row_selection_rejects_unknown_row_ids(planned: dict[str, Any]) -> None:
    """A typo'd row id selects nothing rather than silently launching everything."""

    with pytest.raises(launch.LaunchError, match="not in this plan"):
        launch.select_rows(planned, kinds=[], row_ids=["train-seed9999"])


def test_row_selection_preserves_manifest_order(planned: dict[str, Any]) -> None:
    """Selection filters; it never reorders."""

    rows = launch.select_rows(planned, kinds=["eval"], row_ids=[])
    assert [row["index"] for row in rows] == sorted(row["index"] for row in rows)
    assert all(row["kind"] == "eval" for row in rows)


def test_driver_command_names_the_checkpoint_for_eval_rows(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    """The chain is told which model to restore; it never discovers one."""

    eval_row = next(row for row in planned["rows"] if row["kind"] == "eval")
    command = launch.driver_command(
        eval_row,
        results_root=tmp_path,
        manifest=planned,
        attempt_id=planned["attempt_id"],
        launch_attempt_id="20260815T130000",
    )
    assert "--checkpoint-dir" in command
    assert command[command.index("--checkpoint-dir") + 1].endswith(eval_row["checkpoint_dir_name"])


def test_zero_gpu_row_is_a_planning_error(planned: dict[str, Any], tmp_path: Path) -> None:
    """This study has no CPU-only rows; a zero-GPU row would run unpinned."""

    row = copy.deepcopy(planned["rows"][0])
    row["resources"]["gpus"] = 0
    with pytest.raises(launch.LaunchError, match="zero-GPU row"):
        launch.sbatch_directives(row, job_name="x", log_dir=tmp_path, account=None, dependency=None)
