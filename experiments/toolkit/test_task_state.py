"""Tests for row claim, deadline-guard, and completion-check primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.toolkit.task_state import (
    _attempt_already_completed,
    _checkpoint_ready,
    _checkpoint_step,
    _claim_row,
    _complete_checkpoint_dirs,
    _deadline_guard_reached,
    _final_train_completed,
    _latest_complete_checkpoint,
    _latest_restorable_checkpoint,
    _resolved_checkpoint,
    _resolved_latest_checkpoint,
    _resume_overrides,
    _terminal_row_status,
    claim_paths_for_statuses,
    local_claim_deadline_unix,
    parse_deadline_unix,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_parse_deadline_unix_accepts_seconds_and_iso() -> None:
    assert parse_deadline_unix(None) is None
    assert parse_deadline_unix("") is None
    assert parse_deadline_unix("1700000000") == 1700000000.0
    assert parse_deadline_unix("2024-01-01T00:00:00+00:00") == 1704067200.0


def test_local_claim_deadline_unix_prefers_explicit_over_env(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1700000000")
    args = argparse.Namespace(local_deadline="1800000000")
    assert local_claim_deadline_unix(args) == 1800000000.0
    args = argparse.Namespace(local_deadline=None)
    assert local_claim_deadline_unix(args) == 1700000000.0


def test_deadline_guard_reached_respects_guard_window() -> None:
    assert _deadline_guard_reached(None, 60) is False
    assert _deadline_guard_reached(1e15, 0) is False
    assert _deadline_guard_reached(0.0, 60) is True


def test_claim_paths_for_statuses_maps_status_paths_to_claim_files(tmp_path: Path) -> None:
    status_path = tmp_path / "run" / "status.json"
    assert claim_paths_for_statuses(None) is None
    claim_paths = claim_paths_for_statuses([status_path, None])
    assert claim_paths == [status_path.with_name("launcher_claim.json"), None]


def test_attempt_already_completed_missing_artifact(tmp_path: Path) -> None:
    status_path = tmp_path / "attempt" / "status.json"
    assert _attempt_already_completed(None) is False
    assert _attempt_already_completed(status_path) is False


def test_attempt_already_completed_requires_checkpoint_and_completed_status(tmp_path: Path) -> None:
    status_path = tmp_path / "attempt" / "status.json"
    checkpoint_path = tmp_path / "attempt" / "checkpoints" / "latest.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("{}")
    _write(status_path, {"status": "running"})
    assert _attempt_already_completed(status_path) is False

    _write(status_path, {"status": "completed"})
    assert _attempt_already_completed(status_path) is True


def test_attempt_already_completed_false_without_checkpoint(tmp_path: Path) -> None:
    status_path = tmp_path / "attempt" / "status.json"
    _write(status_path, {"status": "completed"})
    assert _attempt_already_completed(status_path) is False


def test_terminal_row_status_reads_launcher_then_run_status(tmp_path: Path) -> None:
    status_path = tmp_path / "attempt" / "launcher_status.json"
    run_status_path = tmp_path / "attempt" / "status.json"
    assert _terminal_row_status(None) is None
    assert _terminal_row_status(status_path) is None

    _write(run_status_path, {"status": "failed"})
    assert _terminal_row_status(status_path) == "failed"

    _write(status_path, {"status": "stopped"})
    assert _terminal_row_status(status_path) == "stopped"

    _write(status_path, {"status": "running"})
    _write(run_status_path, {"status": "running"})
    assert _terminal_row_status(status_path) is None


def test_claim_row_first_claim_succeeds_and_writes_payload(tmp_path: Path) -> None:
    claim_path = tmp_path / "launcher_claim.json"
    assert _claim_row(claim_path, {"status": "claimed"}) is True
    assert json.loads(claim_path.read_text())["status"] == "claimed"


def test_claim_row_second_claim_blocked_without_terminal_status(tmp_path: Path) -> None:
    claim_path = tmp_path / "launcher_claim.json"
    status_path = tmp_path / "status.json"
    _write(status_path, {"status": "running"})
    assert _claim_row(claim_path, {"status": "claimed"}, status_path) is True
    assert _claim_row(claim_path, {"status": "claimed"}, status_path) is False


def test_claim_row_reclaims_after_failed_row(tmp_path: Path) -> None:
    claim_path = tmp_path / "launcher_claim.json"
    status_path = tmp_path / "status.json"
    _write(status_path, {"status": "running"})
    assert _claim_row(claim_path, {"status": "claimed"}, status_path) is True

    _write(status_path, {"status": "failed"})
    assert _claim_row(claim_path, {"status": "claimed"}, status_path) is True
    reclaimed = json.loads(claim_path.read_text())
    assert reclaimed["reclaimed"] is True
    assert reclaimed["reclaim_reason"] == "failed"


def _touch_complete_checkpoint(attempt_dir: Path, step: int, *, tmp_suffix: bool = False) -> Path:
    name = f"step_{step:06d}" + (".tmp" if tmp_suffix else "")
    checkpoint_dir = attempt_dir / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "COMPLETE").write_text("")
    return checkpoint_dir


def test_checkpoint_ready_reflects_latest_pointer(tmp_path: Path) -> None:
    assert _checkpoint_ready(tmp_path) is False
    _write(tmp_path / "checkpoints" / "latest.json", {"step": 0})
    assert _checkpoint_ready(tmp_path) is True


def test_checkpoint_step_parses_step_name_and_falls_back() -> None:
    assert _checkpoint_step(Path("step_000042")) == (42, "step_000042")
    assert _checkpoint_step(Path("not-a-step")) == (-1, "not-a-step")


def test_complete_checkpoint_dirs_excludes_incomplete_and_tmp(tmp_path: Path) -> None:
    assert _complete_checkpoint_dirs(tmp_path) == []
    (tmp_path / "checkpoints" / "step_000000").mkdir(parents=True)
    _touch_complete_checkpoint(tmp_path, 1)
    _touch_complete_checkpoint(tmp_path, 2, tmp_suffix=True)
    dirs = _complete_checkpoint_dirs(tmp_path)
    assert [d.name for d in dirs] == ["step_000001"]


def test_latest_complete_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    assert _latest_complete_checkpoint(tmp_path) is None
    _touch_complete_checkpoint(tmp_path, 1)
    _touch_complete_checkpoint(tmp_path, 3)
    _touch_complete_checkpoint(tmp_path, 2)
    assert _latest_complete_checkpoint(tmp_path).name == "step_000003"


def test_latest_restorable_checkpoint_requires_manifest(tmp_path: Path) -> None:
    _touch_complete_checkpoint(tmp_path, 3)
    latest = _touch_complete_checkpoint(tmp_path, 2)
    (latest / "manifest.json").write_text("{}")

    assert _latest_restorable_checkpoint(tmp_path) == latest


def test_final_train_completed_requires_status_and_checkpoint(tmp_path: Path) -> None:
    assert _final_train_completed(tmp_path) is False

    _write(tmp_path / "status.json", {"status": "completed"})
    assert _final_train_completed(tmp_path) is False

    _touch_complete_checkpoint(tmp_path, 1)
    assert _final_train_completed(tmp_path) is True

    _write(tmp_path / "status.json", {"status": "running"})
    assert _final_train_completed(tmp_path) is False


def test_final_train_completed_false_on_invalid_json(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("not json")
    _touch_complete_checkpoint(tmp_path, 1)
    assert _final_train_completed(tmp_path) is False


def test_resume_overrides_empty_without_checkpoint(tmp_path: Path) -> None:
    assert _resume_overrides(tmp_path) == []


def test_resume_overrides_empty_once_fully_completed(tmp_path: Path) -> None:
    _touch_complete_checkpoint(tmp_path, 1)
    _write(tmp_path / "status.json", {"status": "completed"})
    assert _resume_overrides(tmp_path) == []


def test_resume_overrides_points_at_highest_checkpoint_when_partial(tmp_path: Path) -> None:
    checkpoint_dir = _touch_complete_checkpoint(tmp_path, 1)
    _write(tmp_path / "status.json", {"status": "stopped"})
    overrides = _resume_overrides(tmp_path)
    assert overrides == [f"load.path={checkpoint_dir}", "load.mode=train_resume"]


def test_resolved_checkpoint_missing_selection_file(tmp_path: Path) -> None:
    assert _resolved_checkpoint(tmp_path) is None


def test_resolved_checkpoint_missing_pointer_file(tmp_path: Path) -> None:
    _write(
        tmp_path / "selected_checkpoint.json",
        {"selection_policy": "latest_checkpoint_pointer", "checkpoint_pointer": str(tmp_path / "checkpoints" / "latest.json")},
    )
    assert _resolved_checkpoint(tmp_path) is None


def test_resolved_checkpoint_requires_complete_and_manifest(tmp_path: Path) -> None:
    pointer_path = tmp_path / "checkpoints" / "latest.json"
    _write(
        tmp_path / "selected_checkpoint.json",
        {"selection_policy": "latest_checkpoint_pointer", "checkpoint_pointer": str(pointer_path)},
    )
    _write(pointer_path, {"checkpoint_dir": "step_000001"})
    checkpoint_dir = tmp_path / "checkpoints" / "step_000001"
    checkpoint_dir.mkdir(parents=True)
    assert _resolved_checkpoint(tmp_path) is None

    (checkpoint_dir / "COMPLETE").write_text("")
    assert _resolved_checkpoint(tmp_path) is None

    (checkpoint_dir / "manifest.json").write_text("{}")
    resolved = _resolved_checkpoint(tmp_path)
    assert resolved is not None
    assert resolved["resolved_checkpoint_dir"] == str(checkpoint_dir)


def test_resolved_latest_checkpoint_ignores_stale_pointer(tmp_path: Path) -> None:
    stale = _touch_complete_checkpoint(tmp_path, 1)
    latest = _touch_complete_checkpoint(tmp_path, 3)
    (stale / "manifest.json").write_text("{}")
    (latest / "manifest.json").write_text("{}")
    pointer = tmp_path / "checkpoints" / "latest.json"
    _write(pointer, {"checkpoint_dir": stale.name})

    resolved = _resolved_latest_checkpoint(tmp_path)

    assert resolved is not None
    assert resolved["selection_policy"] == "latest_complete_checkpoint"
    assert resolved["checkpoint_pointer_data"] == {"checkpoint_dir": stale.name}
    assert resolved["resolved_checkpoint_dir"] == str(latest)
