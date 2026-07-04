"""Tests for row claim, deadline-guard, and completion-check primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.toolkit.task_state import (
    _attempt_already_completed,
    _claim_row,
    _deadline_guard_reached,
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
