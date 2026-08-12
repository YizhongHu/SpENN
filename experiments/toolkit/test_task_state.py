"""Tests for row claim, deadline-guard, and completion-check primitives."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Sequence

from experiments.toolkit.task_state import (
    _attempt_already_completed,
    _checkpoint_ready,
    _checkpoint_step,
    _claim_component,
    _claim_row,
    _complete_checkpoint_dirs,
    _deadline_guard_reached,
    _final_train_completed,
    _latest_complete_checkpoint,
    _resolved_checkpoint,
    _resume_overrides,
    _terminal_row_status,
    allocation_deadline_unix,
    claim_paths_for_statuses,
    claim_row_for_pass,
    local_claim_deadline_unix,
    next_attempt_dir,
    parse_deadline_unix,
    pass_claim_path,
    pass_claims_dir,
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


# ---------------------------------------------------------------------------
# Pass-scoped row claims (ADR-C012)
# ---------------------------------------------------------------------------


def test_claim_component_is_injective_over_path_separators() -> None:
    assert _claim_component("train/run-a/attempt-0001") == "train%2Frun-a%2Fattempt-0001"
    # Two row ids that a naive separator-stripping scheme would fuse must not
    # share a claim directory: a collision silently drops a row from the pass.
    assert _claim_component("a/b") != _claim_component("a_b")
    assert _claim_component("a%2Fb") != _claim_component("a/b")
    for unusable in ("", ".", ".."):
        try:
            _claim_component(unusable)
        except ValueError:
            continue
        raise AssertionError(f"{unusable!r} must not be accepted as a claim key")


def test_pass_claims_dir_is_namespaced_by_pass(tmp_path: Path) -> None:
    assert pass_claims_dir(tmp_path, "pass-a") == tmp_path / "_claims" / "pass-a"
    assert pass_claim_path(tmp_path, "pass-a", "row/a") == (
        tmp_path / "_claims" / "pass-a" / "row%2Fa"
    )
    assert not (tmp_path / "_claims").exists()


def test_claim_row_for_pass_is_exclusive_and_writes_a_receipt(tmp_path: Path) -> None:
    assert claim_row_for_pass(tmp_path, "pass-a", "row-00", {"worker": "w0"}) is True
    assert claim_row_for_pass(tmp_path, "pass-a", "row-00", {"worker": "w1"}) is False

    receipt = json.loads((pass_claims_dir(tmp_path, "pass-a") / "row-00" / "claim.json").read_text())
    assert receipt["row"] == "row-00"
    assert receipt["pass_id"] == "pass-a"
    assert receipt["worker"] == "w0"


def test_claim_row_for_pass_never_releases_a_failed_row(tmp_path: Path) -> None:
    """ADR-C012: the opposite of ``_claim_row``'s release-by-reclaim policy."""

    row_dir = tmp_path / "rows" / "row-00"
    assert claim_row_for_pass(tmp_path, "pass-a", "row-00") is True
    _write(next_attempt_dir(row_dir) / "status.json", {"status": "failed"})

    # ``_claim_row`` hands the row back at exactly this point; this must not.
    assert claim_row_for_pass(tmp_path, "pass-a", "row-00") is False
    _write(row_dir / "attempt1" / "status.json", {"status": "stopped"})
    assert claim_row_for_pass(tmp_path, "pass-a", "row-00") is False

    # Retry is expressed by a new pass, which starts from an empty namespace.
    assert claim_row_for_pass(tmp_path, "pass-b", "row-00") is True


def test_next_attempt_dir_numbers_from_one_and_never_reuses(tmp_path: Path) -> None:
    row_dir = tmp_path / "rows" / "row-00"
    first = next_attempt_dir(row_dir)
    assert first == row_dir / "attempt1"
    assert first.is_dir()

    (first / "evidence").write_text("keep me")
    second = next_attempt_dir(row_dir)
    assert second == row_dir / "attempt2"
    assert (first / "evidence").read_text() == "keep me"


def test_allocation_deadline_unix_prefers_explicit_then_env_var_then_slurm() -> None:
    env = {"PBS_ALLOCATION_END": "1800000000", "SLURM_JOB_END_TIME": "1700000000"}
    assert allocation_deadline_unix(None, environ={}) is None
    assert allocation_deadline_unix(None, environ=env) == 1700000000.0
    assert allocation_deadline_unix(None, env_var="PBS_ALLOCATION_END", environ=env) == 1800000000.0
    assert allocation_deadline_unix("1900000000", env_var="PBS_ALLOCATION_END", environ=env) == 1900000000.0
    assert allocation_deadline_unix("", environ=env) == 1700000000.0
    assert allocation_deadline_unix("2024-01-01T00:00:00+00:00", environ={}) == 1704067200.0


def test_allocation_deadline_unix_reads_process_environment_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    assert allocation_deadline_unix() is None
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1700000000")
    assert allocation_deadline_unix() == 1700000000.0


# --- the 2026-08-07 thundering-retry regression ----------------------------
#
# Four workers each re-claimed and re-ran one deterministically broken row in a
# single pass, because the claim was released on failure. The property asserted
# below is that a failing row executes EXACTLY ONCE per pass no matter how many
# workers contend for it, and that a fresh pass re-runs only that row. Real
# processes against a real directory are required: what is under test is the
# atomicity of ``mkdir`` under contention, which a mock cannot exercise.

_MP_CONTEXT = multiprocessing.get_context(
    "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
)


def _row_dir(run_root: Path, row: str) -> Path:
    return Path(run_root) / "rows" / row


def _row_completed(row_dir: Path) -> bool:
    """Return whether any attempt of this row completed, per the tracked check."""

    return any(
        _attempt_already_completed(attempt / "status.json")
        for attempt in sorted(row_dir.glob("attempt*"))
    )


def _attempt_count(row_dir: Path) -> int:
    return len([path for path in row_dir.glob("attempt*") if path.is_dir()])


def _simulate_row_execution(attempt_dir: Path, *, fails: bool) -> None:
    """Write the artifacts a real row execution would leave behind."""

    if fails:
        # A permanently broken row: no checkpoint pointer, terminal status.
        _write(attempt_dir / "status.json", {"status": "failed"})
        return
    # Checkpoint first, then status: a reader must never see "completed"
    # without the pointer ``_attempt_already_completed`` also requires.
    _write(attempt_dir / "checkpoints" / "latest.json", {"step": 1})
    _write(attempt_dir / "status.json", {"status": "completed"})


def _wait_for_peers(gate_dir: Path, worker_id: str, n_workers: int, timeout: float = 60.0) -> None:
    """Block until every worker has arrived, so the claims really do collide."""

    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / worker_id).write_text("")
    deadline = time.monotonic() + timeout
    while len(list(gate_dir.iterdir())) < n_workers and time.monotonic() < deadline:
        time.sleep(0.001)


def _claim_pass_worker(
    run_root: str,
    pass_id: str,
    rows: Sequence[str],
    failing_row: str,
    worker_id: str,
    n_workers: int,
    max_sweeps: int = 3,
) -> None:
    """One allocation-pool worker: sweep the plan, claim, run, refill."""

    root = Path(run_root)
    _wait_for_peers(root / "_ready" / pass_id, worker_id, n_workers)
    lost_claims = 0
    executed: list[str] = []
    for _ in range(max_sweeps):
        claimed = 0
        for row in rows:
            row_dir = _row_dir(root, row)
            if _row_completed(row_dir):
                continue
            if not claim_row_for_pass(root, pass_id, row, {"worker": worker_id}):
                lost_claims += 1
                continue
            claimed += 1
            attempt_dir = next_attempt_dir(row_dir)
            executed.append(f"{row}:{attempt_dir.name}")
            _simulate_row_execution(attempt_dir, fails=row == failing_row)
        if claimed == 0:
            # Nothing left this worker may take; a correct policy converges here.
            break
    _write(
        root / "_workers" / f"{pass_id}.{worker_id}.json",
        {"worker": worker_id, "lost_claims": lost_claims, "executed": executed},
    )


def _run_pass(
    run_root: Path,
    pass_id: str,
    rows: Sequence[str],
    failing_row: str,
    n_workers: int,
) -> list[dict]:
    processes = [
        _MP_CONTEXT.Process(
            target=_claim_pass_worker,
            args=(str(run_root), pass_id, list(rows), failing_row, f"w{index}", n_workers),
        )
        for index in range(n_workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    for process in processes:
        assert process.exitcode == 0, f"worker exited with {process.exitcode}"
    return [
        json.loads(path.read_text())
        for path in sorted((run_root / "_workers").glob(f"{pass_id}.*.json"))
    ]


def test_pass_scoped_claims_run_a_failing_row_once_per_pass(tmp_path: Path) -> None:
    rows = [f"row-{index:02d}" for index in range(8)]
    failing_row = "row-03"
    n_workers = 6

    receipts = _run_pass(tmp_path, "pass-a", rows, failing_row, n_workers)

    executed = [entry for receipt in receipts for entry in receipt["executed"]]
    assert len(executed) == len(rows), f"pass-a executed {executed}"
    assert {entry.split(":")[0] for entry in executed} == set(rows)
    for row in rows:
        assert _attempt_count(_row_dir(tmp_path, row)) == 1, f"{row} ran more than once"
    assert sum(receipt["lost_claims"] for receipt in receipts) > 0, "workers never actually contended"

    failing_status = json.loads((_row_dir(tmp_path, failing_row) / "attempt1" / "status.json").read_text())
    assert failing_status["status"] == "failed"
    assert not _row_completed(_row_dir(tmp_path, failing_row))

    # A fresh pass re-claims only what did not complete.
    retry_receipts = _run_pass(tmp_path, "pass-b", rows, failing_row, n_workers)

    retried = [entry for receipt in retry_receipts for entry in receipt["executed"]]
    assert retried == [f"{failing_row}:attempt2"], f"pass-b executed {retried}"
    assert _attempt_count(_row_dir(tmp_path, failing_row)) == 2
    for row in rows:
        if row == failing_row:
            continue
        assert _attempt_count(_row_dir(tmp_path, row)) == 1, f"{row} re-ran in pass-b"


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
