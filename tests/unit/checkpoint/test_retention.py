"""Torch-free tests for pure checkpoint retention policies."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tpen.checkpoint import HoldUntilSelection, KeepLast, PinRecord, RetainAll
from tpen.checkpoint.reference import CheckpointRef


def _manifest(step: int, marker: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": {"model": "model.pt"},
        "hashes": {"marker": marker},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": marker, "git_sha": "deadbeef"},
    }


def _ref(root: Path, step: int, marker: str | None = None) -> CheckpointRef:
    marker = marker or str(root)
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(marker.encode("utf-8"))
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(_manifest(step, marker), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return CheckpointRef.from_directory(checkpoint_dir)


def _actions(snapshot) -> dict[int, str]:
    return {decision.ref.next_iteration: decision.action for decision in snapshot.decisions}


def test_retain_all_is_sorted_serializable_and_has_no_delete_targets(tmp_path: Path) -> None:
    refs = [_ref(tmp_path / "stream", step, f"model-{step}") for step in (1, 2, 3)]

    snapshot = RetainAll().decide(refs, pin_state=())

    assert [decision.ref.next_iteration for decision in snapshot.decisions] == [1, 2, 3]
    assert snapshot.deletion_targets == ()
    assert json.loads(snapshot.to_json()) == snapshot.to_dict()
    assert snapshot.to_json().encode("utf-8")


def test_keep_last_snapshot_is_byte_identical_after_seeded_shuffle(tmp_path: Path) -> None:
    refs = [_ref(tmp_path / "stream", step, f"model-{step}") for step in (1, 2, 3, 4)]
    shuffled = list(refs)
    random.Random(8671).shuffle(shuffled)

    first = KeepLast(2).decide(refs, pin_state=())
    second = KeepLast(2).decide(shuffled, pin_state=())

    assert first.to_json() == second.to_json()
    assert _actions(first) == {1: "delete", 2: "delete", 3: "retain", 4: "retain"}


def test_keep_last_is_scoped_per_stream_root(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    refs = [
        _ref(first_root, 1, "first-1"),
        _ref(first_root, 2, "first-2"),
        _ref(second_root, 1, "second-1"),
        _ref(second_root, 2, "second-2"),
    ]

    snapshot = KeepLast(1).decide(refs, pin_state=())

    assert [(decision.ref.checkpoint_dir.parent.name, decision.ref.next_iteration, decision.action)
            for decision in snapshot.decisions] == [
        ("first", 1, "delete"),
        ("second", 1, "delete"),
        ("first", 2, "retain"),
        ("second", 2, "retain"),
    ]


def test_keep_last_protects_each_latest_and_any_live_pin(tmp_path: Path) -> None:
    refs = [_ref(tmp_path / "stream", step, f"model-{step}") for step in (1, 2, 3)]
    pin = PinRecord(token="eval-token", ref=refs[0], owner="worker", reason="evaluation")

    snapshot = KeepLast(1).decide(refs, (pin,), latest=refs[2])

    assert _actions(snapshot) == {1: "retain", 2: "delete", 3: "retain"}
    assert snapshot.decisions[0].protected
    assert snapshot.decisions[-1].protected


def test_missing_pin_state_retains_everything_and_does_not_change_inputs(tmp_path: Path) -> None:
    refs = [_ref(tmp_path / "stream", step, f"model-{step}") for step in (1, 2, 3)]
    before = tuple(refs)

    snapshot = KeepLast(1).decide(refs)

    assert snapshot.status == "missing_pin_state"
    assert snapshot.deletion_targets == ()
    assert tuple(refs) == before


def test_ambiguous_pin_state_retains_everything(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "stream", 1, "model-1")
    pins = (
        PinRecord(token="same-token", ref=ref, owner="one", reason="hold"),
        PinRecord(token="same-token", ref=ref, owner="two", reason="hold"),
    )

    snapshot = KeepLast(1).decide((ref,), pins, latest=ref)

    assert snapshot.status == "ambiguous_pin_state"
    assert snapshot.deletion_targets == ()
    assert all(decision.action == "retain" for decision in snapshot.decisions)


def test_relocated_same_content_pin_refs_are_ambiguous(tmp_path: Path) -> None:
    first = _ref(tmp_path / "first", 1, "same-content")
    relocated = _ref(tmp_path / "second", 1, "same-content")
    pins = (
        PinRecord(token="first-token", ref=first, owner="one", reason="hold"),
        PinRecord(token="second-token", ref=relocated, owner="two", reason="hold"),
    )

    snapshot = KeepLast(1).decide((first, relocated), pins, latest=(first, relocated))

    assert snapshot.status == "ambiguous_pin_state"
    assert snapshot.deletion_targets == ()


@pytest.mark.parametrize("limit", [True, False])
def test_keep_last_rejects_bool_limits(limit: bool) -> None:
    with pytest.raises(TypeError, match="limit"):
        KeepLast(limit)


def test_keep_last_rejects_nonpositive_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        KeepLast(0)


def test_legacy_keep_last_migration_is_explicit() -> None:
    policy = KeepLast.from_legacy_keep_last(2)

    assert policy.limit == 2


def test_unbounded_keep_last_is_explicit_retain_all(tmp_path: Path) -> None:
    refs = [_ref(tmp_path / "stream", step, f"model-{step}") for step in (1, 2)]

    snapshot = KeepLast(None).decide(refs, pin_state=())

    assert snapshot.config == (("limit", None),)
    assert snapshot.deletion_targets == ()


def test_hold_until_selection_retains_until_selection_then_plans_exact_targets(
    tmp_path: Path,
) -> None:
    refs = [_ref(tmp_path / "stream", step, f"model-{step}") for step in (1, 2, 3)]
    pin = PinRecord(token="hold-token", ref=refs[0], owner="worker", reason="hold")

    pending = HoldUntilSelection().decide(refs, (pin,), latest=refs[2])
    selected = HoldUntilSelection().decide(
        refs,
        (pin,),
        latest=refs[2],
        selected=refs[1],
    )

    assert pending.status == "selection_pending"
    assert pending.deletion_targets == ()
    assert _actions(selected) == {1: "retain", 2: "retain", 3: "retain"}


def test_hold_until_selection_deletes_unselected_unprotected_refs(tmp_path: Path) -> None:
    refs = [_ref(tmp_path / "stream", step, f"model-{step}") for step in (1, 2, 3)]

    snapshot = HoldUntilSelection().decide(
        refs,
        pin_state=(),
        latest=refs[2],
        selected=refs[0],
    )

    assert _actions(snapshot) == {1: "retain", 2: "delete", 3: "retain"}
    assert snapshot.deletion_targets == (refs[1],)
