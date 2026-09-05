"""Torch-free tests for how the publication catalog reads a torn row.

``publications.jsonl`` is load-bearing for restore, resume and reconcile, and
its reader is deliberately fail-loud: silently dropping a published checkpoint's
identity row is worse than refusing to read the file. These tests pin the reader
staying loud while gaining a *diagnosis* -- separating a row that was torn
mid-write, and is therefore repairable, from a row that is corrupt, and is not.

The distinction is drawn on the terminating newline, which is what commits a
record, and not on parseability. That matters in both directions: a reader that
skipped unparseable rows would lose a published checkpoint, and a reader that
rejected every unterminated line would refuse to read a catalog that is entirely
intact. Both directions are exercised below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.checkpoint.catalog import (
    CheckpointCatalog,
    IncompletePublicationRecordError,
    publication_catalog_path,
    reconcile_publication,
)
from tpen.checkpoint.reference import CheckpointRef


def _manifest(step: int) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": {"model": "model.pt"},
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "run", "git_sha": "deadbeef"},
    }


def _write_checkpoint(root: Path, step: int = 7) -> Path:
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(b"immutable-model")
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(_manifest(step), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint_dir


def _catalog_with(tmp_path: Path, *, steps: tuple[int, ...] = (7,)) -> CheckpointCatalog:
    catalog = CheckpointCatalog(publication_catalog_path(tmp_path))
    for step in steps:
        catalog.publish(CheckpointRef.from_directory(_write_checkpoint(tmp_path, step)))
    return catalog


# --- the reader keeps refusing to read corruption ----------------------------


def test_a_terminated_malformed_row_still_raises(tmp_path: Path) -> None:
    """Fail-loud on committed corruption is the behaviour being preserved."""

    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all}\n")

    with pytest.raises(ValueError) as caught:
        catalog.records()

    assert not isinstance(caught.value, IncompletePublicationRecordError), (
        "a terminated row was committed; it must not be reported as repairable"
    )
    assert "invalid checkpoint publication" in str(caught.value)


def test_a_row_that_parses_but_has_the_wrong_schema_still_raises(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "something.else/v1"}) + "\n")

    with pytest.raises(ValueError, match="unsupported checkpoint publication"):
        catalog.records()


# --- but it now distinguishes a torn final row -------------------------------


def test_an_unterminated_final_row_raises_the_recoverable_error(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "tpen.checkpoint-publica')  # torn mid-write

    with pytest.raises(IncompletePublicationRecordError) as caught:
        catalog.records()

    message = str(caught.value)
    assert "never committed" in message
    assert "reconcile_publication" in message, "the error must name the repair"
    assert "every earlier row is intact" in message


def test_the_recoverable_error_is_still_a_value_error(tmp_path: Path) -> None:
    """Existing ``except ValueError`` callers must keep working unchanged."""

    assert issubclass(IncompletePublicationRecordError, ValueError)


# --- the over-restrictive direction ------------------------------------------


def test_an_unterminated_final_row_that_parses_is_yielded_not_rejected(
    tmp_path: Path,
) -> None:
    """A complete record that merely lost its terminator is not corruption.

    This is the mutation that closes too far: rejecting every unterminated line
    would pass every other test in this file and only surface when a real
    restore could not start on an intact catalog.
    """

    catalog = _catalog_with(tmp_path)
    expected = catalog.records()
    assert len(expected) == 1

    # Strip the final newline, leaving a complete but uncommitted row.
    text = catalog.path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    catalog.path.write_text(text[:-1], encoding="utf-8")

    assert catalog.records() == expected


def test_a_well_formed_catalog_reads_unchanged(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path, steps=(7, 9))

    refs = catalog.records()

    assert [ref.next_iteration for ref in refs] == [7, 9]


def test_blank_lines_are_still_skipped(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")

    assert len(catalog.records()) == 1


# --- the falsifier, end to end -----------------------------------------------


def test_publishing_after_a_torn_row_never_merges_into_it(tmp_path: Path) -> None:
    """The core defect: an interrupted append destroying the NEXT record.

    ``publish`` reads before it writes, so the torn catalog blocks it -- which
    is the fail-loud behaviour. The bytes still must not merge when the row is
    appended through the shared primitive, so that is asserted directly.
    """

    from tpen.artifacts import append_jsonl

    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "tpen.checkpoint-publica')

    # A torn catalog blocks new publications rather than corrupting them.
    with pytest.raises(IncompletePublicationRecordError):
        catalog.publish(CheckpointRef.from_directory(_write_checkpoint(tmp_path, 9)))

    append_jsonl(catalog.path, {"schema": "tpen.checkpoint-publication/v1"})

    lines = catalog.path.read_text(encoding="utf-8").splitlines()
    assert lines[-2] == '{"schema": "tpen.checkpoint-publica'
    assert json.loads(lines[-1]) == {"schema": "tpen.checkpoint-publication/v1"}, (
        "the record written after the torn row must survive intact"
    )


def test_the_repair_named_in_the_error_message_actually_works(tmp_path: Path) -> None:
    """Follow the recipe the exception prints, and the catalog must come back.

    Written because the first draft of that message named
    ``reconcile_publication`` alone, which cannot work: it reads the catalog
    before it writes, so it raises on the very file it was meant to repair.
    """

    checkpoint_dir = _write_checkpoint(tmp_path, 7)
    catalog = CheckpointCatalog(publication_catalog_path(tmp_path))
    catalog.publish(CheckpointRef.from_directory(checkpoint_dir))
    expected = catalog.records()

    # Lose the catalog row to a torn append.
    text = catalog.path.read_text(encoding="utf-8")
    catalog.path.write_text(text + '{"schema": "tpen.checkpoint-publica', encoding="utf-8")
    with pytest.raises(IncompletePublicationRecordError):
        catalog.records()

    # Step 1 of the recipe: drop the unterminated final line.
    body = catalog.path.read_text(encoding="utf-8")
    catalog.path.write_text(body[: body.rindex("\n") + 1], encoding="utf-8")

    # Step 2: reconcile rebuilds any missing row from the committed checkpoint.
    reconcile_publication(tmp_path, checkpoint_dir)

    assert catalog.records() == expected


def test_reconcile_rebuilds_a_row_lost_entirely_to_a_torn_append(tmp_path: Path) -> None:
    """The recoverability that makes this error class 'recoverable' at all."""

    checkpoint_dir = _write_checkpoint(tmp_path, 7)
    catalog = CheckpointCatalog(publication_catalog_path(tmp_path))

    reconcile_publication(tmp_path, checkpoint_dir)

    refs = catalog.records()
    assert [ref.next_iteration for ref in refs] == [7]
