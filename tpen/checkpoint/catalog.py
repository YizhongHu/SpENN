"""Append-only serialization and catalog for published checkpoint refs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from tpen.artifacts import append_jsonl

from .artifact import read_latest, write_latest
from .receipt import backfill_publication_receipt, publication_receipt_path
from .reference import (
    CHECKPOINT_REF_SCHEMA,
    CheckpointRef,
    deserialize_checkpoint_ref,
    serialize_checkpoint_ref,
)
from .schema import read_manifest

PUBLICATION_CATALOG_FILENAME = "publications.jsonl"
PUBLICATION_RECORD_SCHEMA = "tpen.checkpoint-publication/v1"


class CheckpointCatalog:
    """Append-only JSONL catalog of immutable checkpoint publications.

    A record contains only the immutable checkpoint ref and its content ID.
    Lifecycle fields such as ``latest``, ``pinned``, ``deleted``, or ``status``
    intentionally do not exist here; those policies belong to later layers.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def publish(self, ref: CheckpointRef) -> CheckpointRef:
        """Publish ``ref`` while preserving append-only history.

        ``content_id`` is the publication identity.  A replay is idempotent
        when the parsed, canonical full CheckpointRef mapping is equal to the
        existing row; raw JSON formatting is intentionally irrelevant.  A
        same-content-id row with different serialized content is a conflict.
        Although ``content_id`` is path-independent, the compared mapping
        includes ``checkpoint_dir``.  A relocation therefore conflicts rather
        than silently creating two locations for one catalog identity: this
        append-only catalog cannot rewrite the original location, and accepting
        both would make the publication target ambiguous.  The duplicate check
        scans the append-only JSONL catalog, so its read cost is O(n) in the
        number of existing publications.
        """

        if not isinstance(ref, CheckpointRef):
            raise TypeError(f"expected CheckpointRef, got {type(ref).__name__}")
        ref.validate()
        serialized_ref = serialize_checkpoint_ref(ref)
        for existing in self.iter_publications():
            if existing.content_id != ref.content_id:
                continue
            if existing.to_dict() == serialized_ref:
                return ref
            raise ValueError(
                "conflicting checkpoint publication for content_id "
                f"{ref.content_id}"
            )
        append_jsonl(self.path, {"schema": PUBLICATION_RECORD_SCHEMA, "ref": serialized_ref})
        return ref

    append = publish

    def iter_publications(self) -> Iterator[CheckpointRef]:
        """Yield serialized refs in append order, rejecting malformed rows."""

        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid checkpoint publication at {self.path}:{line_number}: {exc}"
                    ) from exc
                yield _deserialize_record(record, path=self.path, line_number=line_number)

    def records(self) -> tuple[CheckpointRef, ...]:
        """Return all catalog refs in append order."""

        return tuple(self.iter_publications())

    read = records
    load = records


PublicationCatalog = CheckpointCatalog


def publication_catalog_path(checkpoint_root: str | Path) -> Path:
    """Return the default append-only publication catalog path."""

    return Path(checkpoint_root) / PUBLICATION_CATALOG_FILENAME


def append_publication(path: str | Path, ref: CheckpointRef) -> CheckpointRef:
    """Append one validated publication to a catalog path."""

    return CheckpointCatalog(path).publish(ref)


def read_publications(path: str | Path) -> tuple[CheckpointRef, ...]:
    """Read all publication refs from a catalog path."""

    return CheckpointCatalog(path).records()


def reconcile_publication(
    checkpoint_root: str | Path, checkpoint_dir: str | Path
) -> CheckpointRef:
    """Ensure one committed checkpoint has its catalog row, latest pointer, and receipt.

    The directory rename is the checkpoint commit, while catalog publication,
    ``latest.json``, and the publication receipt are separate durable
    operations.  A retry therefore needs to reconcile all three instead of
    treating a complete directory as proof that all of them succeeded.
    ``CheckpointCatalog.publish`` supplies the append-idempotent row
    operation: an existing identical ref is not appended again, while a
    conflicting row still fails closed.

    This helper only repairs non-destructive indexes. It must not rewrite the
    committed payload or remove any checkpoint directories; storage disposition
    is outside TPEN.

    The receipt backfill is last and best-effort, matching
    ``save_checkpoint``'s own ordering and failure handling: unlike the
    catalog row and ``latest.json`` above, a missing or unrecoverable receipt
    never fails this call. See
    ``tpen.checkpoint.receipt.backfill_publication_receipt``.
    """

    root = Path(checkpoint_root)
    directory = Path(checkpoint_dir)
    ref = CheckpointRef.from_directory(directory)
    CheckpointCatalog(publication_catalog_path(root)).publish(ref)

    manifest = read_manifest(directory / "manifest.json", mode="model_only")
    expected_latest = {
        "checkpoint_dir": directory.name,
        "step": ref.next_iteration,
        "created_at_unix": manifest.created_at_unix,
    }
    try:
        latest = read_latest(root)
    except (FileNotFoundError, ValueError):
        latest = None
    if latest != expected_latest:
        write_latest(
            root,
            directory,
            step=ref.next_iteration,
            created_at_unix=manifest.created_at_unix,
        )
    backfill_publication_receipt(
        ref, directory, manifest.files, publication_receipt_path(root)
    )
    return ref


def _deserialize_record(record: Any, *, path: Path, line_number: int) -> CheckpointRef:
    if not isinstance(record, Mapping) or record.get("schema") != PUBLICATION_RECORD_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint publication at {path}:{line_number}; "
            f"expected schema {PUBLICATION_RECORD_SCHEMA!r}"
        )
    ref_data = record.get("ref")
    if not isinstance(ref_data, Mapping) or ref_data.get("schema") != CHECKPOINT_REF_SCHEMA:
        raise ValueError(f"publication at {path}:{line_number} lacks a valid CheckpointRef")
    return deserialize_checkpoint_ref(ref_data)


__all__ = [
    "PUBLICATION_CATALOG_FILENAME",
    "PUBLICATION_RECORD_SCHEMA",
    "CheckpointCatalog",
    "PublicationCatalog",
    "append_publication",
    "publication_catalog_path",
    "reconcile_publication",
    "read_publications",
]
