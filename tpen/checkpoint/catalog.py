"""Append-only serialization and catalog for published checkpoint refs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from tpen.artifacts import append_jsonl

from .reference import (
    CHECKPOINT_REF_SCHEMA,
    CheckpointRef,
    deserialize_checkpoint_ref,
    serialize_checkpoint_ref,
)

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
        """Validate and append one publication record without rewriting prior rows."""

        if not isinstance(ref, CheckpointRef):
            raise TypeError(f"expected CheckpointRef, got {type(ref).__name__}")
        ref.validate()
        append_jsonl(
            self.path,
            {
                "schema": PUBLICATION_RECORD_SCHEMA,
                "ref": serialize_checkpoint_ref(ref),
            },
        )
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
    "read_publications",
]
