"""Typed publication size and duration receipts for one checkpoint write.

A publication receipt is a second, append-only index alongside
``publications.jsonl`` (see :mod:`tpen.checkpoint.catalog`). Where the
publication catalog carries identity, the receipt carries the bytes and
durations of the write: a durable size fact that exists independently of
whether the ``ResourceUsage`` callback is configured for the run.

**Sequencing, for a Stage-T3 reader building fault-injection tests from this
module alone.** ``save_checkpoint`` (``tpen/checkpoint/save.py``) writes every
component file into a ``.tmp`` staging directory, writes ``manifest.json`` and
``COMPLETE``, then commits with ``tmp_dir.rename(final_dir)``. Only after that
rename does it construct the :class:`~tpen.checkpoint.reference.CheckpointRef`,
publish it to the catalog, and update ``latest.json``. This module's receipt is
built and appended as the LAST step of that sequence, strictly after
``write_latest`` returns. It never runs before the rename (the files it
measures would not exist yet at their final names) and its own append does not
change, reorder, or replace any earlier step -- it is purely additive at the
end of the existing publication sequence.

**The manifest self-size cycle.** ``manifest.json`` cannot record its own byte
size before it is written. This module resolves that by never trying:
:class:`CheckpointManifest` carries no size field for itself, and no size data
of any kind is written into ``manifest.json`` or ``COMPLETE``. All sizes,
including the manifest's and ``COMPLETE``'s own, live only in this module's
separate ``publication_receipts.jsonl`` record, read via ``stat()`` after both
files are fully written and closed. Building the receipt never reopens
``manifest.json`` or ``COMPLETE`` for writing, so neither file is mutated by
this module.

**No directory scan.** :func:`measure_checkpoint_files` looks up exactly the
files the manifest's own ``files`` mapping names, plus the two files that are
always present and load-bearing enough to have named constants (``manifest``
and ``COMPLETE``). It never calls ``Path.iterdir``, ``Path.glob``,
``Path.rglob``, or ``os.walk``. A file present in the checkpoint directory but
not named by the manifest (there should never be one) is invisible to the
receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tpen.artifacts import append_jsonl

from .artifact import COMPLETE_MARKER
from .reference import CheckpointRef

PUBLICATION_RECEIPT_FILENAME = "publication_receipts.jsonl"
PUBLICATION_RECEIPT_SCHEMA = "tpen.checkpoint-publication-receipt/v1"
MANIFEST_FILENAME = "manifest.json"

#: Components whose bytes are the restorable model/train-resume state, as
#: opposed to descriptive metadata about that state. Matches the component
#: names ``save_checkpoint`` uses as keys in its ``files`` mapping.
PAYLOAD_COMPONENT_NAMES = frozenset({"model", "optimizer", "trainer", "sampler", "rng"})

#: Pseudo-component names for the two files every checkpoint directory has
#: that are not listed in the manifest's own ``files`` mapping.
_MANIFEST_COMPONENT = "manifest"
_COMPLETE_COMPONENT = "complete"


@dataclass(frozen=True, slots=True)
class CheckpointFileSize:
    """Logical byte size of one file inside a published checkpoint directory.

    Parameters
    ----------
    component : str
        Manifest component name (e.g. ``"model"``), or one of the two fixed
        pseudo-component names ``"manifest"`` / ``"complete"`` for the files
        every checkpoint directory carries outside the manifest's own
        ``files`` mapping.
    relative_path : str
        File name relative to the checkpoint directory.
    size_bytes : int
        ``Path.stat().st_size`` of the file, read after the file was closed.
        Never derived from a directory scan or from block allocation.
    """

    component: str
    relative_path: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be nonnegative, got {self.size_bytes}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of this file size record."""

        return {
            "component": self.component,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CheckpointPublished:
    """Scalar summary of one checkpoint's publication size and duration.

    Deliberately carries no per-file breakdown -- :class:`CheckpointPublicationReceipt`
    is the typed record for that -- so this stays small enough to log or hand
    to a metric sink on its own.

    Parameters
    ----------
    checkpoint_dir : str
        Name of the committed ``step_*`` directory (not a full path; stable
        across a run tree being moved or collected).
    content_id : str
        The published :class:`~tpen.checkpoint.reference.CheckpointRef`'s
        path-independent content identity.
    file_count : int
        Number of files this summary's totals were computed over. Always
        equal to ``len(files)`` on the sibling :class:`CheckpointPublicationReceipt`.
    payload_bytes : int
        Sum of :attr:`CheckpointFileSize.size_bytes` over files whose
        component is in :data:`PAYLOAD_COMPONENT_NAMES`.
    metadata_bytes : int
        Sum of :attr:`CheckpointFileSize.size_bytes` over the remaining files
        (``resolved_config``, ``manifest``, ``complete``).
    total_bytes : int
        ``payload_bytes + metadata_bytes``, equivalently the sum over every
        measured file. Never computed independently of the two components, so
        the identity cannot drift.
    write_duration_sec : float
        Wall-clock seconds from immediately before ``tmp_dir.mkdir`` to
        immediately after ``tmp_dir.rename(final_dir)`` in
        ``save_checkpoint``, measured with ``time.perf_counter()``. Covers
        every component write, hashing, and the manifest/``COMPLETE`` writes,
        but not catalog publication or the ``latest.json`` update. Always
        present and non-negative: the receipt is only ever built after a
        successful commit, so this value is never absent.
    publish_duration_sec : float
        Wall-clock seconds from immediately after ``tmp_dir.rename`` to
        immediately after ``write_latest`` returns, measured with
        ``time.perf_counter()``. Covers catalog-row publication and the
        ``latest.json`` update, not the receipt append itself. Always present
        and non-negative, for the same reason as ``write_duration_sec``.
    """

    checkpoint_dir: str
    content_id: str
    file_count: int
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    write_duration_sec: float
    publish_duration_sec: float

    def __post_init__(self) -> None:
        if self.file_count < 0:
            raise ValueError(f"file_count must be nonnegative, got {self.file_count}")
        for name in ("payload_bytes", "metadata_bytes", "total_bytes"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
        if self.payload_bytes + self.metadata_bytes != self.total_bytes:
            raise ValueError(
                "total_bytes must equal payload_bytes + metadata_bytes, got "
                f"{self.total_bytes} != {self.payload_bytes} + {self.metadata_bytes}"
            )
        for name in ("write_duration_sec", "publish_duration_sec"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of this scalar summary."""

        return {
            "checkpoint_dir": self.checkpoint_dir,
            "content_id": self.content_id,
            "file_count": self.file_count,
            "payload_bytes": self.payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "write_duration_sec": self.write_duration_sec,
            "publish_duration_sec": self.publish_duration_sec,
        }


@dataclass(frozen=True, slots=True)
class CheckpointPublicationReceipt:
    """Full typed receipt: every measured file's size plus the scalar summary.

    Parameters
    ----------
    summary : CheckpointPublished
        Scalar totals and durations.
    files : tuple of CheckpointFileSize
        One entry per measured file, in the same order
        :func:`measure_checkpoint_files` returned them.
    """

    summary: CheckpointPublished
    files: tuple[CheckpointFileSize, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the full receipt."""

        return {
            "schema": PUBLICATION_RECEIPT_SCHEMA,
            "summary": self.summary.to_dict(),
            "files": [entry.to_dict() for entry in self.files],
        }


def measure_checkpoint_files(
    checkpoint_dir: Path, files: Mapping[str, str]
) -> tuple[CheckpointFileSize, ...]:
    """Return typed per-file logical sizes for one committed checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Committed (post-rename) ``step_*`` directory.
    files : Mapping[str, str]
        The manifest's own component-to-relative-filename mapping (e.g.
        ``{"model": "model.pt", ...}``), exactly as written to
        ``manifest.json``.

    Returns
    -------
    tuple of CheckpointFileSize
        One entry per file in ``files``, in mapping-iteration order, followed
        by the fixed ``manifest`` and ``complete`` entries. Sizes come from
        ``Path.stat().st_size``; no directory is ever listed or walked.
    """

    entries = [
        CheckpointFileSize(
            component=component,
            relative_path=relative_path,
            size_bytes=(checkpoint_dir / relative_path).stat().st_size,
        )
        for component, relative_path in files.items()
    ]
    entries.append(
        CheckpointFileSize(
            component=_MANIFEST_COMPONENT,
            relative_path=MANIFEST_FILENAME,
            size_bytes=(checkpoint_dir / MANIFEST_FILENAME).stat().st_size,
        )
    )
    entries.append(
        CheckpointFileSize(
            component=_COMPLETE_COMPONENT,
            relative_path=COMPLETE_MARKER,
            size_bytes=(checkpoint_dir / COMPLETE_MARKER).stat().st_size,
        )
    )
    return tuple(entries)


def build_publication_receipt(
    ref: CheckpointRef,
    checkpoint_dir: Path,
    files: Mapping[str, str],
    *,
    write_duration_sec: float,
    publish_duration_sec: float,
) -> CheckpointPublicationReceipt:
    """Measure and assemble the full typed receipt for one committed checkpoint.

    Parameters
    ----------
    ref : CheckpointRef
        The just-published ref this receipt describes.
    checkpoint_dir : pathlib.Path
        Committed (post-rename) ``step_*`` directory.
    files : Mapping[str, str]
        The manifest's own component-to-relative-filename mapping.
    write_duration_sec, publish_duration_sec : float
        Durations measured by the caller; see :class:`CheckpointPublished`.

    Returns
    -------
    CheckpointPublicationReceipt
        Full receipt whose summary totals are sums over ``files``, so the
        typed-sum identity holds by construction rather than by a separate
        check.
    """

    measured = measure_checkpoint_files(checkpoint_dir, files)
    payload_bytes = sum(
        entry.size_bytes for entry in measured if entry.component in PAYLOAD_COMPONENT_NAMES
    )
    metadata_bytes = sum(
        entry.size_bytes for entry in measured if entry.component not in PAYLOAD_COMPONENT_NAMES
    )
    summary = CheckpointPublished(
        checkpoint_dir=checkpoint_dir.name,
        content_id=ref.content_id,
        file_count=len(measured),
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        total_bytes=payload_bytes + metadata_bytes,
        write_duration_sec=write_duration_sec,
        publish_duration_sec=publish_duration_sec,
    )
    return CheckpointPublicationReceipt(summary=summary, files=measured)


def publication_receipt_path(checkpoint_root: str | Path) -> Path:
    """Return the default append-only publication receipt log path."""

    return Path(checkpoint_root) / PUBLICATION_RECEIPT_FILENAME


def append_publication_receipt(
    path: str | Path, receipt: CheckpointPublicationReceipt
) -> None:
    """Append one publication receipt as a JSONL record."""

    append_jsonl(Path(path), receipt.to_dict())


__all__ = [
    "PAYLOAD_COMPONENT_NAMES",
    "PUBLICATION_RECEIPT_FILENAME",
    "PUBLICATION_RECEIPT_SCHEMA",
    "CheckpointFileSize",
    "CheckpointPublicationReceipt",
    "CheckpointPublished",
    "append_publication_receipt",
    "build_publication_receipt",
    "measure_checkpoint_files",
    "publication_receipt_path",
]
