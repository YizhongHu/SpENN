"""Single-write, torn-tail-safe appends for line-oriented artifact files.

TPEN's run artifacts are append-only, line-oriented files -- ``publications.jsonl``,
``events.jsonl``, ``occurrences.jsonl``, the rank-local resource profiles, and the
metric logs.  They live on Netscratch, which is NFS.  This module owns the single
primitive every one-record-per-open writer uses to extend such a file, so that the
torn-write invariant has exactly one implementation rather than one per call site.

What this module does NOT rely on
---------------------------------
It relies on **neither** of the two guarantees a reader might reasonably assume.

*Atomicity of a single* ``write(2)`` *to a regular file.*  POSIX grants atomicity
only for writes to a pipe or FIFO below ``PIPE_BUF``.  A write to a regular file
may store fewer bytes than requested -- on ``ENOSPC``, on inode exhaustion, or on
a signal -- and it is allowed to do so silently by returning a short count.

*``O_APPEND`` offset resolution on NFS.*  NFSv3 has no server-side append
operation.  The client resolves the current end of file and then writes at that
offset, so two clients can obtain the same offset and overwrite one another.
``O_APPEND`` therefore does not hold across NFS clients at all.  Under a single
writer this is immaterial; under concurrent writers it is unfixed here and is
the separate concern of rank-sharding in the distributed program.

What it does provide
--------------------
1. **One write call per record.**  The record and its terminating newline are
   encoded together and handed to an *unbuffered* binary handle, so exactly one
   ``write`` reaches the kernel.  This narrows the interruption window to a
   single kernel-visible operation; it does not make that operation atomic.
2. **Detection instead of silence.**  A short write raises
   :class:`PartialAppendError` rather than returning normally, so a caller can
   never mistake a torn record for a stored one.
3. **Bounded damage.**  If a record is nevertheless left unterminated, the next
   append terminates it *in the same single write* that stores the new record.
   The new record therefore lands on its own line instead of being concatenated
   onto the torn bytes.  Without this, one interrupted append destroys the *next*
   record too -- a strictly worse outcome than losing the interrupted one.

The resulting guarantee is precise and deliberately modest: **damage from a torn
write is bounded to the torn record itself and is detectable when the file is
read.**  It is *not* "torn writes cannot occur", which is unachievable on the
filesystem these files actually live on.
"""

from __future__ import annotations

import io
from pathlib import Path

__all__ = ["PartialAppendError", "append_record", "ends_without_newline"]


class PartialAppendError(OSError):
    """Raised when the underlying write stored fewer bytes than the record.

    Subclasses :class:`OSError` so that existing ``except OSError`` handlers
    around artifact appends keep catching it; a partial append is a storage
    failure and callers already treat storage failures that way.
    """


def ends_without_newline(path: Path) -> bool:
    """Report whether `path` holds an unterminated final line.

    A missing or empty file has no unterminated line.  Only the final line of a
    file can lack a terminator, because a newline is precisely what separates it
    from a following line -- so this single byte is a complete test.

    Parameters
    ----------
    path : pathlib.Path
        File to inspect.

    Returns
    -------
    bool
        True when the file is non-empty and its last byte is not ``b"\\n"``.
    """

    try:
        with path.open("rb") as handle:
            if handle.seek(0, io.SEEK_END) == 0:
                return False
            handle.seek(-1, io.SEEK_END)
            return handle.read(1) != b"\n"
    except FileNotFoundError:
        # Losing the race against creation is indistinguishable from "no file",
        # and both mean there is no unterminated line to close out.
        return False


def append_record(path: str | Path, record: str) -> None:
    """Append `record` as one newline-terminated line using a single write.

    Parameters
    ----------
    path : str or pathlib.Path
        File to extend.  Parent directories are created if absent.
    record : str
        One complete record, without its trailing newline.  Must not contain a
        newline itself: this file format is one record per line, and an embedded
        newline would silently split one record into two unrelated rows.

    Raises
    ------
    ValueError
        If `record` contains a newline.
    PartialAppendError
        If the write stored fewer bytes than the encoded record.  The file is
        left with an unterminated final line, which the next
        :func:`append_record` closes out rather than joining onto.
    """

    if "\n" in record:
        raise ValueError("record must not contain a newline; one record is one line")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Closing out a torn predecessor is folded into THIS record's single write.
    # Terminating it in a separate write would reintroduce exactly the two-write
    # window the primitive exists to remove.
    prefix = "\n" if ends_without_newline(target) else ""
    encoded = f"{prefix}{record}\n".encode()

    # buffering=0 yields a raw FileIO: one write call in, one write syscall out,
    # with no BufferedWriter chunking the payload or retrying a short write.
    with target.open("ab", buffering=0) as handle:
        written = handle.write(encoded)

    if written != len(encoded):
        raise PartialAppendError(
            f"partial append to {target}: stored {written} of {len(encoded)} bytes; "
            "the final line is unterminated and the record was not committed"
        )
