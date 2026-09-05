"""Torch-free tests for the single-write, torn-tail-safe append primitive.

The defect these pin: ``append_jsonl`` could leave an unterminated line that the
*next* append concatenated onto -- destroying a later, entirely valid record
rather than merely losing the interrupted one.

The CAUSE is size-dependent, and the original diagnosis was corrected by
measurement. For a record above the 8 KiB text buffer the two writes really do
flush separately (measured: 40,014 of 40,015 bytes on disk before the newline),
which is the ``occurrences.jsonl`` case. For a smaller record -- a catalog row,
a receipt row, most log lines -- both writes coalesce into a single flush, so
the two-call structure was NOT the trigger there; the exposure is a short write
or a failure at flush, with the same damage shape. Do not repeat the
two-separate-writes story about small records: it is measurably wrong for them.

Two things make a test here easy to get wrong, so they are stated up front:

* **Appending twice proves nothing.** The failure is a crash *between* writes.
  Every test below either constructs the torn state on disk explicitly or
  induces a real short write, then appends through the production path.
* **The single-write property is a syscall count, not a file size.** An
  on-disk-size measurement observes flush visibility and cannot distinguish one
  write from two that coalesced in a buffer. It is pinned here with a handle
  that counts ``write`` calls.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from tpen.artifacts import append_jsonl
from tpen.distributed import (
    ExecutionTopology,
    ProfileRecord,
    ProfileScope,
    RankLocalJSONLWriter,
)
from tpen.durable_append import PartialAppendError, append_record, ends_without_newline
from tpen.logging.base import LogRecord
from tpen.logging.jsonl import JSONL
from tpen.process_resources import ProcessResourceResult

REPO_ROOT = Path(__file__).resolve().parents[2]


class _RecordingHandle:
    """Wrap a binary handle and record the size of every ``write`` call."""

    def __init__(self, inner, calls: list[int], *, truncate_by: int = 0) -> None:
        self._inner = inner
        self._calls = calls
        self._truncate_by = truncate_by

    def write(self, data: bytes) -> int:
        self._calls.append(len(data))
        if self._truncate_by:
            # A real short write: the kernel stores a prefix and reports it.
            return self._inner.write(data[: -self._truncate_by])
        return self._inner.write(data)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def write_calls(monkeypatch: pytest.MonkeyPatch):
    """Count writes on binary-append handles only, leaving reads untouched."""

    def _install(*, truncate_by: int = 0) -> list[int]:
        calls: list[int] = []
        real_open = Path.open

        def spy(self, mode="r", *args, **kwargs):
            handle = real_open(self, mode, *args, **kwargs)
            if "a" in mode and "b" in mode:
                return _RecordingHandle(handle, calls, truncate_by=truncate_by)
            return handle

        monkeypatch.setattr(Path, "open", spy)
        return calls

    return _install


# --- I1: exactly one write call per record -----------------------------------


def test_a_record_reaches_the_kernel_in_exactly_one_write(write_calls, tmp_path: Path) -> None:
    calls = write_calls()
    path = tmp_path / "log.jsonl"

    append_record(path, '{"a": 1}')

    assert calls == [len(b'{"a": 1}\n')], "record and newline must be one write, not two"


def test_a_record_larger_than_the_text_buffer_is_still_one_write(
    write_calls, tmp_path: Path
) -> None:
    # Over 8 KiB is where the old text-mode path measurably flushed mid-record.
    calls = write_calls()
    path = tmp_path / "big.jsonl"
    record = json.dumps({"blob": "x" * 40_000}, sort_keys=True)

    append_record(path, record)

    assert calls == [len(record.encode()) + 1]
    assert path.read_text(encoding="utf-8") == record + "\n"


def test_closing_out_a_torn_line_does_not_cost_a_second_write(
    write_calls, tmp_path: Path
) -> None:
    """The repair newline rides in the same write, not a preceding one.

    Terminating the torn line separately would reintroduce the very two-write
    window the primitive exists to remove.
    """

    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n{"torn": ', encoding="utf-8")
    calls = write_calls()

    append_record(path, '{"b": 2}')

    assert calls == [len(b'\n{"b": 2}\n')], "repair newline must ride in the same write"


# --- I2: a torn tail never merges with the next record -----------------------


def test_a_new_record_does_not_join_onto_an_unterminated_line(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n{"torn": ', encoding="utf-8")

    append_record(path, '{"b": 2}')

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"a": 1}', '{"torn": ', '{"b": 2}']
    assert not any("torn" in line and '"b"' in line for line in lines), (
        "the torn bytes and the new record must never share a line"
    )
    assert json.loads(lines[-1]) == {"b": 2}, "the new record must survive intact"


def test_the_torn_state_produced_by_a_real_short_write_also_does_not_merge(
    write_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Construct the torn state through the actual failure, not by hand.

    Dropping the final byte of the write drops exactly the terminator, which is
    the shape a crash between flush and completion leaves behind.
    """

    path = tmp_path / "log.jsonl"
    append_record(path, '{"a": 1}')
    write_calls(truncate_by=1)

    with pytest.raises(PartialAppendError):
        append_record(path, '{"b": 2}')

    assert ends_without_newline(path), "the failed append must leave an unterminated line"

    # Lift the induced failure: the next append is the production path recovering.
    monkeypatch.undo()
    append_record(path, '{"c": 3}')

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"a": 1}', '{"b": 2}', '{"c": 3}']
    assert json.loads(lines[-1]) == {"c": 3}


# --- I3: a short write is detected, never silent -----------------------------


def test_a_short_write_raises_instead_of_reporting_success(
    write_calls, tmp_path: Path
) -> None:
    path = tmp_path / "log.jsonl"
    write_calls(truncate_by=3)

    with pytest.raises(PartialAppendError) as caught:
        append_record(path, '{"a": 1}')

    assert "partial append" in str(caught.value)
    assert isinstance(caught.value, OSError), "callers already handle OSError here"


# --- I5: the guard is inert on well-formed files -----------------------------


def test_appending_to_a_terminated_file_inserts_no_blank_line(tmp_path: Path) -> None:
    """The over-restrictive direction: a guard that always repairs corrupts."""

    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")

    append_record(path, '{"b": 2}')

    assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"b": 2}\n'


def test_a_fresh_file_gets_no_leading_newline(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "log.jsonl"

    append_record(path, '{"a": 1}')

    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_ends_without_newline_reports_false_for_absent_and_empty_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")

    assert ends_without_newline(missing) is False
    assert ends_without_newline(empty) is False


# --- one record is one line --------------------------------------------------


def test_a_record_containing_a_newline_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one record is one line"):
        append_record(tmp_path / "log.jsonl", '{"a": 1}\n{"b": 2}')


def test_a_json_escaped_newline_is_accepted(tmp_path: Path) -> None:
    """The over-restrictive direction: escaped newlines are ordinary content."""

    path = tmp_path / "log.jsonl"
    record = json.dumps({"text": "first\nsecond"}, sort_keys=True)
    assert "\n" not in record

    append_record(path, record)

    assert json.loads(path.read_text(encoding="utf-8")) == {"text": "first\nsecond"}


# --- the writers that route through the primitive ----------------------------


def _torn(path: Path) -> None:
    path.write_text('{"a": 1}\n{"torn": ', encoding="utf-8")


def test_append_jsonl_does_not_join_onto_an_unterminated_line(tmp_path: Path) -> None:
    path = tmp_path / "publications.jsonl"
    _torn(path)

    append_jsonl(path, {"b": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == '{"b": 2}'
    assert lines[-2] == '{"torn": '


def test_jsonl_logger_does_not_join_onto_an_unterminated_line(tmp_path: Path) -> None:
    """`tpen/logging/jsonl.py` reimplemented the idiom under its own name.

    "``append_jsonl`` repaired while ``logging/jsonl.py`` still tears" is the
    named failure mode for this slice; this is the test that would catch it.
    """

    path = tmp_path / "metrics.jsonl"
    _torn(path)

    JSONL(path).log(LogRecord(step=1, namespace="train", metrics={"loss": 0.5}))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[-2] == '{"torn": '
    assert json.loads(lines[-1])["metrics"] == {"loss": 0.5}


# --- census, re-run at this head as an anti-regression guard ------------------

TEXT_APPEND_OPEN = re.compile(r"""open\(\s*["']a["']""")

# Text-mode append writers deliberately NOT routed through the primitive in this
# slice, tracked as item bc8925a8 rather than left silent.
#
# The name is historical and imprecise: csv.py genuinely writes many rows per
# open, but sidecar.py only does so via ``extend``. Its production call path is
# ``append`` -> ``extend((receipt,))`` -- ONE record per open, and its reader
# raises on a bad row exactly as the catalog's does. It is out of scope by
# ruling, not because its failure shape differs.
KNOWN_BATCH_WRITERS = {
    Path("tpen/logging/csv.py"),
    Path("tpen/statistics/sidecar.py"),
}


def test_no_one_record_writer_still_opens_a_file_in_text_append_mode() -> None:
    """Behaviour-scoped census: the property is a write pattern, not a name.

    A name-scoped search for callers of ``append_jsonl`` reports three sites and
    misses the modules that reimplement the idiom -- which is how the original
    census undercounted. This greps for the *pattern* instead, so a newly added
    append writer anywhere in ``tpen/`` fails this test and must be classified
    as one-record or batch rather than quietly inheriting the exposure.

    Blind to: writers outside ``tpen/``, ``os.open``, and ``"a+"``/``"ab"``.
    """

    found = {
        source.relative_to(REPO_ROOT)
        for source in (REPO_ROOT / "tpen").rglob("*.py")
        if TEXT_APPEND_OPEN.search(source.read_text(encoding="utf-8"))
    }

    assert found == KNOWN_BATCH_WRITERS, (
        "text-mode append writers changed; classify each as one-record "
        f"(route through append_record) or batch. Unexpected: {found - KNOWN_BATCH_WRITERS}, "
        f"missing: {KNOWN_BATCH_WRITERS - found}"
    )


def test_the_removed_trailing_newline_helper_has_exactly_one_replacement() -> None:
    """One invariant, one mechanism -- two would be free to drift apart."""

    receipt = (REPO_ROOT / "tpen" / "checkpoint" / "receipt.py").read_text(encoding="utf-8")

    assert "def _ensure_trailing_newline" not in receipt


# --- buffering=0 is a load-bearing token, and nothing above pins it ----------


class _DropsOneByteOnce(io.RawIOBase):
    """A raw handle whose FIRST write stores one byte fewer than asked.

    This is the shape POSIX permits for a regular file: a partial store with a
    short count returned, and no error.  Every later write is honoured in full,
    so a layer that RETRIES the remainder completes the record -- which is
    precisely the behaviour these tests must be able to see.
    """

    def __init__(self, inner: io.FileIO) -> None:
        self._inner = inner
        self._shortened = False

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        payload = bytes(data)
        if not self._shortened and len(payload) > 1:
            self._shortened = True
            return self._inner.write(payload[:-1])
        return self._inner.write(payload)

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            super().close()


@pytest.fixture
def short_write_below_the_opened_layer(monkeypatch: pytest.MonkeyPatch):
    """Inject the short write BELOW whatever layer ``append_record`` opens.

    The ``write_calls`` fixture wraps the object ``Path.open`` returns, so it
    sits ABOVE any buffering and cannot observe it.  This fixture instead
    honours the ``buffering`` the production code asks for: with ``buffering=0``
    the caller receives the raw shim and sees the short count; with any other
    buffering it receives a ``BufferedWriter`` over the shim, which returns
    ``len(data)`` unconditionally and loops the raw handle until every byte is
    stored.  The outcome is therefore decided by the production open, not by
    the test's own wrapper.
    """

    real_open = Path.open

    def spy(self, mode="r", *args, **kwargs):
        if "a" in mode and "b" in mode:
            buffering = args[0] if args else kwargs.get("buffering", -1)
            shim = _DropsOneByteOnce(io.FileIO(str(self), "a"))
            return shim if buffering == 0 else io.BufferedWriter(shim)
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)


def test_the_append_handle_is_unbuffered_so_a_short_write_is_reported_not_retried(
    short_write_below_the_opened_layer, tmp_path: Path
) -> None:
    """``buffering=0`` is the token that makes BOTH I1 and I3 true.

    Deleting it leaves every other test in this file green: the recording
    handle counts one ``write`` either way, and ``BufferedWriter.write`` returns
    the full length, so ``written != len(encoded)`` can never fire and
    ``PartialAppendError`` becomes unreachable.  Measured before this test
    existed: removing ``buffering=0`` killed 0 of 25 tests.
    """

    path = tmp_path / "log.jsonl"

    with pytest.raises(PartialAppendError):
        append_record(path, '{"a": 1}')


def test_the_production_append_opens_a_raw_unbuffered_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin the property directly: the handle must be raw, not buffered.

    Stated as ``RawIOBase`` rather than ``buffering == 0`` so a refactor to a
    different unbuffered spelling still passes, while any buffered handle --
    which silently retries short writes and may split one record across several
    syscalls -- fails.
    """

    real_open = Path.open
    handles: list[object] = []

    def spy(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if "a" in mode and "b" in mode:
            handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", spy)

    append_record(tmp_path / "log.jsonl", '{"a": 1}')

    assert len(handles) == 1
    assert isinstance(handles[0], io.RawIOBase), (
        "append_record must open an unbuffered raw handle: a BufferedWriter "
        "returns len(data) from write() unconditionally, which defeats the "
        "short-write check, and loops the raw handle, which defeats one-write."
    )


# --- the unstated ASCII invariant the torn-row diagnosis depends on ----------


def test_every_routed_writer_emits_ascii_only_bytes(tmp_path: Path) -> None:
    """The torn-row DIAGNOSIS silently depends on records being pure ASCII.

    ``iter_publications`` opens the catalog as UTF-8 text.  A torn write that
    split a multi-byte character raises ``UnicodeDecodeError`` out of the file
    iteration itself -- before the ``except json.JSONDecodeError`` that produces
    ``IncompletePublicationRecordError`` -- so the operator would get no
    diagnosis, no repair recipe and no line number.  That cannot happen today
    only because every routed writer uses ``json.dumps``'s default
    ``ensure_ascii=True``, an invariant stated nowhere and pinned by nothing.
    This states it.
    """

    direct = tmp_path / "direct.jsonl"
    append_jsonl(direct, {"text": "é中\U0001f600"})

    metrics = tmp_path / "metrics.jsonl"
    JSONL(metrics).log(LogRecord(step=1, namespace="é", metrics={"loss": 0.5}))

    topology = ExecutionTopology(
        global_rank=0, global_size=1, local_rank=0, local_size=1,
        node_rank=0, node_size=1, host="éhost", pid=1, device="cpu", job_id=None,
    )
    profiles = tmp_path / "profiles"
    RankLocalJSONLWriter(profiles, topology).write(
        ProfileRecord(
            ProfileScope.PROCESS,
            0.0,
            topology,
            process=ProcessResourceResult(
                user_cpu_seconds=1, system_cpu_seconds=1,
                read_block_operations=1, write_block_operations=1,
                voluntary_context_switches=1, involuntary_context_switches=1,
                peak_rss_mb=1,
            ),
        )
    )

    written = [direct, metrics, *profiles.rglob("*.jsonl")]
    assert len(written) >= 3
    for path in written:
        raw = path.read_bytes()
        assert raw.isascii(), f"{path} emitted non-ASCII bytes: {raw!r}"
