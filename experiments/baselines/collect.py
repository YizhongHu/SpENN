"""Collect per-run baseline records into a single ``results.jsonl``.

Contract: every run directory produced by any code in the comparison drops one
``baseline_record.json`` next to its other artifacts. This collector walks a run
root, validates each of those files against
:class:`experiments.baselines.records.BaselineRecord`, and writes one JSON
object per line.

The collector holds no per-code knowledge on purpose -- translating a FermiNet
or TPEN run into the common record is the emitter's job. That keeps this file
from growing into an adapter framework.

Examples
--------
::

    uv run python -m experiments.baselines.collect \\
        --run-root path/to/runs --output path/to/results.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experiments.baselines.records import BaselineRecord, RecordValidationError

RECORD_FILENAME = "baseline_record.json"


@dataclass(frozen=True)
class CollectionReport:
    """Outcome of one collection pass.

    Parameters
    ----------
    records : list of BaselineRecord
        Records that validated, ordered by record-file path.
    failures : list of tuple of (str, str)
        ``(path, reason)`` for every record file that failed to parse or
        validate. Failures are reported, never silently dropped.
    """

    records: list[BaselineRecord]
    failures: list[tuple[str, str]]


def find_record_files(run_root: Path) -> list[Path]:
    """Return every per-run record file under ``run_root``.

    Parameters
    ----------
    run_root : pathlib.Path
        Directory tree to scan.

    Returns
    -------
    list of pathlib.Path
        Sorted paths, so collection order is deterministic.
    """

    return sorted(run_root.rglob(RECORD_FILENAME))


def collect(run_root: Path) -> CollectionReport:
    """Read and validate every record under ``run_root``.

    Parameters
    ----------
    run_root : pathlib.Path
        Directory tree to scan.

    Returns
    -------
    CollectionReport
        Valid records and per-file failures.

    Raises
    ------
    FileNotFoundError
        If ``run_root`` does not exist or is not a directory.
    """

    if not run_root.is_dir():
        raise FileNotFoundError(f"run root is not a directory: {run_root}")

    records: list[BaselineRecord] = []
    failures: list[tuple[str, str]] = []
    for path in find_record_files(run_root):
        # Report the path relative to the scanned root: an absolute facility
        # path must never reach a committed artifact.
        relative = path.parent.relative_to(run_root).as_posix() or "."
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append((relative, f"unreadable record: {error}"))
            continue
        try:
            record = BaselineRecord.from_json_dict(payload)
        except RecordValidationError as error:
            failures.append((relative, str(error)))
            continue
        # Stamp provenance only when the emitter left it blank; never overwrite
        # what the emitter asserted. The record is frozen, hence `replace`.
        if record.run_dir is None:
            record = dataclasses.replace(record, run_dir=relative)
        records.append(record)
    return CollectionReport(records=records, failures=failures)


def write_jsonl(records: Sequence[BaselineRecord], output: Path) -> None:
    """Write records as one JSON object per line.

    Parameters
    ----------
    records : sequence of BaselineRecord
        Validated records.
    output : pathlib.Path
        Destination file; parent directories are created.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=False) + "\n")


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True, help="directory tree of runs to scan")
    parser.add_argument("--output", type=Path, required=True, help="results.jsonl destination")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the collector.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` when every record validated, ``1`` when any record failed. A
        partial pass is a failure: a silently short results file would
        understate a comparison without anyone noticing.
    """

    args = _parser().parse_args(argv)
    report = collect(args.run_root)
    write_jsonl(report.records, args.output)
    print(f"collected {len(report.records)} record(s) -> {args.output}")
    for path, reason in report.failures:
        print(f"invalid record at {path}: {reason}", file=sys.stderr)
    if report.failures:
        print(f"{len(report.failures)} record(s) failed validation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
