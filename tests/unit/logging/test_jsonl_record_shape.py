"""Pin the on-disk shape of a ``metrics.jsonl`` record.

Written after a rehearsal harness read `record.get("key", "")` from every line,
collected ``{""}``, and reported "no MCSE emitted" against a row that carried
one -- a false red on a green system, produced by an instrument reading a schema
that does not exist. Nothing in the repository stated the real shape, so the
harness's belief was never checkable against anything.

Metric names live INSIDE the ``metrics`` mapping. There is no top-level ``key``
field, and every downstream reader -- job scripts, collectors, report tooling --
depends on that.
"""

from __future__ import annotations

import json
from pathlib import Path

from tpen.logging import JSONL
from tpen.logging.base import LogRecord


def test_a_metrics_record_carries_step_namespace_and_a_metrics_mapping(tmp_path: Path) -> None:
    """The record has exactly three top-level fields, and names are nested."""

    path = tmp_path / "metrics.jsonl"
    JSONL(path).log(
        LogRecord(step=7, namespace="eval/mcmc_energy", metrics={"local_energy_mcse": 1.5})
    )

    (line,) = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)

    assert set(record) == {"step", "namespace", "metrics"}
    assert record["step"] == 7
    assert record["namespace"] == "eval/mcmc_energy"
    assert record["metrics"] == {"local_energy_mcse": 1.5}

    # The specific misreading that caused the false red: there is no top-level
    # "key", so a reader keyed on one silently collects nothing from every line
    # rather than failing loudly.
    assert "key" not in record
    assert "local_energy_mcse" not in record


def test_appended_records_are_one_json_object_per_line(tmp_path: Path) -> None:
    """Append-only, one object per line, so a reader may parse line by line."""

    path = tmp_path / "metrics.jsonl"
    logger = JSONL(path)
    logger.log(LogRecord(step=1, namespace="train", metrics={"loss": 2.0}))
    logger.log(LogRecord(step=2, namespace="train", metrics={"loss": 1.0}))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["metrics"]["loss"] for line in lines] == [2.0, 1.0]

    # Union of the per-line metrics mappings is how a consumer discovers which
    # names a run emitted; assert that idiom works rather than only the shape.
    names: set[str] = set()
    for line in lines:
        names |= set(json.loads(line)["metrics"])
    assert names == {"loss"}
