"""Tests for diagnostic artifact index maintenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from spenn.diagnostics.artifacts import update_diagnostic_index


def test_diagnostic_index_preserves_unknown_entries_and_counts_rows(tmp_path: Path) -> None:
    index = tmp_path / "diagnostics" / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_dir": str(tmp_path),
                "artifacts": [{"name": "unknown_future", "kind": "json", "path": "diagnostics/future.json"}],
            }
        ),
        encoding="utf-8",
    )
    table = tmp_path / "diagnostics" / "probe" / "probe.csv"
    table.parent.mkdir(parents=True)
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"x": 1})
        writer.writerow({"x": 2})

    update_diagnostic_index(
        run_dir=tmp_path,
        artifacts=[
            {
                "name": "probe",
                "kind": "csv",
                "path": table,
                "enabled": True,
                "expected": True,
                "created_by": "Probe",
            }
        ],
    )

    payload = json.loads(index.read_text())
    by_name = {entry["name"]: entry for entry in payload["artifacts"]}
    assert "unknown_future" in by_name
    assert by_name["probe"]["path"] == "diagnostics/probe/probe.csv"
    assert by_name["probe"]["exists"] is True
    assert by_name["probe"]["readable"] is True
    assert by_name["probe"]["rows"] == 2


def test_diagnostic_index_records_disabled_artifact(tmp_path: Path) -> None:
    update_diagnostic_index(
        run_dir=tmp_path,
        artifacts=[
            {
                "name": "sampled_eval_table",
                "kind": "csv",
                "path": "diagnostics/energy/sampled_eval_table.csv",
                "enabled": False,
                "expected": False,
                "created_by": "EnergyEvaluation",
                "warning": "disabled",
            }
        ],
    )

    payload = json.loads((tmp_path / "diagnostics" / "index.json").read_text())
    entry = payload["artifacts"][0]
    assert entry["enabled"] is False
    assert entry["expected"] is False
    assert entry["exists"] is False
    assert entry["warning"] == "disabled"
