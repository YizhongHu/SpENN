"""Logger tests for configured scaffold runs."""

from __future__ import annotations

import json
from pathlib import Path

from spenn.logging import CSV, JSONL, LogRecord
from spenn.run import load_config, run_from_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "hooke" / "configs" / "smoke" / "scaffold.yaml"


def test_csv_and_jsonl_loggers_write_scaffold_metric(tmp_path: Path) -> None:
    """The scaffold lifecycle writes CSV and JSONL metric records."""

    code = run_from_config(load_config(str(CONFIG), [f"run.root={tmp_path}"]), config_path=str(CONFIG), command="pytest loggers")

    assert code == 0
    run_dir = _single_run_dir(tmp_path)
    assert (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines() == [
        "step,namespace,key,value",
        "0,scaffold,scaffold_completed,true",
    ]
    assert [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ] == [
        {
            "event": None,
            "metrics": {"scaffold_completed": True},
            "namespace": "scaffold",
            "step": 0,
        }
    ]


def test_loggers_can_be_used_directly(tmp_path: Path) -> None:
    """Flat logger classes write records without run context."""

    csv = CSV(tmp_path / "metrics.csv")
    jsonl = JSONL(tmp_path / "metrics.jsonl")
    record = LogRecord(step=None, namespace="direct", metrics={"ok": True, "empty": None}, event="unit")

    csv.log(record)
    jsonl.log(record)

    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines() == [
        "step,namespace,key,value",
        ",direct,ok,true",
        ",direct,empty,",
    ]
    assert json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8")) == {
        "event": "unit",
        "metrics": {"empty": None, "ok": True},
        "namespace": "direct",
        "step": None,
    }


def _single_run_dir(root: Path) -> Path:
    run_dirs = sorted((root / "hooke_scaffold" / "scaffold").iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]
