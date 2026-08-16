"""Render one collected He-v1 attempt as a receipt a reader can audit.

The report states what it does not know as loudly as what it does:

- an absent value renders as ``absent``; it is never blank and never ``0``;
- every aggregate carries its coverage (``n/N rows``), so a median over two of
  nine rows cannot be read as a median over nine;
- ``local_energy_stderr`` is labelled an IID standard error, because pairing an
  IID bar against a correlation-corrected one is the comparison the
  estimator-pairing rule forbids; and
- a row whose delivered GPU did not match its requested stratum is listed as a
  failure, not a footnote.

This module imports no ``tpen`` (``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import absence  # noqa: E402
import collect as collect_stage  # noqa: E402
import layout  # noqa: E402
import plan as plan_stage  # noqa: E402

REPORT_FILENAME = "report.md"

#: Columns whose estimator has to be named wherever the number appears.
ESTIMATOR_LABELS: Mapping[str, str] = {
    "local_energy_stderr": "IID stderr (not an MCSE)",
}


def read_collected(results_root: str | Path, collect_attempt_id: str) -> dict[str, Any]:
    """Read one collected attempt."""

    path = layout.collect_attempt_dir(results_root, collect_attempt_id) / collect_stage.COLLECTED_FILENAME
    payload = layout.read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"collected table {path} is not a mapping")
    return payload


def render(collected: Mapping[str, Any]) -> str:
    """Render the markdown report for one collected attempt."""

    lines: list[str] = []
    lines += _header_lines(collected)
    lines += _row_lines(collected)
    lines += _aggregate_lines(collected)
    lines += _gate_lines(collected)
    lines += _failure_lines(collected)
    return "\n".join(lines) + "\n"


def _header_lines(collected: Mapping[str, Any]) -> list[str]:
    declared = bool(collected.get("gate_spec_declared"))
    lines = [
        f"# He-v1 study report — {collected['study']}",
        "",
        f"- plan attempt: `{collected['plan_attempt_id']}` (plan hash `{collected['plan_hash']}`)",
        f"- collect attempt: `{collected['collect_attempt_id']}`",
        f"- rows: {collected['n_rows']} ({collected['n_pass']} pass, {collected['n_fail']} fail)",
        f"- gate spec source: `{collected.get('gate_spec_source', 'absent')}`",
        f"- checkpoint hashing: {'on' if collected.get('checkpoint_hashing') else 'off'}",
        "- resume/restart: forbidden; rows are sized to finish",
        "",
    ]
    if not declared:
        lines += [
            "> No tolerance was predeclared for this attempt, so every value gate reports",
            "> `absent` with its observed value retained. That is the honest state of an",
            "> ungated run: the thresholds are predeclared in H-F1, and an absent gate is",
            "> never a pass.",
            "",
        ]
    return lines


def _row_lines(collected: Mapping[str, Any]) -> list[str]:
    lines = ["## Rows", "", "One line per planned row, in manifest order.", ""]
    header = [
        "row_id",
        "kind",
        "status",
        "seed",
        "step",
        "chain",
        "stratum",
        "delivered",
        "energy",
        f"stderr ({ESTIMATOR_LABELS['local_energy_stderr']})",
        "checkpoint_sha256",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in collected["rows"]:
        identity = row["identity"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{identity['row_id']}`",
                    str(identity["kind"]),
                    str(row["status"]),
                    str(identity["seed"]),
                    _cell(identity["checkpoint_step"]),
                    _cell(identity["chain"]),
                    str(identity["requested_stratum"]),
                    _cell(identity["delivered_device"]),
                    _metric(row, "local_energy_mean"),
                    _metric(row, "local_energy_stderr"),
                    _short_hash(_cell(identity["checkpoint_sha256"])),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _aggregate_lines(collected: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Aggregates over evaluation rows",
        "",
        "Every aggregate names how many rows supplied a value. Absent rows are excluded",
        "from the statistic and counted, never treated as zero.",
        "",
        "| metric | coverage | mean | median | min | max |",
        "|---|---|---|---|---|---|",
    ]
    summaries = collected.get("summaries", {})
    for key in collected["metric_keys"]:
        summary = summaries.get(key)
        if not isinstance(summary, Mapping):
            continue
        label = key if key not in ESTIMATOR_LABELS else f"{key} — {ESTIMATOR_LABELS[key]}"
        coverage = f"{summary['n_present']}/{summary['n_rows']} rows"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{label}`",
                    coverage,
                    _cell(summary["mean"]),
                    _cell(summary["median"]),
                    _cell(summary["min"]),
                    _cell(summary["max"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _gate_lines(collected: Mapping[str, Any]) -> list[str]:
    counts: dict[str, dict[str, int]] = {}
    for row in collected["rows"]:
        for gate_row in row["gates"]:
            bucket = counts.setdefault(str(gate_row["name"]), {"pass": 0, "fail": 0, "absent": 0})
            bucket[str(gate_row["status"])] = bucket.get(str(gate_row["status"]), 0) + 1
    lines = ["## Gates", ""]
    if not counts:
        lines += ["No evaluation row produced a gate outcome.", ""]
        return lines
    lines += [
        "`absent` means the gate could not be decided — the metric was not emitted, the",
        "fit was unavailable, or no threshold was declared. It never means it passed.",
        "",
        "| gate | pass | fail | absent |",
        "|---|---|---|---|",
    ]
    for name in sorted(counts):
        bucket = counts[name]
        lines.append(
            f"| `{name}` | {bucket.get('pass', 0)} | {bucket.get('fail', 0)} | "
            f"{bucket.get('absent', 0)} |"
        )
    lines.append("")
    return lines


def _failure_lines(collected: Mapping[str, Any]) -> list[str]:
    failed = [row for row in collected["rows"] if row["status"] != "pass"]
    lines = ["## Failures", ""]
    if not failed:
        lines += ["None.", ""]
        return lines
    for row in failed:
        lines.append(f"- `{row['identity']['row_id']}`")
        for reason in row["reasons"]:
            lines.append(f"  - {reason}")
    lines.append("")
    return lines


def _metric(row: Mapping[str, Any], key: str) -> str:
    return _cell(row["metrics"].get(key, absence.cell(None)))


def _cell(cell: Any) -> str:
    return absence.render(absence.cell_value(cell))


def _short_hash(text: str) -> str:
    if text == absence.ABSENT_TEXT:
        return text
    return f"`{text[:12]}`"


def write_report(collected: Mapping[str, Any], *, results_root: Path, attempt_id: str) -> Path:
    """Write the rendered report for one collected attempt."""

    directory = layout.report_attempt_dir(results_root, attempt_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPORT_FILENAME
    path.write_text(render(collected), encoding="utf-8")
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_REPORT), attempt_id)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse report command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, help="Durable study results root.")
    parser.add_argument(
        "--collect-attempt-id", default=None, help="Collect attempt (defaults to latest)."
    )
    parser.add_argument("--report-attempt-id", default=None, help="Explicit report attempt id.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render one collected attempt."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    collect_attempt_id = layout.resolve_attempt_id(
        results_root, layout.STAGE_COLLECT, args.collect_attempt_id
    )
    collected = read_collected(results_root, collect_attempt_id)
    path = write_report(
        collected,
        results_root=results_root,
        attempt_id=args.report_attempt_id or plan_stage.now_attempt_id(),
    )
    print(f"[he-v1] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
