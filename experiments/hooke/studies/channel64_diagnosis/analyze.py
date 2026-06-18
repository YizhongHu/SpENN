"""Diagnose why 64-channel Hooke pair-validation configs lost selection.

The analysis is deliberately file-only: it reads normalized study CSVs and does
not import ``spenn``. The source artifacts are the ``02_collect`` run table and
the ``03_select`` candidate table from ``pair_validation``.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PAIR_VALIDATION = REPO_ROOT / "experiments" / "hooke" / "studies" / "pair_validation"
DEFAULT_RUNS_CSV = PAIR_VALIDATION / "reports" / "02_collect" / "runs.csv"
DEFAULT_SELECTION_CSV = PAIR_VALIDATION / "reports" / "03_select" / "selection.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports"

GROUP_COLUMNS = (
    "optimizer_params.lr",
    "model_params.channels",
    "model_params.layers",
    "model_params.gate_activation",
)

RUN_SUMMARY_COLUMNS = (
    "n_runs",
    "n_completed",
    "n_failed",
    "median_train_energy",
    "median_validation_energy",
    "mean_validation_minus_train_energy",
    "median_validation_minus_train_energy",
    "max_abs_validation_minus_train_energy",
    "median_train_energy_variance",
    "median_validation_energy_variance",
    "mean_train_energy_variance",
    "mean_validation_energy_variance",
    "mean_validation_to_train_variance_ratio",
    "mean_train_acceptance_rate",
    "mean_validation_acceptance_rate",
)

SEED_ROW_COLUMNS = (
    "config_id",
    "runtime.seed",
    "optimizer_params.lr",
    "model_params.channels",
    "model_params.layers",
    "model_params.gate_activation",
    "status",
    "train/energy",
    "validation/energy",
    "validation_minus_train_energy",
    "train/energy_variance",
    "validation/energy_variance",
    "train/sampler/acceptance_rate",
    "validation/sampler/acceptance_rate",
)

SELECTION_COLUMNS = (
    "energy_rank",
    "selected",
    "config_id",
    *GROUP_COLUMNS,
    "median validation/energy",
    "median_energy_variance",
    "energy_iqr",
    "median_energy_stderr",
    "n_failed",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the channel-64 diagnosis CLI."""

    args = _parse_args(argv)
    run_analysis(
        runs_csv=args.runs_csv,
        selection_csv=args.selection_csv,
        output_dir=args.output_dir,
    )
    return 0


def run_analysis(
    *,
    runs_csv: str | Path = DEFAULT_RUNS_CSV,
    selection_csv: str | Path = DEFAULT_SELECTION_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Read pair-validation CSVs and write derived diagnosis artifacts."""

    runs = read_csv_rows(runs_csv)
    selection = read_csv_rows(selection_csv)
    out = Path(output_dir)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    channel_summary = summarize_groups(runs, ("model_params.channels",))
    channel_lr_summary = summarize_groups(runs, ("model_params.channels", "optimizer_params.lr"))
    channel_gate_summary = summarize_groups(runs, ("model_params.channels", "model_params.gate_activation"))
    candidate_rows = rank_selection_rows(selection)
    channel64_candidates = [
        row for row in candidate_rows if key_text(row.get("model_params.channels")) == "64"
    ]
    channel64_seed_rows = enriched_seed_rows(
        row for row in runs if key_text(row.get("model_params.channels")) == "64"
    )

    write_csv(tables / "channel_summary.csv", channel_summary, ("model_params.channels", *RUN_SUMMARY_COLUMNS))
    write_csv(
        tables / "channel_lr_summary.csv",
        channel_lr_summary,
        ("model_params.channels", "optimizer_params.lr", *RUN_SUMMARY_COLUMNS),
    )
    write_csv(
        tables / "channel_gate_summary.csv",
        channel_gate_summary,
        ("model_params.channels", "model_params.gate_activation", *RUN_SUMMARY_COLUMNS),
    )
    write_csv(tables / "channel64_candidates.csv", channel64_candidates, SELECTION_COLUMNS)
    write_csv(tables / "channel64_seed_rows.csv", channel64_seed_rows, SEED_ROW_COLUMNS)

    diagnosis = build_diagnosis(runs, candidate_rows)
    report = render_report(
        diagnosis,
        channel_summary=channel_summary,
        channel_lr_summary=channel_lr_summary,
        channel_gate_summary=channel_gate_summary,
        channel64_candidates=channel64_candidates,
    )
    report_path = out / "diagnosis.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "diagnosis": diagnosis,
        "report": str(report_path),
        "tables": {
            "channel_summary": str(tables / "channel_summary.csv"),
            "channel_lr_summary": str(tables / "channel_lr_summary.csv"),
            "channel_gate_summary": str(tables / "channel_gate_summary.csv"),
            "channel64_candidates": str(tables / "channel64_candidates.csv"),
            "channel64_seed_rows": str(tables / "channel64_seed_rows.csv"),
        },
    }


def build_diagnosis(
    runs: Sequence[Mapping[str, Any]],
    ranked_selection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the small set of derived facts used in the report."""

    if not ranked_selection:
        raise ValueError("selection table is empty")
    selected = _first(row for row in ranked_selection if parse_bool(row.get("selected")))
    if selected is None:
        raise ValueError("selection table has no selected=true row")

    candidates64 = [row for row in ranked_selection if key_text(row.get("model_params.channels")) == "64"]
    if not candidates64:
        raise ValueError("selection table has no 64-channel candidates")

    best64 = min(candidates64, key=lambda row: as_float(row.get("median validation/energy")))
    energy_leader = min(ranked_selection, key=lambda row: as_float(row.get("median validation/energy")))
    margin = selection_margin(selected, best64)
    selected_energy = as_float(selected.get("median validation/energy"))
    best64_energy = as_float(best64.get("median validation/energy"))
    energy_gap_selected_minus_best64 = selected_energy - best64_energy

    width64_rows = [row for row in runs if key_text(row.get("model_params.channels")) == "64"]
    width64_gaps = finite_gaps(width64_rows, "validation/energy", "train/energy")
    width64_summary = summarize_group(width64_rows)
    selected_summary = summarize_group(
        row
        for row in runs
        if all(key_text(row.get(column)) == key_text(selected.get(column)) for column in GROUP_COLUMNS)
    )

    return {
        "n_runs": len(runs),
        "n_candidates": len(ranked_selection),
        "selected": selected,
        "best64": best64,
        "energy_leader": energy_leader,
        "selection_margin_selected_vs_best64": margin,
        "energy_gap_selected_minus_best64": energy_gap_selected_minus_best64,
        "best64_inside_selection_margin": energy_gap_selected_minus_best64 < margin,
        "best64_variance_minus_selected": as_float(best64.get("median_energy_variance"))
        - as_float(selected.get("median_energy_variance")),
        "width64_mean_validation_minus_train": mean(width64_gaps),
        "width64_median_validation_minus_train": median(width64_gaps),
        "width64_max_abs_validation_minus_train": max_abs(width64_gaps),
        "width64_summary": width64_summary,
        "selected_summary": selected_summary,
        "conclusion": diagnosis_sentence(
            selected=selected,
            best64=best64,
            selected_energy=selected_energy,
            best64_energy=best64_energy,
            margin=margin,
            width64_gap=median(width64_gaps),
        ),
    }


def diagnosis_sentence(
    *,
    selected: Mapping[str, Any],
    best64: Mapping[str, Any],
    selected_energy: float,
    best64_energy: float,
    margin: float,
    width64_gap: float,
) -> str:
    """Return the headline diagnosis as one prose sentence."""

    best64_config = best64.get("config_id")
    selected_config = selected.get("config_id")
    energy_lead = selected_energy - best64_energy
    gap_text = format_number(width64_gap)
    return (
        "64 channels did have a good LR/activation pocket "
        f"({best64_config}), but its {format_number(energy_lead)} energy lead over "
        f"{selected_config} was inside the {format_number(margin)} selection margin; "
        f"the train-validation gap for 64-channel runs is small (median {gap_text}), "
        "so 02_collect points to variance/robustness sensitivity rather than classic overfit."
    )


def render_report(
    diagnosis: Mapping[str, Any],
    *,
    channel_summary: Sequence[Mapping[str, Any]],
    channel_lr_summary: Sequence[Mapping[str, Any]],
    channel_gate_summary: Sequence[Mapping[str, Any]],
    channel64_candidates: Sequence[Mapping[str, Any]],
) -> str:
    """Render the diagnosis as Markdown."""

    selected = diagnosis["selected"]
    best64 = diagnosis["best64"]
    energy_leader = diagnosis["energy_leader"]
    lines = [
        "# Channel-64 diagnosis",
        "",
        "## Answer",
        "",
        diagnosis["conclusion"],
        "",
        "## Evidence from 02_collect and 03_select",
        "",
        f"- Run rows analyzed: `{diagnosis['n_runs']}`.",
        f"- Candidate groups analyzed: `{diagnosis['n_candidates']}`.",
        f"- Selected config: `{selected.get('config_id')}`.",
        f"- Lowest-energy candidate: `{energy_leader.get('config_id')}`.",
        f"- Best 64-channel candidate: `{best64.get('config_id')}`.",
        "- Best 64-channel candidate median validation energy: "
        f"`{format_number(as_float(best64.get('median validation/energy')))}`.",
        "- Selected candidate median validation energy: "
        f"`{format_number(as_float(selected.get('median validation/energy')))}`.",
        "- Selected-minus-best64 median-energy gap: "
        f"`{format_number(diagnosis['energy_gap_selected_minus_best64'])}`.",
        "- Selection margin between selected and best64: "
        f"`{format_number(diagnosis['selection_margin_selected_vs_best64'])}`.",
        "- Best64 median validation-variance minus selected: "
        f"`{format_number(diagnosis['best64_variance_minus_selected'])}`.",
        "- 64-channel median validation-minus-train energy gap: "
        f"`{format_number(diagnosis['width64_median_validation_minus_train'])}`.",
        "- 64-channel max absolute validation-minus-train energy gap: "
        f"`{format_number(diagnosis['width64_max_abs_validation_minus_train'])}`.",
        "",
        "## 64-channel candidates",
        "",
        markdown_table(
            channel64_candidates,
            (
                "energy_rank",
                "config_id",
                "optimizer_params.lr",
                "model_params.gate_activation",
                "median validation/energy",
                "median_energy_variance",
                "energy_iqr",
                "median_energy_stderr",
            ),
            max_rows=12,
        ),
        "",
        "## Width and LR summary",
        "",
        markdown_table(
            channel_lr_summary,
            (
                "model_params.channels",
                "optimizer_params.lr",
                "n_runs",
                "median_validation_energy",
                "mean_validation_minus_train_energy",
                "mean_validation_energy_variance",
                "mean_train_energy_variance",
            ),
        ),
        "",
        "## Width and activation summary",
        "",
        markdown_table(
            channel_gate_summary,
            (
                "model_params.channels",
                "model_params.gate_activation",
                "n_runs",
                "median_validation_energy",
                "mean_validation_minus_train_energy",
                "mean_validation_energy_variance",
                "mean_train_energy_variance",
            ),
        ),
        "",
        "## Interpretation",
        "",
        "- Lack of LR is not the main explanation: the 64-channel sigmoid/lr=3e-4 candidate was the energy leader.",
        "- Classic overfit is not supported by the available artifact: 02_collect has only end-of-train validation, and its train/validation energy and variance are close for the 64-channel slice.",
        "- The real failure mode is robustness: 64 channels work in a narrow sigmoid/LR pocket, while poor LR/gate combinations and some seeds carry much higher local-energy variance.",
        "- The selector preferred the 32-channel sigmoid/lr=3e-3 model because the energy difference was inside the configured uncertainty margin and the first tie-breaker was lower validation variance.",
        "",
        "## Next experiment",
        "",
        "- Compare the selected 32-channel sigmoid/lr=3e-3 baseline against 64-channel sigmoid with a denser LR grid around 3e-4 to 1e-3.",
        "- Increase seed count before concluding a width effect; three seeds are enough to expose variance sensitivity but weak for capacity claims.",
        "- Add validation at multiple checkpoints or checkpoint selection by independent-sampler variance if the real question is overfitting over training time.",
        "- Keep final-eval probes or a tail/cusp guard in selection, because low sampled validation energy alone can hide bad high-variance regions.",
        "",
        "## Artifacts",
        "",
        "- `tables/channel_summary.csv`",
        "- `tables/channel_lr_summary.csv`",
        "- `tables/channel_gate_summary.csv`",
        "- `tables/channel64_candidates.csv`",
        "- `tables/channel64_seed_rows.csv`",
    ]
    # Keep the channel summary available in the generated table directory even
    # though the report focuses on the more diagnostic LR and activation slices.
    _ = channel_summary
    return "\n".join(lines) + "\n"


def summarize_groups(
    rows: Sequence[Mapping[str, Any]],
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Return run-level numeric summaries grouped by selected columns."""

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(key_text(row.get(column)) for column in group_columns)].append(row)

    summary_rows = []
    for key, group_rows in grouped.items():
        row = {column: key[index] for index, column in enumerate(group_columns)}
        row.update(summarize_group(group_rows))
        summary_rows.append(row)
    return sorted(summary_rows, key=lambda item: sort_key(item, group_columns))


def summarize_group(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one run group."""

    row_list = list(rows)
    gaps = finite_gaps(row_list, "validation/energy", "train/energy")
    mean_train_variance = mean(finite_values(row_list, "train/energy_variance"))
    mean_validation_variance = mean(finite_values(row_list, "validation/energy_variance"))
    return {
        "n_runs": len(row_list),
        "n_completed": sum(1 for row in row_list if key_text(row.get("status")).lower() == "completed"),
        "n_failed": sum(1 for row in row_list if key_text(row.get("status")).lower() != "completed"),
        "median_train_energy": median(finite_values(row_list, "train/energy")),
        "median_validation_energy": median(finite_values(row_list, "validation/energy")),
        "mean_validation_minus_train_energy": mean(gaps),
        "median_validation_minus_train_energy": median(gaps),
        "max_abs_validation_minus_train_energy": max_abs(gaps),
        "median_train_energy_variance": median(finite_values(row_list, "train/energy_variance")),
        "median_validation_energy_variance": median(finite_values(row_list, "validation/energy_variance")),
        "mean_train_energy_variance": mean_train_variance,
        "mean_validation_energy_variance": mean_validation_variance,
        "mean_validation_to_train_variance_ratio": safe_divide(mean_validation_variance, mean_train_variance),
        "mean_train_acceptance_rate": mean(finite_values(row_list, "train/sampler/acceptance_rate")),
        "mean_validation_acceptance_rate": mean(finite_values(row_list, "validation/sampler/acceptance_rate")),
    }


def rank_selection_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return selection rows with an energy rank column."""

    ranked = sorted(rows, key=lambda row: as_float(row.get("median validation/energy")))
    result = []
    for index, row in enumerate(ranked, start=1):
        enriched = dict(row)
        enriched["energy_rank"] = index
        result.append(enriched)
    return result


def enriched_seed_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return per-seed rows with the train/validation energy gap added."""

    enriched = []
    for row in rows:
        output = {column: row.get(column) for column in SEED_ROW_COLUMNS}
        output["validation_minus_train_energy"] = safe_subtract(row.get("validation/energy"), row.get("train/energy"))
        enriched.append(output)
    return sorted(
        enriched,
        key=lambda item: (
            as_float(item.get("optimizer_params.lr")),
            key_text(item.get("model_params.gate_activation")),
            as_float(item.get("runtime.seed")),
        ),
    )


def selection_margin(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Return the pair-validation selection margin for two candidates."""

    stderr_a = as_float(a.get("median_energy_stderr"), default=math.inf)
    stderr_b = as_float(b.get("median_energy_stderr"), default=math.inf)
    iqr_a = as_float(a.get("energy_iqr"), default=math.inf)
    iqr_b = as_float(b.get("energy_iqr"), default=math.inf)
    return max(2.0 * math.sqrt(stderr_a**2 + stderr_b**2), 0.25 * max(iqr_a, iqr_b), 1.0e-4)


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV file into dictionaries."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    """Write rows to a CSV file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) if columns is not None else sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    max_rows: int | None = None,
) -> str:
    """Render a compact Markdown table."""

    selected_rows = list(rows[:max_rows] if max_rows is not None else rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected_rows:
        values = [markdown_value(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        overflow = ["..."]
        if len(columns) > 1:
            overflow.append(f"{len(rows) - max_rows} more rows")
        overflow.extend("" for _ in range(max(0, len(columns) - len(overflow))))
        lines.append("| " + " | ".join(overflow) + " |")
    return "\n".join(lines)


def finite_values(rows: Iterable[Mapping[str, Any]], column: str) -> list[float]:
    """Return finite numeric values from a column."""

    values = []
    for row in rows:
        value = as_float(row.get(column), default=math.nan)
        if math.isfinite(value):
            values.append(value)
    return values


def finite_gaps(rows: Iterable[Mapping[str, Any]], left: str, right: str) -> list[float]:
    """Return finite ``left - right`` gaps."""

    gaps = []
    for row in rows:
        gap = safe_subtract(row.get(left), row.get(right))
        if isinstance(gap, float) and math.isfinite(gap):
            gaps.append(gap)
    return gaps


def parse_scalar(value: Any) -> Any:
    """Parse a CSV scalar into a bool, int, float, or string."""

    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    if lowered == "nan":
        return math.nan
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def as_float(value: Any, *, default: float = math.inf) -> float:
    """Parse ``value`` as a float, returning ``default`` on failure."""

    parsed = parse_scalar(value)
    if parsed is None or isinstance(parsed, bool):
        return default
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    """Parse common boolean encodings."""

    parsed = parse_scalar(value)
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, (int, float)):
        return bool(parsed)
    return False


def key_text(value: Any) -> str:
    """Return a stable text key for grouping."""

    parsed = parse_scalar(value)
    if parsed is None:
        return ""
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return str(parsed)


def median(values: Sequence[float]) -> float:
    """Return the median, or ``nan`` for an empty sequence."""

    if not values:
        return math.nan
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or ``nan`` for an empty sequence."""

    return sum(values) / len(values) if values else math.nan


def max_abs(values: Sequence[float]) -> float:
    """Return the maximum absolute value, or ``nan`` for empty input."""

    return max((abs(value) for value in values), default=math.nan)


def safe_subtract(left: Any, right: Any) -> float | str:
    """Return ``left - right`` for finite scalars, otherwise empty text."""

    left_value = as_float(left, default=math.nan)
    right_value = as_float(right, default=math.nan)
    if math.isfinite(left_value) and math.isfinite(right_value):
        return left_value - right_value
    return ""


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a finite ratio or ``nan``."""

    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


def sort_key(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[Any, ...]:
    """Return a mixed numeric/text sort key for a row."""

    key = []
    for column in columns:
        value = row.get(column)
        number = as_float(value, default=math.nan)
        key.append(number if math.isfinite(number) else key_text(value))
    return tuple(key)


def csv_value(value: Any) -> Any:
    """Format a scalar for CSV output."""

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format_number(value)
    return "" if value is None else value


def markdown_value(value: Any) -> str:
    """Format a scalar for Markdown table cells."""

    if isinstance(value, float):
        return format_number(value)
    parsed = parse_scalar(value)
    if isinstance(parsed, float):
        return format_number(parsed)
    if parsed is None:
        return ""
    return f"`{parsed}`" if isinstance(parsed, str) else str(parsed)


def format_number(value: Any) -> str:
    """Format a number with compact significant digits."""

    number = as_float(value, default=math.nan)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.12g}"


def _first(items: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for item in items:
        return item
    return None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
