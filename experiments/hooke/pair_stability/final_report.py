"""Render final pair-stability reports from compact ``08_final_collect`` tables.

This stage is intentionally report-oriented and fast. It consumes only compact
tables written by ``final_collect.py``; it does not read raw final-eval task
records, final-train metrics, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from run_utils import (
    STAGE_FINAL_COLLECT,
    STAGE_FINAL_REPORT,
    attempt_ids,
    new_attempt_id,
    read_json,
    stage_dir,
    write_json,
    write_latest,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

COMPACT_TABLES = (
    "run_index.csv",
    "architecture_summary.csv",
    "energy_by_run.csv",
    "local_energy_histograms.csv",
    "cusp_profile_summary.csv",
    "tail_profile_summary.csv",
    "stratified_summary.csv",
    "hooke_orbital_summary.csv",
    "symmetry_summary.csv",
    "trace_summary.csv",
    "training_curve_summary.csv",
    "resource_summary.csv",
    "failure_modes.csv",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: Sequence[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return math.fsum(clean) / len(clean) if clean else None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def _resolve_collect_attempt_id(results_root: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    latest = stage_dir(results_root, STAGE_FINAL_COLLECT) / "latest.json"
    if latest.is_file():
        payload = read_json(latest)
        if payload.get("attempt_id"):
            return str(payload["attempt_id"])
    attempts = attempt_ids(stage_dir(results_root, STAGE_FINAL_COLLECT))
    if not attempts:
        raise FileNotFoundError(f"no final-collect attempts under {stage_dir(results_root, STAGE_FINAL_COLLECT)}")
    return attempts[-1]


def _load_collect_tables(results_root: Path, collect_attempt_id: str) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    collect_dir = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    if not collect_dir.is_dir():
        raise FileNotFoundError(f"missing final-collect attempt: {collect_dir}")
    return collect_dir, {name: _read_csv(collect_dir / name) for name in COMPACT_TABLES}


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rhu/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_no_data(path: Path, title: str) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _basis_label(row: dict[str, Any]) -> str:
    winner = row.get("winner_kind", "")
    basis = row.get("basis_class", row.get("basis", ""))
    return f"{basis} / {winner}" if winner else str(basis)


def _heatmap_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Return real-scale heatmap cell means for plotting and annotations."""

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get(value_key))
        if value is None:
            continue
        cells[(str(row.get(row_key, "")), str(row.get(col_key, "")))].append(value)
    if not cells:
        return [], [], []
    y_labels = sorted({key[0] for key in cells})
    x_labels = sorted({key[1] for key in cells})
    matrix = []
    for y in y_labels:
        row_values = []
        for x in x_labels:
            row_values.append(_mean(cells.get((y, x), [])))
        matrix.append(row_values)
    return y_labels, x_labels, matrix


def _save_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    transform: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_labels, x_labels, matrix = _heatmap_matrix(rows, row_key=row_key, col_key=col_key, value_key=value_key)
    if not matrix:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    from matplotlib.colors import SymLogNorm

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(x_labels)), max(3.5, 0.8 * len(y_labels))))
    finite_values = [value for row in matrix for value in row if value is not None]
    vmax = max(abs(value) for value in finite_values) if finite_values else 1.0
    data = [[math.nan if value is None else value for value in row] for row in matrix]
    colorbar_label = value_key
    if transform == "signed_log":
        nonzero = [abs(value) for value in finite_values if value != 0.0]
        norm = SymLogNorm(linthresh=min(nonzero), vmin=-vmax, vmax=vmax, base=10) if nonzero else None
        image = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="auto")
        colorbar_label = f"{value_key} (symmetric log color; labels are real scale)"
    else:
        image = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_title(title)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if value is not None:
                ax.text(x_index, y_index, f"{value:.2g}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _energy_variance_points(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return positive log-log points for the 1B energy/stability scatter."""

    points = []
    for row in rows:
        energy_error = _as_float(row.get("energy_error"))
        variance = _as_float(row.get("local_energy_var"))
        if energy_error is None or variance is None:
            continue
        abs_error = abs(energy_error)
        if abs_error <= 0.0 or variance <= 0.0:
            continue
        points.append(
            {
                "abs_energy_error": abs_error,
                "local_energy_var": variance,
                "architecture": str(row.get("basis_class", row.get("architecture", ""))),
                "normalization": str(row.get("normalization", "")),
                "winner_kind": str(row.get("winner_kind", "")),
            }
        )
    return points


def _save_energy_variance_scatter(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = _energy_variance_points(rows)
    if not points:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    from matplotlib.lines import Line2D

    architectures = sorted({str(point["architecture"]) for point in points})
    normalizations = sorted({str(point["normalization"]) for point in points})
    cmap = plt.get_cmap("tab20" if len(architectures) > 10 else "tab10")
    colors = {architecture: cmap(index % cmap.N) for index, architecture in enumerate(architectures)}
    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">", "h", "p"]
    marker_by_norm = {normalization: markers[index % len(markers)] for index, normalization in enumerate(normalizations)}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for point in points:
        ax.scatter(
            point["abs_energy_error"],
            point["local_energy_var"],
            color=colors[str(point["architecture"])],
            marker=marker_by_norm[str(point["normalization"])],
            s=58,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("abs energy error |E - 2|")
    ax.set_ylabel("local-energy variance")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[architecture], markeredgecolor="black", markersize=7, label=architecture)
        for architecture in architectures
    ]
    shape_handles = [
        Line2D([0], [0], marker=marker_by_norm[normalization], color="black", markerfacecolor="lightgray", markeredgecolor="black", linestyle="none", markersize=7, label=normalization)
        for normalization in normalizations
    ]
    architecture_legend = ax.legend(handles=color_handles, title="Architecture", fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.add_artist(architecture_legend)
    ax.legend(handles=shape_handles, title="Normalization", fontsize=7, title_fontsize=8, loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _local_energy_distribution_groups(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Group compact local-energy histograms by normalization and architecture."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalization = str(row.get("normalization", ""))
        architecture = _basis_label(row)
        groups[(normalization, architecture)].append(row)
    normalizations = sorted({key[0] for key in groups})
    architectures = sorted({key[1] for key in groups})
    return normalizations, architectures, groups


def _save_local_energy_distribution_grid(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalizations, architectures, groups = _local_energy_distribution_groups(rows)
    if not groups:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    n_rows = len(normalizations)
    n_cols = len(architectures)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(4.0, 3.2 * n_cols), max(3.0, 2.4 * n_rows)), squeeze=False, sharex=False, sharey=False)
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            values = groups.get((normalization, architecture), [])
            if values:
                centers = [_as_float(row.get("bin_center")) for row in values]
                counts = [_as_float(row.get("count")) for row in values]
                widths = [
                    (right - left) if (left := _as_float(row.get("bin_left"))) is not None and (right := _as_float(row.get("bin_right"))) is not None else 1.0
                    for row in values
                ]
                ax.bar([value for value in centers if value is not None], [value or 0.0 for value in counts], width=widths, align="center", color="#4C78A8", edgecolor="black", alpha=0.85)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\ncount")
            if row_index == n_rows - 1:
                ax.set_xlabel("local_energy")
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_line_plot(path: Path, rows: Sequence[dict[str, Any]], *, x_key: str, y_key: str, group_keys: Sequence[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        label = "/".join(str(row.get(key, "")) for key in group_keys if row.get(key, "") != "")
        groups[label or "all"].append((x, y))
    if not groups:
        _save_no_data(path, title)
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in sorted(groups.items()):
        values = sorted(values)
        ax.plot([point[0] for point in values], [point[1] for point in values], marker="o", label=label)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    if len(groups) <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_bar(path: Path, rows: Sequence[dict[str, Any]], *, label_key: str, value_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [(str(row.get(label_key, "")), _as_float(row.get(value_key))) for row in rows]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        _save_no_data(path, title)
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(values)), 4))
    ax.bar(range(len(values)), [value for _, value in values])
    ax.set_xticks(range(len(values)), [label for label, _ in values], rotation=45, ha="right")
    ax.set_ylabel(value_key)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _with_basis_label(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "basis_label": _basis_label(row)} for row in rows]


def _write_figures(figures_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    architecture = _with_basis_label(tables["architecture_summary.csv"])
    energy = tables["energy_by_run.csv"]
    histograms = tables["local_energy_histograms.csv"]
    stratified = _with_basis_label(tables["stratified_summary.csv"])

    specs = [
        ("1A_real_scale_energy_error_heatmap.png", lambda path: _save_heatmap(path, architecture, row_key="basis_label", col_key="normalization", value_key="energy_error_median", title="Median signed final energy error")),
        ("1A_log_scale_energy_error_heatmap.png", lambda path: _save_heatmap(path, architecture, row_key="basis_label", col_key="normalization", value_key="energy_error_median", title="Median signed final energy error", transform="signed_log")),
        ("1B_energy_error_vs_local_energy_variance.png", lambda path: _save_energy_variance_scatter(path, energy, title="Absolute energy error vs local-energy variance")),
        ("1C_local_energy_distribution_grid.png", lambda path: _save_local_energy_distribution_grid(path, histograms, title="MCMC local-energy histograms")),
        ("2A_cusp_even_slope_by_com.png", lambda path: _save_line_plot(path, tables["cusp_profile_summary.csv"], x_key="r12", y_key="even_slope_median", group_keys=("basis_class", "normalization", "winner_kind", "com_id", "direction_id"), title="Cusp even slope by CoM path")),
        ("2B_cusp_c_minus_1_by_com.png", lambda path: _save_line_plot(path, tables["cusp_profile_summary.csv"], x_key="r12", y_key="c_minus_1_median", group_keys=("basis_class", "normalization", "winner_kind", "com_id", "direction_id"), title="Cusp C_-1 by CoM path")),
        ("2C_cusp_odd_slant_by_com.png", lambda path: _save_line_plot(path, tables["cusp_profile_summary.csv"], x_key="r12", y_key="odd_slant_median", group_keys=("basis_class", "normalization", "winner_kind", "com_id", "direction_id"), title="Cusp odd slant by CoM path")),
        ("3A_tail_local_energy_by_path.png", lambda path: _save_line_plot(path, tables["tail_profile_summary.csv"], x_key="radius", y_key="local_energy_median", group_keys=("basis_class", "normalization", "winner_kind", "com_id", "tail_path"), title="Tail local energy by path")),
        ("3B_tail_logabs_with_reference.png", lambda path: _save_line_plot(path, tables["tail_profile_summary.csv"], x_key="radius", y_key="logabs_median", group_keys=("basis_class", "normalization", "winner_kind", "com_id", "tail_path"), title="Tail logabs by path")),
        ("3C_tail_outlier_heatmap.png", lambda path: _save_heatmap(path, architecture, row_key="basis_label", col_key="normalization", value_key="tail_outlier_fraction_median", title="Tail outlier fraction")),
        ("4_stratified_geometry_aggregate_heatmap.png", lambda path: _save_heatmap(path, [row for row in stratified if row.get("stratum") == "all"], row_key="basis_label", col_key="normalization", value_key="median_abs_energy_error", title="Stratified median absolute energy error")),
        ("5A_hooke_orbital_local_energy_distribution.png", lambda path: _save_line_plot(path, tables["hooke_orbital_summary.csv"], x_key="r12_center", y_key="local_energy_median", group_keys=("basis_class", "normalization", "winner_kind", "com_bin"), title="Hooke-orbital local-energy medians")),
        ("5B_hooke_orbital_local_energy_vs_r12.png", lambda path: _save_line_plot(path, tables["hooke_orbital_summary.csv"], x_key="r12_center", y_key="local_energy_median", group_keys=("basis_class", "normalization", "winner_kind", "com_bin"), title="Hooke-orbital local energy vs r12 by CoM bin")),
        ("5C_hooke_orbital_local_energy_vs_radius.png", lambda path: _save_line_plot(path, tables["hooke_orbital_summary.csv"], x_key="R_norm_center", y_key="local_energy_median", group_keys=("basis_class", "normalization", "winner_kind", "r12_bin"), title="Hooke-orbital local energy vs CoM radius by r12 bin")),
        ("6_symmetry_failure_counts.png", lambda path: _save_bar(path, tables["symmetry_summary.csv"], label_key="symmetry_task", value_key="sign_mismatch_count", title="Symmetry sign mismatch counts")),
        ("7_trace_failure_counts.png", lambda path: _save_bar(path, tables["trace_summary.csv"], label_key="trace_kind", value_key="comparison_error_count", title="Trace comparison error counts")),
        ("8_training_curves.png", lambda path: _save_line_plot(path, tables["training_curve_summary.csv"], x_key="step", y_key="energy_mean", group_keys=("basis_class", "normalization", "winner_kind"), title="Final train energy curves")),
    ]
    for filename, writer in specs:
        writer(figures_dir / filename)
        written.append(filename)

    strata = sorted({str(row.get("stratum", "")) for row in stratified if row.get("stratum", "") not in {"", "all"}})
    for stratum in strata:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stratum)
        filename = f"4_stratified_geometry_{safe}_heatmap.png"
        rows = [row for row in stratified if str(row.get("stratum", "")) == stratum]
        _save_heatmap(figures_dir / filename, rows, row_key="basis_label", col_key="normalization", value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {stratum}")
        written.append(filename)
    return written


def _copy_tables(collect_dir: Path, tables_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in tables.items():
        shutil.copyfile(collect_dir / name, tables_dir / name)
        counts[name] = len(rows)
    return counts


def build_report(
    *,
    results_root: str | Path,
    report_attempt_id: str | None = None,
    final_collect_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Write ``09_final_report`` artifacts from compact collect outputs."""

    results_root = Path(results_root)
    final_collect_attempt_id = _resolve_collect_attempt_id(results_root, final_collect_attempt_id)
    report_attempt_id = report_attempt_id or final_collect_attempt_id or new_attempt_id()
    report_dir = stage_dir(results_root, STAGE_FINAL_REPORT) / report_attempt_id
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    collect_dir, tables = _load_collect_tables(results_root, final_collect_attempt_id)
    table_counts = _copy_tables(collect_dir, tables_dir, tables)
    figures = _write_figures(figures_dir, tables)

    report = {
        "study": "pair_stability",
        "stage": STAGE_FINAL_REPORT,
        "attempt_id": report_attempt_id,
        "final_collect_attempt_id": final_collect_attempt_id,
        "final_collect_dir": str(collect_dir),
        "tables": table_counts,
        "figures": figures,
        "caveats": [
            "final_report.py consumes 08_final_collect compact tables only.",
            "Raw final-eval and final-train artifacts are reduced by final_collect.py.",
            "Runtime/resource summaries are reported separately from model-quality ranking.",
        ],
    }
    write_json(report_dir / "final_report.json", report)
    (report_dir / "report.md").write_text(_report_markdown(report, tables))
    write_latest(stage_dir(results_root, STAGE_FINAL_REPORT), report_attempt_id)
    return {"attempt_dir": str(report_dir), "report": report}


def _report_markdown(report: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    architecture = sorted(tables["architecture_summary.csv"], key=lambda row: (row.get("basis_class", ""), row.get("normalization", ""), row.get("winner_kind", "")))
    lines = [
        "# Hooke Pair-Stability Final Report",
        "",
        "## Scope And Provenance",
        "",
        "This report consumes `08_final_collect` compact tables only. It does not parse raw model, train, or eval records.",
        f"Final collect attempt: `{report['final_collect_attempt_id']}`.",
        "",
        "## Final Champion Summary",
        "",
        "See `tables/run_index.csv` and `tables/energy_by_run.csv`.",
        "",
        "## Family-Level Ranking",
        "",
    ]
    if architecture:
        lines.extend(["| basis_class | normalization | winner_kind | n_success/n_expected | energy_error_median | local_energy_var_median |", "|---|---|---|---:|---:|---:|"])
        for row in architecture[:20]:
            lines.append(
                f"| {row.get('basis_class', '')} | {row.get('normalization', '')} | {row.get('winner_kind', '')} | "
                f"{row.get('n_success', '')}/{row.get('n_expected', '')} | {row.get('energy_error_median', '')} | {row.get('local_energy_var_median', '')} |"
            )
    else:
        lines.append("No architecture summary rows were found.")
    lines.extend(
        [
            "",
            "## Energy And Local-Energy Results",
            "",
            "Energy figures use signed error relative to exact Hooke energy `E = 2`; distribution plots use collected histograms.",
            "",
            "## Cusp Diagnostics",
            "",
            "Cusp tables preserve center-of-mass and direction columns when present.",
            "",
            "## Tail Diagnostics",
            "",
            "Tail tables preserve path columns. Exact log-amplitude references are included when collect inputs provide them.",
            "",
            "## Stratified Geometry Diagnostics",
            "",
            "Stratified summaries include per-stratum rows and `stratum=all` aggregate rows.",
            "",
            "## Hooke-Orbital Diagnostics",
            "",
            "Hooke-orbital summaries are binned by CoM-radius and `r12` bins.",
            "",
            "## Symmetry Diagnostics",
            "",
            "See `tables/symmetry_summary.csv` and the symmetry figures.",
            "",
            "## Trace Diagnostics",
            "",
            "See `tables/trace_summary.csv` and the trace figures.",
            "",
            "## Training And Resource Summary",
            "",
            "See `tables/training_curve_summary.csv` and `tables/resource_summary.csv`. Runtime is not mixed into quality ranking.",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.extend(["", "## Next-Scan Implications", "", "Use energy, local-energy variance, pathology, cusp/tail, symmetry, and trace summaries jointly. Keep energy and stability winners separate.", "", "## Tables And Figures", "", "Tables:"])
    for name, n_rows in report["tables"].items():
        lines.append(f"- `tables/{name}`: {n_rows} rows")
    lines.append("")
    lines.append("Figures:")
    for name in report["figures"]:
        lines.append(f"- `figures/{name}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-report arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-collect-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write final report artifacts."""

    args = parse_args(argv)
    result = build_report(
        results_root=args.results_root,
        report_attempt_id=args.attempt_id,
        final_collect_attempt_id=args.final_collect_attempt_id,
    )
    report = result["report"]
    print(
        f"[pair_stability] final report consumed 08_final_collect/{report['final_collect_attempt_id']} "
        f"-> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
