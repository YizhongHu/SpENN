#!/usr/bin/env python
"""Apply the declared selection rule to the collected validation-scan table.

Reads ``manifest.yaml`` and ``runs.csv`` (from collect.py) and writes
``selection.csv``, ``selected_config.yaml``, and ``selection_report.md``.

The selection rule is reproducible from local outputs only: the selector never
reads W&B and never uses the exact reference energy. Geometry diagnostics are
reported and flagged but never decide the winner (the manifest declares the
eligibility criteria and tie-breakers; geometry fields are in neither).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import yaml

from collect import group_keys, load_manifest  # same-directory study module


def read_runs_csv(path: Path) -> list[dict[str, str]]:
    """Read the collected run table as raw string cells."""

    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(cell: str | None) -> float | None:
    if cell is None or cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _parse_bool(cell: str | None) -> bool | None:
    if cell is None or cell == "":
        return None
    lowered = cell.strip().lower()
    if lowered in ("true", "1"):
        return True
    if lowered in ("false", "0"):
        return False
    return None


def run_is_eligible(row: dict[str, str], manifest: dict) -> tuple[bool, list[str]]:
    """Check one completed run against the manifest eligibility rules."""

    reasons: list[str] = []
    eligibility = manifest.get("eligibility", {})
    for check in eligibility.get("require", ()):
        if _parse_bool(row.get(check)) is not True:
            reasons.append(f"{check} is not true")
    minimum_fraction = eligibility.get("local_energy_finite_fraction")
    if minimum_fraction is not None:
        fraction = _parse_float(row.get("validation/local_energy_finite_fraction"))
        if fraction is None or fraction < float(minimum_fraction):
            reasons.append(
                f"validation/local_energy_finite_fraction={fraction} < {minimum_fraction}"
            )
    return (not reasons, reasons)


def run_energy(row: dict[str, str], manifest: dict) -> float:
    """Return the run's selection energy; failed/ineligible runs are +inf."""

    failed_value = float(manifest["validation"].get("failed_run_value", math.inf))
    if row.get("status") != "completed":
        return failed_value
    eligible, _ = run_is_eligible(row, manifest)
    if not eligible:
        return failed_value
    energy = _parse_float(row.get(str(manifest["validation"]["metric"])))
    if energy is None or not math.isfinite(energy):
        return failed_value
    return energy


def _group_rows(rows: list[dict[str, str]], manifest: dict) -> dict[tuple[str, ...], list[dict[str, str]]]:
    keys = group_keys(manifest)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return groups


def _aggregate(values: list[float], how: str) -> float:
    if not values:
        return math.inf
    if how == "median":
        return median(values)
    if how == "mean":
        return sum(values) / len(values)
    if how == "min":
        return min(values)
    raise ValueError(f"unsupported aggregate {how!r}")


def _median_metric(rows: list[dict[str, str]], metric: str) -> float:
    """Median of a metric over rows that logged a finite value (else +inf)."""

    values = [
        value
        for row in rows
        if (value := _parse_float(row.get(metric))) is not None and math.isfinite(value)
    ]
    return median(values) if values else math.inf


def seed_energy_spread(energies: list[float]) -> float:
    """Max-min spread over finite seed energies; +inf below two finite seeds."""

    finite = [energy for energy in energies if math.isfinite(energy)]
    if len(finite) < 2:
        return math.inf
    return max(finite) - min(finite)


def score_group(
    group_values: tuple[str, ...], rows: list[dict[str, str]], manifest: dict
) -> dict[str, object]:
    """Score one config group: aggregate energy plus tie-breaker values."""

    seed_key = str(manifest["seed_key"])
    expected_seeds = [str(seed) for seed in manifest["grid"][seed_key]]
    by_seed = {str(row.get(seed_key, "")): row for row in rows}

    # Every expected seed contributes; missing seeds count as failed (+inf).
    energies = {
        seed: run_energy(by_seed[seed], manifest) if seed in by_seed else math.inf
        for seed in expected_seeds
    }
    # Seeds beyond the manifest grid still count if present.
    for seed, row in by_seed.items():
        if seed not in energies:
            energies[seed] = run_energy(row, manifest)

    aggregate = str(manifest["validation"].get("aggregate", "median"))
    score = _aggregate(list(energies.values()), aggregate)
    eligible_rows = [row for row in rows if math.isfinite(run_energy(row, manifest))]

    keys = group_keys(manifest)
    group = dict(zip(keys, group_values))
    tie_breaker_values: list[float] = []
    for tie_breaker in manifest.get("tie_breakers", ()):
        if tie_breaker == "seed_energy_spread":
            tie_breaker_values.append(seed_energy_spread(list(energies.values())))
        elif tie_breaker in keys:
            value = _parse_float(group.get(tie_breaker))
            tie_breaker_values.append(value if value is not None else math.inf)
        else:
            tie_breaker_values.append(_median_metric(eligible_rows, str(tie_breaker)))

    config_id = rows[0].get("config_id", "") if rows else ""
    return {
        **group,
        "config_id": config_id,
        "score": score,
        "tie_breakers": tie_breaker_values,
        "seed_energies": energies,
        "n_seeds_expected": len(expected_seeds),
        "n_failed_seeds": sum(1 for energy in energies.values() if not math.isfinite(energy)),
        "rows": rows,
    }


def rank_groups(rows: list[dict[str, str]], manifest: dict) -> list[dict[str, object]]:
    """Rank config groups by aggregate energy, then manifest tie-breakers."""

    scored = [
        score_group(values, group_rows, manifest)
        for values, group_rows in _group_rows(rows, manifest).items()
    ]
    scored.sort(key=lambda group: (group["score"], *group["tie_breakers"], group["config_id"]))
    return scored


def geometry_flags(rows: list[dict[str, str]], manifest: dict) -> list[str]:
    """Flag suspicious sampler geometry; informational only, never selective."""

    fields = list(manifest.get("diagnostic_fields", {}).get("sampler_geometry", ()))
    q01_min = float(manifest.get("geometry_flags", {}).get("electron_distance_q01_min", 0.0))
    flags: list[str] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        label = f"{row.get('config_id')} seed={row.get(str(manifest['seed_key']), '?')}"
        n_electrons = _parse_float(row.get("validation/sampler/n_electrons"))
        for field in fields:
            value = _parse_float(row.get(field))
            if value is None:
                if field.startswith("validation/sampler/electron_distance") and (
                    n_electrons is None or n_electrons < 2
                ):
                    continue  # pair metrics are legitimately absent below N=2
                flags.append(f"{label}: {field} missing")
            elif not math.isfinite(value):
                flags.append(f"{label}: {field} nonfinite ({value})")
        q01 = _parse_float(row.get("validation/sampler/electron_distance_q01"))
        if q01 is not None and math.isfinite(q01) and q01 < q01_min:
            flags.append(
                f"{label}: validation/sampler/electron_distance_q01={q01:g} below {q01_min:g} "
                "(near-coalescence tail)"
            )
    return flags


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


def write_selection_csv(ranked: list[dict[str, object]], manifest: dict, path: Path) -> None:
    keys = group_keys(manifest)
    tie_breaker_names = [str(name) for name in manifest.get("tie_breakers", ())]
    columns = [
        "rank",
        "config_id",
        *keys,
        "score",
        *[f"tie_breaker/{name}" for name in tie_breaker_names],
        "n_seeds_expected",
        "n_failed_seeds",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for rank, group in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    group["config_id"],
                    *[group.get(key, "") for key in keys],
                    _fmt(float(group["score"])),
                    *[_fmt(value) for value in group["tie_breakers"]],
                    group["n_seeds_expected"],
                    group["n_failed_seeds"],
                ]
            )


def write_selected_config(winner: dict[str, object], manifest: dict, path: Path) -> None:
    keys = group_keys(manifest)
    overrides = [f"{key}={winner.get(key)}" for key in keys]
    payload = {
        "study": manifest["study"]["name"],
        "train_config": manifest.get("train_config"),
        "selected": {
            "config_id": winner["config_id"],
            **{key: winner.get(key) for key in keys},
        },
        "overrides": overrides,
        "selection": {
            "metric": manifest["validation"]["metric"],
            "aggregate": manifest["validation"].get("aggregate", "median"),
            "score": float(winner["score"]),
            "seed_energies": {
                seed: ("inf" if not math.isfinite(energy) else energy)
                for seed, energy in winner["seed_energies"].items()
            },
            "n_failed_seeds": winner["n_failed_seeds"],
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_report(
    ranked: list[dict[str, object]], flags: list[str], manifest: dict, path: Path
) -> None:
    keys = group_keys(manifest)
    geometry_fields = list(manifest.get("diagnostic_fields", {}).get("sampler_geometry", ()))
    lines = [
        f"# Selection report: {manifest['study']['name']}",
        "",
        f"- selection metric: `{manifest['validation']['metric']}`"
        f" ({manifest['validation'].get('aggregate', 'median')} over seeds)",
        f"- failed/ineligible seeds count as `{manifest['validation'].get('failed_run_value', 'inf')}`",
        f"- tie-breakers: {', '.join(f'`{t}`' for t in manifest.get('tie_breakers', ()))}",
        "- inputs: local run outputs only (no W&B, no exact reference energy)",
        "",
        "## Ranking",
        "",
        "| rank | config_id | " + " | ".join(keys) + " | score | failed seeds |",
        "|---|---|" + "---|" * (len(keys) + 2),
    ]
    for rank, group in enumerate(ranked, start=1):
        cells = [str(rank), str(group["config_id"]), *[str(group.get(key, "")) for key in keys]]
        cells += [_fmt(float(group["score"])), f"{group['n_failed_seeds']}/{group['n_seeds_expected']}"]
        lines.append("| " + " | ".join(cells) + " |")

    if ranked:
        winner = ranked[0]
        lines += [
            "",
            "## Selected",
            "",
            f"`{winner['config_id']}` with {manifest['validation'].get('aggregate', 'median')} "
            f"`{manifest['validation']['metric']}` = {_fmt(float(winner['score']))}",
            "",
            "Seed energies: "
            + ", ".join(f"{seed}: {_fmt(energy)}" for seed, energy in winner["seed_energies"].items()),
        ]

    lines += ["", "## Sampler geometry diagnostics", ""]
    if geometry_fields and ranked:
        winner_rows = ranked[0]["rows"]
        lines += [
            "Winner geometry (informational; geometry never decides selection):",
            "",
            "| seed | " + " | ".join(field.rsplit("/", 1)[-1] for field in geometry_fields) + " |",
            "|---|" + "---|" * len(geometry_fields),
        ]
        for row in winner_rows:
            cells = [str(row.get(str(manifest["seed_key"]), "?"))]
            for field in geometry_fields:
                cells.append(row.get(field, "") or "-")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if flags:
        lines += ["### Flags", ""]
        lines += [f"- {flag}" for flag in flags]
    else:
        lines.append("No suspicious walker geometry flagged.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "manifest.yaml",
        help="Study manifest path.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "runs.csv",
        help="Collected run table from collect.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for selection.csv, selected_config.yaml, selection_report.md.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    rows = read_runs_csv(args.runs)
    if not rows:
        print(f"no runs found in {args.runs}")
        return 1

    ranked = rank_groups(rows, manifest)
    flags = geometry_flags(rows, manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_selection_csv(ranked, manifest, args.output_dir / "selection.csv")
    write_selected_config(ranked[0], manifest, args.output_dir / "selected_config.yaml")
    write_report(ranked, flags, manifest, args.output_dir / "selection_report.md")

    winner = ranked[0]
    print(
        json.dumps(
            {
                "selected": winner["config_id"],
                "score": "inf" if not math.isfinite(float(winner["score"])) else float(winner["score"]),
                "groups": len(ranked),
                "geometry_flags": len(flags),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
