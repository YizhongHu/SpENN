#!/usr/bin/env python
"""Apply the declared selection rule to the collected validation-scan table.

Reads ``manifest.yaml`` and ``runs.csv`` (from collect.py) and writes
``selection.csv``, ``selected_config.yaml``, and ``selection_report.md``.

The selection rule is reproducible from local outputs only: the selector never
reads W&B and never uses the exact reference energy.

Selection is margin-aware: a config only clearly beats another when its
aggregate validation energy is lower by more than the manifest-declared
selection margin (a function of estimator stderr, seed-to-seed IQR, and an
absolute floor). Candidates within the margin are tied and ranked by the
manifest tie-breakers, in order. Geometry diagnostics are reported and flagged
but only decide via the explicit ``geometry_warning_count`` tie-breaker.
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


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of pre-sorted values (numpy default)."""

    if not sorted_values:
        return math.inf
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def seed_energy_iqr(energies: list[float]) -> float:
    """Seed-to-seed interquartile range over finite seed energies.

    Below two finite seeds there is no spread estimate; returns 0.0 so the
    margin is then driven by stderr and the absolute floor alone (with
    ``selection.require_all_seeds`` such groups score +inf anyway).
    """

    finite = sorted(energy for energy in energies if math.isfinite(energy))
    if len(finite) < 2:
        return 0.0
    return _quantile(finite, 0.75) - _quantile(finite, 0.25)


def score_group(
    group_values: tuple[str, ...], rows: list[dict[str, str]], manifest: dict
) -> dict[str, object]:
    """Score one config group: aggregate energy plus margin/tie-breaker stats."""

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

    selection = manifest.get("selection", {})
    n_failed = sum(1 for energy in energies.values() if not math.isfinite(energy))
    aggregate = str(manifest["validation"].get("aggregate", "median"))
    if n_failed and selection.get("require_all_seeds", False):
        score = math.inf
    else:
        score = _aggregate(list(energies.values()), aggregate)
    eligible_rows = [row for row in rows if math.isfinite(run_energy(row, manifest))]

    keys = group_keys(manifest)
    group = dict(zip(keys, group_values))
    stats = {
        "median_energy_stderr": _median_metric(eligible_rows, "validation/energy_stderr"),
        "energy_iqr": seed_energy_iqr(list(energies.values())),
        "median_energy_variance": _median_metric(eligible_rows, "validation/energy_variance"),
        "median_wall_time_sec": _median_metric(eligible_rows, "runtime/wall_time_sec"),
        "geometry_warning_count": float(len(geometry_flags(rows, manifest))),
    }

    tie_breaker_values: list[float] = []
    for tie_breaker in selection.get("tie_breakers", ()):
        tie_breaker_values.append(_tie_breaker_value(str(tie_breaker), group, stats, eligible_rows))

    config_id = rows[0].get("config_id", "") if rows else ""
    return {
        **group,
        "config_id": config_id,
        "score": score,
        "stats": stats,
        "tie_breakers": tie_breaker_values,
        "seed_energies": energies,
        "n_seeds_expected": len(expected_seeds),
        "n_success": sum(1 for energy in energies.values() if math.isfinite(energy)),
        "n_failed_seeds": n_failed,
        "rows": rows,
    }


def _tie_breaker_value(
    name: str, group: dict[str, str], stats: dict[str, float], eligible_rows: list[dict[str, str]]
) -> float:
    """Resolve one declared tie-breaker to a number (lower always wins)."""

    if name == "validation_energy_iqr":
        return stats["energy_iqr"]
    if name == "geometry_warning_count":
        return stats["geometry_warning_count"]
    if name in group:  # hyperparameter tie-breaker, e.g. model_params.channels
        value = _parse_float(group.get(name))
        return value if value is not None else math.inf
    return _median_metric(eligible_rows, name)


def selection_margin(a: dict[str, object], b: dict[str, object], manifest: dict) -> float:
    """Margin below which two configs' aggregate energies are tied.

    ``max(k * sqrt(stderr_a^2 + stderr_b^2), f * max(iqr_a, iqr_b), floor)``
    per the manifest ``selection`` policy. Nonfinite stats (missing metrics)
    are dropped from the max so one broken column cannot tie everything.
    """

    selection = manifest.get("selection", {})
    floor = float(selection.get("absolute_energy_floor", 0.0))
    margin_cfg = selection.get("margin", {})
    multiplier = float(margin_cfg.get("stderr_multiplier", 2.0))
    iqr_fraction = float(margin_cfg.get("seed_iqr_fraction", 0.25))

    stderr_a = float(a["stats"]["median_energy_stderr"])
    stderr_b = float(b["stats"]["median_energy_stderr"])
    stderr_term = multiplier * math.sqrt(stderr_a**2 + stderr_b**2)
    iqr_term = iqr_fraction * max(float(a["stats"]["energy_iqr"]), float(b["stats"]["energy_iqr"]))

    finite_terms = [term for term in (stderr_term, iqr_term) if math.isfinite(term)]
    return max([*finite_terms, floor])


def _pick_round_winner(
    remaining: list[dict[str, object]], manifest: dict
) -> tuple[dict[str, object], dict[str, object]]:
    """Pick one winner from the remaining groups; returns (winner, decision)."""

    best = min(remaining, key=lambda group: (group["score"], group["config_id"]))
    if math.isfinite(float(best["score"])):
        margins = {
            str(group["config_id"]): selection_margin(best, group, manifest) for group in remaining
        }
        tied = [
            group
            for group in remaining
            if float(group["score"]) <= float(best["score"]) + margins[str(group["config_id"])]
        ]
    else:
        # All-failed groups carry no energy information; they tie as a block.
        margins = {}
        tied = [group for group in remaining if not math.isfinite(float(group["score"]))]

    winner = min(tied, key=lambda group: (tuple(group["tie_breakers"]), group["config_id"]))

    tie_breaker_names = [str(name) for name in manifest.get("selection", {}).get("tie_breakers", ())]
    deciding = None
    if len(tied) > 1:
        deciding = "config_id"  # fallback when every tie-breaker is equal
        for index, name in enumerate(tie_breaker_names):
            if len({float(group["tie_breakers"][index]) for group in tied}) > 1:
                deciding = name
                break
    decision = {
        "tie_set": [str(group["config_id"]) for group in tied],
        "margins_vs_best": {cid: margins[cid] for cid in margins if any(
            cid == str(group["config_id"]) for group in tied
        )},
        "deciding_tie_breaker": deciding,
    }
    return winner, decision


def rank_groups(rows: list[dict[str, str]], manifest: dict) -> list[dict[str, object]]:
    """Rank config groups by margin-aware energy comparison, then tie-breakers.

    Iteratively selects a winner from the remaining groups: the lowest
    aggregate energy wins outright only when every other candidate is more
    than the selection margin away; candidates within the margin are tied and
    the manifest tie-breakers decide, in declared order. Each round's decision
    is recorded on the winner under ``decision``.
    """

    scored = [
        score_group(values, group_rows, manifest)
        for values, group_rows in _group_rows(rows, manifest).items()
    ]
    ranked: list[dict[str, object]] = []
    remaining = list(scored)
    while remaining:
        winner, decision = _pick_round_winner(remaining, manifest)
        winner["decision"] = decision
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


def geometry_flags(rows: list[dict[str, str]], manifest: dict) -> list[str]:
    """Flag suspicious sampler geometry for the given rows.

    Informational and a late tie-breaker only (``geometry_warning_count``);
    geometry never overrides a clear energy difference.
    """

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


_STAT_COLUMNS = (
    "median_energy_stderr",
    "energy_iqr",
    "median_energy_variance",
    "geometry_warning_count",
    "median_wall_time_sec",
)


def write_selection_csv(ranked: list[dict[str, object]], manifest: dict, path: Path) -> None:
    keys = group_keys(manifest)
    columns = [
        "rank",
        "config_id",
        *keys,
        "score",
        *_STAT_COLUMNS,
        "n_success",
        "n_failed_seeds",
        "n_seeds_expected",
        "deciding_tie_breaker",
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
                    *[_fmt(float(group["stats"][stat])) for stat in _STAT_COLUMNS],
                    group["n_success"],
                    group["n_failed_seeds"],
                    group["n_seeds_expected"],
                    group["decision"]["deciding_tie_breaker"] or "",
                ]
            )


def write_selected_config(winner: dict[str, object], manifest: dict, path: Path) -> None:
    keys = group_keys(manifest)
    overrides = [f"{key}={winner.get(key)}" for key in keys]
    selection = manifest.get("selection", {})
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
            "stats": {stat: float(winner["stats"][stat]) for stat in _STAT_COLUMNS},
            "tie_set": winner["decision"]["tie_set"],
            "deciding_tie_breaker": winner["decision"]["deciding_tie_breaker"],
            "policy": {
                "absolute_energy_floor": selection.get("absolute_energy_floor"),
                "margin": selection.get("margin"),
                "require_all_seeds": selection.get("require_all_seeds"),
                "tie_breakers": list(selection.get("tie_breakers", ())),
            },
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_report(
    ranked: list[dict[str, object]], flags: list[str], manifest: dict, path: Path
) -> None:
    keys = group_keys(manifest)
    selection = manifest.get("selection", {})
    tie_breaker_names = [str(name) for name in selection.get("tie_breakers", ())]
    geometry_fields = list(manifest.get("diagnostic_fields", {}).get("sampler_geometry", ()))
    lines = [
        f"# Selection report: {manifest['study']['name']}",
        "",
        f"- selection metric: `{manifest['validation']['metric']}`"
        f" ({manifest['validation'].get('aggregate', 'median')} over seeds)",
        f"- failed/ineligible seeds count as `{manifest['validation'].get('failed_run_value', 'inf')}`"
        + (
            "; any failed seed fails the whole config (`require_all_seeds`)"
            if selection.get("require_all_seeds")
            else ""
        ),
        "- selection margin: `max("
        f"{selection.get('margin', {}).get('stderr_multiplier', 2.0)} * sqrt(stderr_A^2 + stderr_B^2), "
        f"{selection.get('margin', {}).get('seed_iqr_fraction', 0.25)} * max(iqr_A, iqr_B), "
        f"{selection.get('absolute_energy_floor', 0.0)})`",
        f"- tie-breakers (in order): {', '.join(f'`{t}`' for t in tie_breaker_names)}",
        "- inputs: local run outputs only (no W&B, no exact reference energy)",
        "",
        "## Ranking",
        "",
        "| rank | config_id | "
        + " | ".join(keys)
        + " | score | stderr | iqr | variance | geom warns | failed seeds |",
        "|---|---|" + "---|" * (len(keys) + 6),
    ]
    for rank, group in enumerate(ranked, start=1):
        cells = [str(rank), str(group["config_id"]), *[str(group.get(key, "")) for key in keys]]
        cells += [
            _fmt(float(group["score"])),
            _fmt(float(group["stats"]["median_energy_stderr"])),
            _fmt(float(group["stats"]["energy_iqr"])),
            _fmt(float(group["stats"]["median_energy_variance"])),
            _fmt(float(group["stats"]["geometry_warning_count"])),
            f"{group['n_failed_seeds']}/{group['n_seeds_expected']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if ranked:
        winner = ranked[0]
        decision = winner["decision"]
        lines += [
            "",
            "## Selected",
            "",
            f"`{winner['config_id']}` with {manifest['validation'].get('aggregate', 'median')} "
            f"`{manifest['validation']['metric']}` = {_fmt(float(winner['score']))}",
            "",
            "Seed energies: "
            + ", ".join(f"{seed}: {_fmt(energy)}" for seed, energy in winner["seed_energies"].items()),
            "",
            "### Selection margin and tie-breakers",
            "",
        ]
        if len(decision["tie_set"]) == 1:
            lines.append(
                "The winner beat every other candidate by more than the selection margin; "
                "no tie-breakers were needed."
            )
        else:
            lines += [
                f"{len(decision['tie_set'])} candidates were within the selection margin of the "
                f"best median energy; the tie was decided by `{decision['deciding_tie_breaker']}`.",
                "",
                "Margins vs best (a candidate is tied when its score is within this margin):",
                "",
            ]
            lines += [
                f"- `{config_id}`: margin = {_fmt(margin)}"
                for config_id, margin in decision["margins_vs_best"].items()
            ]
            tied_groups = [
                group for group in ranked if str(group["config_id"]) in decision["tie_set"]
            ]
            lines += [
                "",
                "| config_id | score | " + " | ".join(tie_breaker_names) + " |",
                "|---|---|" + "---|" * len(tie_breaker_names),
            ]
            for group in tied_groups:
                cells = [str(group["config_id"]), _fmt(float(group["score"]))]
                cells += [_fmt(float(value)) for value in group["tie_breakers"]]
                lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Sampler geometry diagnostics", ""]
    if geometry_fields and ranked:
        winner_rows = ranked[0]["rows"]
        lines += [
            "Winner geometry (informational; geometry only decides via the "
            "`geometry_warning_count` tie-breaker):",
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
                "tie_set": winner["decision"]["tie_set"],
                "deciding_tie_breaker": winner["decision"]["deciding_tie_breaker"],
                "groups": len(ranked),
                "geometry_flags": len(flags),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
