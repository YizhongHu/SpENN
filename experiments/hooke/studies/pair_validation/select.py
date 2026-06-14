"""Select the Hooke pair validation winner from normalized local run tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    from .study_manifest import collect_report_dir, expected_validation_seeds, select_report_dir
except ImportError:  # pragma: no cover - direct script execution
    from study_manifest import collect_report_dir, expected_validation_seeds, select_report_dir

GROUP_KEYS = (
    "optimizer_params.lr",
    "model_params.channels",
    "model_params.layers",
    "model_params.gate_activation",
)
REPORT_TOP_CANDIDATE_LIMIT = 10

FORBIDDEN_SELECTION_METRICS = {
    "validation/energy_error",
    "validation/energy_abs_error",
    "eval/energy_error",
    "eval/energy_abs_error",
}


@dataclass
class Candidate:
    """Aggregated validation metrics for one non-seed config."""

    config_id: str
    key: tuple[str, ...]
    hyperparameters: dict[str, Any]
    rows: list[dict[str, Any]]
    n_expected: int
    n_present: int
    n_success: int
    n_failed: int
    n_missing_seed: int
    median_energy: float
    median_energy_stderr: float
    energy_iqr: float
    median_energy_variance: float
    median_wall_time_sec: float
    geometry_warning_count: int
    geometry_warnings: list[str]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selector CLI."""

    args = _parse_args(argv)
    select_runs(manifest_path=args.manifest, runs_path=args.runs, output_dir=args.output_dir)
    return 0


def select_runs(
    *,
    manifest_path: str | Path,
    runs_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Select a deterministic winner and write selection artifacts."""

    manifest = _load_yaml(manifest_path)
    _validate_selection_contract(manifest)
    if runs_path is None:
        runs_path = Path(collect_report_dir(manifest)) / "runs.csv"
    rows = _read_runs_csv(runs_path)
    candidates = _aggregate_candidates(rows, manifest)
    if not candidates:
        raise ValueError("no candidate rows found in runs.csv")

    winner, tied, decisions = _choose_winner(candidates, manifest)
    output = Path(output_dir) if output_dir is not None else Path(select_report_dir(manifest))
    output.mkdir(parents=True, exist_ok=True)
    _write_selection_csv(candidates, winner, output / "selection.csv")
    _write_selection_jsonl(candidates, winner, output / "selection.jsonl")
    selected = _selected_config(manifest, winner, tied, decisions, runs_path, output)
    OmegaConf.save(config=OmegaConf.create(selected), f=output / "selected_config.yaml", resolve=False)
    (output / "selection_report.md").write_text(
        _selection_report(manifest, candidates, winner, tied, decisions),
        encoding="utf-8",
    )
    return selected


def geometry_warnings(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    """Return sampler-geometry warning messages for one run row."""

    policy = manifest.get("geometry_warnings") if isinstance(manifest.get("geometry_warnings"), Mapping) else {}
    warnings: list[str] = []
    require_radius_q99 = bool(policy.get("require_radius_q99", False))
    if require_radius_q99 and _is_missing(row.get("validation/sampler/radius_q99")):
        warnings.append("validation/sampler/radius_q99 missing")
    elif not _is_missing(row.get("validation/sampler/radius_q99")) and not _is_finite(row.get("validation/sampler/radius_q99")):
        warnings.append("validation/sampler/radius_q99 nonfinite")

    radius_max = row.get("validation/sampler/radius_max")
    if not _is_missing(radius_max) and not _is_finite(radius_max):
        warnings.append("validation/sampler/radius_max nonfinite")

    position_rms = row.get("validation/sampler/position_rms")
    if not _is_missing(position_rms) and not _is_finite(position_rms):
        warnings.append("validation/sampler/position_rms nonfinite")

    threshold_n = policy.get("require_electron_distance_q01_for_n_particles_ge")
    if threshold_n is not None:
        n_particles = _infer_n_particles(row, manifest)
        if n_particles is None:
            warnings.append("unknown N for electron-distance requirement")
        elif n_particles >= int(float(threshold_n)):
            q01 = row.get("validation/sampler/electron_distance_q01")
            if _is_missing(q01):
                warnings.append("validation/sampler/electron_distance_q01 missing")
            elif not _is_finite(q01):
                warnings.append("validation/sampler/electron_distance_q01 nonfinite")
            else:
                minimum = policy.get("min_electron_distance_q01")
                if minimum is not None and float(q01) < float(minimum):
                    warnings.append(
                        "validation/sampler/electron_distance_q01 below "
                        f"min_electron_distance_q01={float(minimum):g}"
                    )
    return warnings


def selection_margin(a: Candidate, b: Candidate, manifest: Mapping[str, Any]) -> float:
    """Compute the manifest-declared pairwise selection margin."""

    selection = _selection_block(manifest)
    margin = selection.get("margin") if isinstance(selection.get("margin"), Mapping) else {}
    stderr_multiplier = float(margin.get("stderr_multiplier", 2.0))
    seed_iqr_fraction = float(margin.get("seed_iqr_fraction", 0.25))
    floor = float(selection.get("absolute_energy_floor", 1.0e-4))
    stderr_term = stderr_multiplier * math.sqrt(a.median_energy_stderr**2 + b.median_energy_stderr**2)
    iqr_term = seed_iqr_fraction * max(a.energy_iqr, b.energy_iqr)
    return max(stderr_term, iqr_term, floor)


def parse_bool(value: Any) -> bool:
    """Parse check flags logged as booleans or numeric scalars."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "t", "yes"}:
        return True
    if text in {"false", "f", "no", "", "none", "null"}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _validate_selection_contract(manifest: Mapping[str, Any]) -> None:
    selection = _selection_block(manifest)
    metric = str(selection.get("metric", "validation/energy"))
    if metric in FORBIDDEN_SELECTION_METRICS:
        raise ValueError(f"selection metric {metric!r} uses exact-reference error and is forbidden")
    if metric != "validation/energy":
        raise ValueError("PR8.3 selector only supports selection.metric=validation/energy")
    aggregate = str(selection.get("aggregate", "median"))
    if aggregate != "median":
        raise ValueError("PR8.3 selector only supports selection.aggregate=median")


def _aggregate_candidates(rows: list[dict[str, Any]], manifest: Mapping[str, Any]) -> list[Candidate]:
    expected_seeds = _expected_seeds(manifest)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(_key_text(row.get(field)) for field in GROUP_KEYS)
        grouped.setdefault(key, []).append(row)

    candidates: list[Candidate] = []
    for key, group_rows in grouped.items():
        seed_rows = {_key_text(row.get("runtime.seed")): row for row in group_rows}
        seed_order = expected_seeds or sorted(seed_rows)
        energy_values: list[float] = []
        stderr_values: list[float] = []
        variance_values: list[float] = []
        wall_time_values: list[float] = []
        warnings: list[str] = []
        n_success = 0
        n_failed = 0
        n_missing_seed = 0

        for seed in seed_order:
            row = seed_rows.get(seed)
            if row is None:
                n_failed += 1
                n_missing_seed += 1
                energy_values.append(math.inf)
                continue
            row_warnings = geometry_warnings(row, manifest)
            warnings.extend(f"seed {seed}: {message}" for message in row_warnings)
            if _is_eligible(row, manifest):
                n_success += 1
                energy_values.append(_as_float(row.get("validation/energy"), default=math.inf))
                stderr_values.append(_as_float(row.get("validation/energy_stderr"), default=math.inf))
                variance_values.append(_as_float(row.get("validation/energy_variance"), default=math.inf))
                wall_time_values.append(_as_float(row.get("runtime/wall_time_sec"), default=math.inf))
            else:
                n_failed += 1
                energy_values.append(math.inf)

        config_id = _first_nonempty(row.get("config_id") for row in group_rows) or _default_config_id(key)
        candidates.append(
            Candidate(
                config_id=str(config_id),
                key=key,
                hyperparameters={field: _parse_scalar(key[index]) for index, field in enumerate(GROUP_KEYS)},
                rows=sorted(group_rows, key=lambda row: _key_text(row.get("runtime.seed"))),
                n_expected=len(seed_order),
                n_present=len(group_rows),
                n_success=n_success,
                n_failed=n_failed,
                n_missing_seed=n_missing_seed,
                median_energy=_median(energy_values),
                median_energy_stderr=_median(stderr_values),
                energy_iqr=_iqr(energy_values),
                median_energy_variance=_median(variance_values),
                median_wall_time_sec=_median(wall_time_values),
                geometry_warning_count=len(warnings),
                geometry_warnings=warnings,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.key)


def _choose_winner(
    candidates: list[Candidate],
    manifest: Mapping[str, Any],
) -> tuple[Candidate, list[Candidate], list[str]]:
    finite = [candidate for candidate in candidates if math.isfinite(candidate.median_energy)]
    if not finite:
        raise ValueError("no candidate has a finite eligible median validation/energy; refusing to select a winner")

    ordered = sorted(finite, key=lambda candidate: (candidate.median_energy, candidate.key))
    leader = ordered[0]
    tied = [candidate for candidate in ordered if candidate is leader or not _clearly_beats(leader, candidate, manifest)]
    decisions: list[str] = []
    if len(tied) == 1:
        decisions.append("Lowest median validation/energy clearly beats every other finite candidate.")
        return leader, tied, decisions

    decisions.append(
        "Primary median validation/energy does not clearly separate the "
        f"{len(tied)} candidates in the primary-energy cohort; see cohort table for margins."
    )
    remaining = list(tied)
    for breaker in _tie_breakers(manifest):
        best_value = min(_breaker_value(candidate, breaker) for candidate in remaining)
        next_remaining = [
            candidate
            for candidate in remaining
            if _same_breaker_value(_breaker_value(candidate, breaker), best_value)
        ]
        decisions.append(
            f"{breaker}: best={_format_number(best_value)}; "
            f"remaining={', '.join(candidate.config_id for candidate in next_remaining)}"
        )
        remaining = next_remaining
        if len(remaining) == 1:
            return remaining[0], tied, decisions

    remaining.sort(key=lambda candidate: candidate.key)
    decisions.append(
        "All declared tie-breakers remained tied; using deterministic config-key order: "
        + ", ".join(candidate.config_id for candidate in remaining)
    )
    return remaining[0], tied, decisions


def _clearly_beats(a: Candidate, b: Candidate, manifest: Mapping[str, Any]) -> bool:
    return a.median_energy + selection_margin(a, b, manifest) < b.median_energy


def _breaker_value(candidate: Candidate, breaker: str) -> float:
    if breaker == "validation/energy_variance":
        return candidate.median_energy_variance
    if breaker == "validation_energy_iqr":
        return candidate.energy_iqr
    if breaker == "validation/energy_stderr":
        return candidate.median_energy_stderr
    if breaker == "geometry_warning_count":
        return float(candidate.geometry_warning_count)
    if breaker == "model_params.channels":
        return _as_float(candidate.hyperparameters.get("model_params.channels"), default=math.inf)
    if breaker == "runtime/wall_time_sec":
        return candidate.median_wall_time_sec
    raise ValueError(f"unsupported tie-breaker {breaker!r}")


def _same_breaker_value(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-12)


def _is_eligible(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    if str(row.get("status", "")).lower() != "completed":
        return False
    eligibility = manifest.get("eligibility") if isinstance(manifest.get("eligibility"), Mapping) else {}
    for key in eligibility.get("require", []) or []:
        if not parse_bool(row.get(str(key))):
            return False
    required_fraction = eligibility.get("local_energy_finite_fraction", 1.0)
    actual_fraction = _as_float(row.get("validation/local_energy_finite_fraction"), default=math.nan)
    if not math.isfinite(actual_fraction) or not math.isclose(actual_fraction, float(required_fraction), rel_tol=0.0, abs_tol=1.0e-12):
        return False
    return math.isfinite(_as_float(row.get("validation/energy"), default=math.inf))


def _write_selection_csv(candidates: list[Candidate], winner: Candidate, path: Path) -> None:
    columns = (
        "selected",
        "config_id",
        *GROUP_KEYS,
        "n_expected",
        "n_present",
        "n_success",
        "n_failed",
        "n_missing_seed",
        "median validation/energy",
        "median_energy_stderr",
        "energy_iqr",
        "median_energy_variance",
        "median_wall_time_sec",
        "geometry_warning_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for candidate in candidates:
            row = {
                "selected": "true" if candidate is winner else "false",
                "config_id": candidate.config_id,
                "n_expected": candidate.n_expected,
                "n_present": candidate.n_present,
                "n_success": candidate.n_success,
                "n_failed": candidate.n_failed,
                "n_missing_seed": candidate.n_missing_seed,
                "median validation/energy": _csv_number(candidate.median_energy),
                "median_energy_stderr": _csv_number(candidate.median_energy_stderr),
                "energy_iqr": _csv_number(candidate.energy_iqr),
                "median_energy_variance": _csv_number(candidate.median_energy_variance),
                "median_wall_time_sec": _csv_number(candidate.median_wall_time_sec),
                "geometry_warning_count": candidate.geometry_warning_count,
            }
            row.update(candidate.hyperparameters)
            writer.writerow(row)


def _write_selection_jsonl(candidates: list[Candidate], winner: Candidate, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = _candidate_payload(candidate, selected=candidate is winner)
            handle.write(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False))
            handle.write("\n")


def _selected_config(
    manifest: Mapping[str, Any],
    winner: Candidate,
    tied: list[Candidate],
    decisions: list[str],
    runs_path: str | Path,
    output_dir: Path,
) -> dict[str, Any]:
    nested_hyperparameters = _nested_hyperparameters(winner.hyperparameters)
    return {
        "study": {
            "name": _select(manifest, "study.name"),
            "version": _select(manifest, "study.version"),
            "source_phase": _select(manifest, "selection.source_phase") or "validation_train",
            "source_runs": str(runs_path),
            "selection_report": str(output_dir / "selection_report.md"),
        },
        "selection": {
            "metric": "validation/energy",
            "aggregate": "median",
            "selected_config_id": winner.config_id,
            "tied_config_ids": [candidate.config_id for candidate in tied],
            "tie_breaker_decisions": decisions,
        },
        "selected": {
            "config_id": winner.config_id,
            **nested_hyperparameters,
            "hyperparameters": winner.hyperparameters,
            "n_success": winner.n_success,
            "n_failed": winner.n_failed,
            "median_energy": _finite_or_text(winner.median_energy),
            "median_energy_stderr": _finite_or_text(winner.median_energy_stderr),
            "energy_iqr": _finite_or_text(winner.energy_iqr),
            "median_energy_variance": _finite_or_text(winner.median_energy_variance),
            "geometry_warning_count": winner.geometry_warning_count,
            "validation_runs": [_validation_run_payload(row) for row in winner.rows],
        },
    }


def _validation_run_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": row.get("run_dir"),
        "training_seed": row.get("runtime.seed"),
        "status": row.get("status"),
        "checkpoint_path": row.get("checkpoint/latest_path"),
        "validation_energy": row.get("validation/energy"),
        "validation_energy_stderr": row.get("validation/energy_stderr"),
        "git_sha": row.get("git/sha"),
    }


def _candidate_payload(candidate: Candidate, *, selected: bool) -> dict[str, Any]:
    return {
        "selected": selected,
        "config_id": candidate.config_id,
        "hyperparameters": candidate.hyperparameters,
        "n_expected": candidate.n_expected,
        "n_present": candidate.n_present,
        "n_success": candidate.n_success,
        "n_failed": candidate.n_failed,
        "n_missing_seed": candidate.n_missing_seed,
        "median_validation_energy": _finite_or_text(candidate.median_energy),
        "median_validation_energy_stderr": _finite_or_text(candidate.median_energy_stderr),
        "validation_energy_iqr": _finite_or_text(candidate.energy_iqr),
        "median_validation_energy_variance": _finite_or_text(candidate.median_energy_variance),
        "median_wall_time_sec": _finite_or_text(candidate.median_wall_time_sec),
        "geometry_warning_count": candidate.geometry_warning_count,
        "geometry_warnings": candidate.geometry_warnings,
    }


def _nested_hyperparameters(hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        current = nested
        parts = str(key).split(".")
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"hyperparameter path collides with scalar: {key}")
            current = child
        current[parts[-1]] = value
    return nested


def _selection_report(
    manifest: Mapping[str, Any],
    candidates: Sequence[Candidate],
    winner: Candidate,
    tied: list[Candidate],
    decisions: list[str],
) -> str:
    finite = [candidate for candidate in candidates if math.isfinite(candidate.median_energy)]
    energy_leader = min(finite, key=lambda candidate: (candidate.median_energy, candidate.key))
    final_ranked = _final_ranking(candidates, tied, manifest)
    status_counts = _run_status_counts(candidates)
    expected_runs = sum(candidate.n_expected for candidate in candidates)
    found_runs = sum(candidate.n_present for candidate in candidates)
    lines = [
        f"# Selection report: {_select(manifest, 'study.name')} {_select(manifest, 'study.version')}",
        "",
        "## Summary",
        "",
        f"- Expected runs: `{expected_runs}`",
        f"- Found runs: `{found_runs}`",
        "- Completed / failed / missing metrics / missing validation: "
        f"`{status_counts.get('completed', 0)} / {status_counts.get('failed', 0)} / "
        f"{status_counts.get('missing_metrics', 0)} / {status_counts.get('missing_validation', 0)}`",
        f"- Selection metric: median `validation/energy`",
        f"- Selected config: `{winner.config_id}`",
        f"- Final rank: `1` of `{len(candidates)}`",
        f"- Lowest median `validation/energy`: `{energy_leader.config_id}` (`{_format_number(energy_leader.median_energy)}`)",
        f"- Primary-energy cohort size: `{len(tied)}`",
        "- Validation is used for model/protocol selection and does not use exact-reference energy.",
        "",
        "## Selected config",
        "",
    ]
    lines.extend(_hyperparameter_table(winner))
    lines.extend(
        [
            "",
            "## Top candidates",
            "",
            f"Top {min(REPORT_TOP_CANDIDATE_LIMIT, len(final_ranked))} candidates sorted by final ranking.",
            "",
        ]
    )
    lines.extend(_top_candidates_table(final_ranked[:REPORT_TOP_CANDIDATE_LIMIT]))
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
        ]
    )
    lines.extend(_decision_rule_lines(manifest))
    lines.extend(["", "## Why this config won", ""])
    lines.extend(_why_config_won_lines(winner, energy_leader, tied, decisions, manifest))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_warning_lines(candidates, winner, status_counts))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- runs.csv",
            "- runs.jsonl",
            "- selection.csv",
            "- selection.jsonl",
            "- selected_config.yaml",
        ]
    )
    return "\n".join(lines) + "\n"


def _final_ranking(
    candidates: Sequence[Candidate],
    tied: Sequence[Candidate],
    manifest: Mapping[str, Any],
) -> list[Candidate]:
    tie_breakers = _tie_breakers(manifest)
    tied_ids = {id(candidate) for candidate in tied}
    ranked_tied = sorted(
        tied,
        key=lambda candidate: tuple(_breaker_value(candidate, breaker) for breaker in tie_breakers) + candidate.key,
    )
    remaining = [candidate for candidate in candidates if id(candidate) not in tied_ids]
    finite_remaining = sorted(
        [candidate for candidate in remaining if math.isfinite(candidate.median_energy)],
        key=lambda candidate: (candidate.median_energy, candidate.key),
    )
    nonfinite_remaining = sorted(
        [candidate for candidate in remaining if not math.isfinite(candidate.median_energy)],
        key=lambda candidate: candidate.key,
    )
    return ranked_tied + finite_remaining + nonfinite_remaining


def _hyperparameter_table(candidate: Candidate) -> list[str]:
    rows = [
        "| hyperparameter | value |",
        "| --- | --- |",
    ]
    for key in GROUP_KEYS:
        rows.append(f"| `{key}` | `{candidate.hyperparameters.get(key)}` |")
    return rows


def _top_candidates_table(candidates: Sequence[Candidate]) -> list[str]:
    rows = [
        "| rank | config_id | lr | channels | layers | gate activation | median energy | stderr | IQR | failures | geometry warnings |",
        "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, candidate in enumerate(candidates, start=1):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{candidate.config_id}`",
                    _format_number(_as_float(candidate.hyperparameters.get("optimizer_params.lr"), default=math.inf)),
                    _format_number(_as_float(candidate.hyperparameters.get("model_params.channels"), default=math.inf)),
                    _format_number(_as_float(candidate.hyperparameters.get("model_params.layers"), default=math.inf)),
                    f"`{candidate.hyperparameters.get('model_params.gate_activation')}`",
                    _format_number(candidate.median_energy),
                    _format_number(candidate.median_energy_stderr),
                    _format_number(candidate.energy_iqr),
                    str(candidate.n_failed),
                    str(candidate.geometry_warning_count),
                ]
            )
            + " |"
        )
    return rows


def _decision_rule_lines(manifest: Mapping[str, Any]) -> list[str]:
    selection = _selection_block(manifest)
    margin = selection.get("margin") if isinstance(selection.get("margin"), Mapping) else {}
    stderr_multiplier = float(margin.get("stderr_multiplier", 2.0))
    seed_iqr_fraction = float(margin.get("seed_iqr_fraction", 0.25))
    floor = float(selection.get("absolute_energy_floor", 1.0e-4))
    tie_breakers = ", ".join(f"`{breaker}`" for breaker in _tie_breakers(manifest))
    return [
        "- Primary metric: median `validation/energy`; lower is better.",
        "- Eligibility: completed run, required checks pass, full finite local-energy fraction, and finite validation energy.",
        "- Failed, missing-validation, missing-metrics, ineligible, or missing-seed replicates count as `+inf` before aggregation.",
        "- Margin: "
        f"`max({stderr_multiplier:g} * sqrt(stderr_a^2 + stderr_b^2), "
        f"{seed_iqr_fraction:g} * max(IQR_a, IQR_b), {floor:g})`.",
        "- A candidate clearly loses to the energy leader only if "
        "`leader_median_energy + selection_margin < candidate_median_energy`.",
        f"- Tie-breakers inside the primary-energy cohort: {tie_breakers}.",
    ]


def _why_config_won_lines(
    winner: Candidate,
    energy_leader: Candidate,
    tied: Sequence[Candidate],
    decisions: Sequence[str],
    manifest: Mapping[str, Any],
) -> list[str]:
    lines = [f"- `{winner.config_id}` is first in the final ranking."]
    if winner is energy_leader and len(tied) == 1:
        lines.append("- It had the lowest median validation energy and cleared the configured selection margin.")
    elif winner is energy_leader:
        lines.append("- It was also the lowest-energy candidate, then remained best after tie-breakers.")
    else:
        delta = winner.median_energy - energy_leader.median_energy
        margin = selection_margin(energy_leader, winner, manifest)
        lines.append(
            f"- `{energy_leader.config_id}` had the lowest median energy, but its "
            f"`{_format_number(delta)}` lead over the selected config was inside the "
            f"`{_format_number(margin)}` selection margin."
        )
    separating = next((decision for decision in decisions[1:] if "remaining=" in decision), None)
    if separating is not None:
        lines.append(f"- First separating tie-breaker: {separating}.")
    elif decisions:
        lines.append(f"- Decision trace: {decisions[-1]}")
    return lines


def _run_status_counts(candidates: Sequence[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        for row in candidate.rows:
            status = str(row.get("status") or "missing").strip().lower()
            counts[status] = counts.get(status, 0) + 1
    return counts


def _warning_lines(
    candidates: Sequence[Candidate],
    winner: Candidate,
    status_counts: Mapping[str, int],
) -> list[str]:
    failed_or_ineligible = sum(candidate.n_failed for candidate in candidates)
    missing_seed = sum(candidate.n_missing_seed for candidate in candidates)
    incomplete_configs = sum(1 for candidate in candidates if candidate.n_success < candidate.n_expected)
    geometry_configs = sum(1 for candidate in candidates if candidate.geometry_warning_count > 0)
    geometry_messages = sum(candidate.geometry_warning_count for candidate in candidates)
    lines = [
        f"- Failed, ineligible, or missing-seed replicates: `{failed_or_ineligible}`.",
        f"- Missing seed replicates: `{missing_seed}`.",
        f"- Missing-metrics runs: `{status_counts.get('missing_metrics', 0)}`.",
        f"- Missing-validation runs: `{status_counts.get('missing_validation', 0)}`.",
        f"- Candidate groups with incomplete eligibility: `{incomplete_configs}`.",
        f"- Geometry warnings: `{geometry_messages}` messages across `{geometry_configs}` candidate groups.",
    ]
    if winner.geometry_warnings:
        joined = "; ".join(winner.geometry_warnings)
        lines.append(f"- Selected config geometry warnings: {joined}.")
    else:
        lines.append("- Selected config geometry warnings: none.")
    return lines


def _read_runs_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [{key: _parse_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    return data if isinstance(data, dict) else {}


def _selection_block(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = manifest.get("selection")
    return selection if isinstance(selection, Mapping) else {}


def _tie_breakers(manifest: Mapping[str, Any]) -> list[str]:
    tie_breakers = _selection_block(manifest).get("tie_breakers") or (
        "validation/energy_variance",
        "validation_energy_iqr",
        "validation/energy_stderr",
        "geometry_warning_count",
        "model_params.channels",
        "runtime/wall_time_sec",
    )
    return [str(item) for item in tie_breakers]


def _expected_seeds(manifest: Mapping[str, Any]) -> list[str]:
    return expected_validation_seeds(manifest)


def _infer_n_particles(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> int | None:
    for key in ("validation/sampler/n_electrons", "system.n_electrons", "system.n_particles"):
        value = row.get(key)
        if not _is_missing(value) and _is_finite(value):
            return int(float(value))
    n_up = row.get("system.spin.n_up")
    n_down = row.get("system.spin.n_down")
    if not _is_missing(n_up) and not _is_missing(n_down) and _is_finite(n_up) and _is_finite(n_down):
        return int(float(n_up) + float(n_down))
    manifest_n = _select(manifest, "system.n_particles")
    if manifest_n is not None:
        return int(float(manifest_n))
    if str(_select(manifest, "study.sector") or "").lower() == "singlet":
        return 2
    return None


def _select(container: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = container
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return _quantile(ordered, 0.75) - _quantile(ordered, 0.25)


def _quantile(ordered: Sequence[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    low = ordered[lower]
    high = ordered[upper]
    if math.isinf(high):
        return high
    if math.isinf(low):
        return low
    return low + (high - low) * (position - lower)


def _parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
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
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _as_float(value: Any, *, default: float) -> float:
    parsed = _parse_scalar(value)
    if parsed is None or isinstance(parsed, bool):
        return default
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return default


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _is_finite(value: Any) -> bool:
    number = _as_float(value, default=math.nan)
    return math.isfinite(number)


def _key_text(value: Any) -> str:
    parsed = _parse_scalar(value)
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return "" if parsed is None else str(parsed)


def _first_nonempty(values: Sequence[Any] | Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _default_config_id(key: tuple[str, ...]) -> str:
    parts = [f"{field.split('.')[-1]}{_slug(value)}" for field, value in zip(GROUP_KEYS, key)]
    return "config_" + "_".join(parts)


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-")


def _csv_number(value: float) -> str:
    return _format_number(value)


def _format_number(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _finite_or_text(value: float) -> float | str:
    return value if math.isfinite(value) else _format_number(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "FORBIDDEN_SELECTION_METRICS",
    "GROUP_KEYS",
    "geometry_warnings",
    "main",
    "parse_bool",
    "select_runs",
    "selection_margin",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
