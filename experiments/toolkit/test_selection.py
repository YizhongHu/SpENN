"""Deterministic table-fixture tests for the champion-selection engine."""

from __future__ import annotations

import math

import pytest

from experiments.toolkit.selection import (
    aggregate_candidates,
    champion_record,
    group_key,
    group_label_from_key,
    normalize_champion_specs,
    parse_group_by,
    reference_columns,
    reference_metrics,
    select_by_spec,
)

SUCCESS = {"completed", "success"}


def _id_for_axes(point, axes, labels):
    return "_".join(f"{labels.get(axis, axis)}-{point[axis]}" for axis in axes)


def _seed_rows():
    """Two configs x two seeds; config a beats config b on eval/tail/outlier."""

    def row(basis, seed, status, outlier, wall):
        return {
            "basis": basis,
            "seed_index": seed,
            "status": status,
            "run_id": f"b-{basis}_seed-{seed}",
            "eval/tail/outlier": outlier,
            "custom/wall_sec": wall,
        }

    return [
        row("B00", "0", "completed", "0.10", "5.0"),
        row("B00", "1", "completed", "0.20", "7.0"),
        row("B01", "0", "completed", "0.90", "1.0"),
        row("B01", "1", "completed", "1.10", "2.0"),
    ]


def _aggregate(rows, *, success_statuses=SUCCESS):
    return aggregate_candidates(
        rows,
        config_keys=("basis",),
        major_axes=("basis",),
        minor_axes=(),
        seed_key="seed_index",
        axis_id_labels={"basis": "b"},
        success_statuses=success_statuses,
        id_for_axes=_id_for_axes,
    )


def test_aggregate_candidates_writes_seed_statistics_and_counts() -> None:
    candidates, used_fallback = _aggregate(_seed_rows())

    assert not used_fallback
    assert [candidate["config_id"] for candidate in candidates] == ["b-B00", "b-B01"]
    b00 = candidates[0]
    assert b00["seeds"] == "0,1"
    assert b00["n_expected"] == 2
    assert b00["n_success"] == 2
    assert b00["n_failed"] == 0
    assert float(b00["eval/tail/outlier_seed_median"]) == pytest.approx(0.15)
    assert float(b00["eval/tail/outlier_seed_mean"]) == pytest.approx(0.15)
    assert float(b00["eval/tail/outlier_seed_stderr"]) == pytest.approx(
        math.sqrt(((0.10 - 0.15) ** 2 + (0.20 - 0.15) ** 2) / 1) / math.sqrt(2)
    )
    assert b00["eval/tail/outlier_seed_n"] == "2"


def test_aggregate_candidates_missing_seed_ranks_as_infinite_median() -> None:
    rows = [row for row in _seed_rows() if not (row["basis"] == "B01" and row["seed_index"] == "1")]

    candidates, _ = _aggregate(rows)

    b01 = candidates[1]
    assert b01["n_missing_seed"] == 1
    assert b01["eval/tail/outlier_seed_median"] == ""  # inf median -> blank CSV number
    assert b01["eval/tail/outlier_seed_n"] == "1"


def test_aggregate_candidates_status_fallback_when_no_successes() -> None:
    rows = _seed_rows()
    for row in rows:
        row["status"] = "failed"

    candidates, used_fallback = _aggregate(rows)

    assert used_fallback
    assert float(candidates[0]["eval/tail/outlier_seed_median"]) == pytest.approx(0.15)


def test_aggregate_candidates_accepts_foreign_status_vocabulary() -> None:
    rows = _seed_rows()
    for row in rows:
        row["status"] = "green"

    candidates, used_fallback = _aggregate(rows, success_statuses={"green"})

    assert not used_fallback
    assert candidates[0]["n_success"] == 2


def test_metric_ladder_selects_clear_winner_and_reports_decisions() -> None:
    candidates, _ = _aggregate(_seed_rows())
    spec = {
        "name": "overall",
        "selector": "metric_ladder",
        "tasks": ["tail"],
        "metric_template": "eval/{task}/outlier",
    }

    row, decisions, metric, value = select_by_spec(
        candidates,
        normalize_champion_specs([spec])[0],
        selected_by_name={},
        default_fallback_metric="custom/wall_sec",
    )

    assert row["config_id"] == "b-B00"
    assert metric == "eval/tail/outlier_seed_median"
    assert float(value) == pytest.approx(0.15)
    assert any("clearly wins" in decision for decision in decisions)


def test_metric_ladder_falls_back_on_overlapping_error_bars() -> None:
    rows = _seed_rows()
    for row in rows:
        row["eval/tail/outlier"] = {"B00": "0.5", "B01": "0.6"}[row["basis"]]
        if row["basis"] == "B00":
            row["eval/tail/outlier"] = {"0": "0.1", "1": "1.1"}[row["seed_index"]]
        else:
            row["eval/tail/outlier"] = {"0": "0.2", "1": "1.0"}[row["seed_index"]]
    candidates, _ = _aggregate(rows)
    spec = {
        "name": "overall",
        "selector": "metric_ladder",
        "tasks": ["tail"],
        "metric_template": "eval/{task}/outlier",
    }

    row, decisions, metric, _value = select_by_spec(
        candidates,
        normalize_champion_specs([spec])[0],
        selected_by_name={},
        default_fallback_metric="custom/wall_sec",
    )

    assert metric == "custom/wall_sec_seed_median"
    assert row["config_id"] == "b-B01"  # smaller wall time
    assert any("fallback" in decision for decision in decisions)


def test_scalar_metric_selector_with_exclusion_picks_next_best() -> None:
    candidates, _ = _aggregate(_seed_rows())
    overall = candidates[0]
    spec = normalize_champion_specs(
        [{"name": "runner_up", "selector": "metric", "metric": "eval/tail/outlier", "exclude": "overall"}]
    )[0]

    row, decisions, _metric, _value = select_by_spec(
        candidates,
        spec,
        selected_by_name={"overall": overall},
        default_fallback_metric="custom/wall_sec",
    )

    assert row["config_id"] == "b-B01"
    assert any("excluded" in decision for decision in decisions)


def test_scalar_metric_selector_mode_max() -> None:
    candidates, _ = _aggregate(_seed_rows())
    spec = normalize_champion_specs(
        [{"name": "worst", "selector": "metric", "metric": "eval/tail/outlier", "mode": "max"}]
    )[0]

    row, _decisions, _metric, _value = select_by_spec(
        candidates,
        spec,
        selected_by_name={},
        default_fallback_metric="custom/wall_sec",
    )

    assert row["config_id"] == "b-B01"


def test_normalize_champion_specs_rejects_incomplete_entries() -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        normalize_champion_specs([{"selector": "metric"}])
    with pytest.raises(ValueError, match="requires selector"):
        normalize_champion_specs([{"name": "x"}])
    with pytest.raises(ValueError, match="at least one selector"):
        normalize_champion_specs([])


def test_select_by_spec_rejects_unknown_selector() -> None:
    with pytest.raises(ValueError, match="unsupported champion selector"):
        select_by_spec(
            [],
            {"name": "x", "selector": "pareto"},
            selected_by_name={},
            default_fallback_metric="custom/wall_sec",
        )


def test_group_by_parsing_and_labels() -> None:
    assert parse_group_by("basis, mechanism") == ("basis", "mechanism")
    assert parse_group_by(["basis"]) == ("basis",)
    with pytest.raises(ValueError, match="at least one column"):
        parse_group_by("")
    key = group_key({"basis": "B00", "mechanism": None}, ("basis", "mechanism"))
    assert key == ("B00", "")
    assert group_label_from_key(("basis", "mechanism"), key) == "basis=B00|mechanism="


def test_reference_metrics_and_columns() -> None:
    metrics = reference_metrics([{"label": "energy", "metric": "eval/energy/mean"}, ("var", "eval/var")])
    assert metrics == (("energy", "eval/energy/mean"), ("var", "eval/var"))
    assert reference_columns(metrics, reference_statistics=("median", "n")) == [
        "energy_seed_median",
        "energy_seed_n",
        "var_seed_median",
        "var_seed_n",
    ]
    assert reference_metrics(None) == ()
    with pytest.raises(ValueError, match="label and metric"):
        reference_metrics([{"label": "", "metric": "m"}])


def test_champion_record_projects_group_reference_and_identity_columns() -> None:
    candidates, _ = _aggregate(_seed_rows())
    winner = candidates[0]

    record = champion_record(
        winner,
        group_keys=("winner",),
        group_key=("overall",),
        config_keys=("basis",),
        winner_kind="energy",
        metric="eval/tail/outlier_seed_median",
        metric_value="0.15",
        reference_metrics=(("outlier", "eval/tail/outlier"),),
        reference_statistics=("median",),
    )

    assert record["winner"] == "overall"
    assert record["winner_kind"] == "energy"
    assert record["config_id"] == "b-B00"
    assert float(record["metric_seed_mean"]) == pytest.approx(0.15)
    assert float(record["outlier_seed_median"]) == pytest.approx(0.15)

    empty = champion_record(
        None,
        group_keys=("winner",),
        group_key=("overall",),
        config_keys=("basis",),
        winner_kind="energy",
        metric="m",
        metric_value="",
        reference_metrics=(),
        reference_statistics=(),
    )
    assert empty["config_id"] == ""
    assert empty["basis"] == ""
