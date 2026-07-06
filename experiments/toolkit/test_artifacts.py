"""Tests for generic staged-run artifact readers and writers."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.toolkit.artifacts import (
    csv_value,
    duration_from_status,
    duration_from_status_file,
    load_json_dict_if_present,
    load_json_if_present,
    metric_key,
    metric_map,
    read_csv,
    read_metrics_jsonl,
    read_metrics_map,
    status_of,
    write_csv,
)


def test_write_csv_read_csv_roundtrip_with_json_encoded_values(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "table.csv"
    rows = [
        {"run_id": "a", "value": 1.5, "tags": ["x", "y"], "extra": "dropped"},
        {"run_id": "b", "value": None},
    ]

    write_csv(path, rows, columns=["run_id", "value", "tags"])

    assert read_csv(path) == [
        {"run_id": "a", "value": "1.5", "tags": json.dumps(["x", "y"])},
        {"run_id": "b", "value": "", "tags": ""},
    ]
    assert read_csv(tmp_path / "absent.csv") == []


def test_write_csv_without_columns_uses_sorted_union(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"

    write_csv(path, [{"b": 1, "a": 2}, {"c": 3}])

    assert path.read_text().splitlines()[0] == "a,b,c"


def test_csv_value_passes_scalars_and_encodes_containers() -> None:
    assert csv_value(1.5) == 1.5
    assert csv_value("text") == "text"
    assert csv_value(None) is None
    assert csv_value({"b": 1, "a": 2}) == json.dumps({"a": 2, "b": 1}, sort_keys=True)


def test_load_json_helpers_handle_missing_and_non_dict_payloads(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps([1, 2]))

    assert load_json_if_present(path) == [1, 2]
    assert load_json_if_present(tmp_path / "absent.json", "fallback") == "fallback"
    assert load_json_dict_if_present(path) == {}
    assert load_json_dict_if_present(tmp_path / "absent.json") == {}


def test_read_metrics_jsonl_expands_both_record_shapes(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "namespace": "train/", "metrics": {"loss": 0.5, "energy": 2.0}}),
                "",
                json.dumps({"step": 0, "namespace": "runtime", "metric": "wall_time_sec", "value": 3.0}),
                json.dumps({"step": 2, "namespace": "train", "metrics": "not-a-dict"}),
            ]
        )
    )

    assert read_metrics_jsonl(path) == [
        {"step": 1, "namespace": "train", "metric": "loss", "value": 0.5},
        {"step": 1, "namespace": "train", "metric": "energy", "value": 2.0},
        {"step": 0, "namespace": "runtime", "metric": "wall_time_sec", "value": 3.0},
    ]
    assert read_metrics_jsonl(tmp_path / "absent.jsonl") == []


def test_metric_map_is_last_value_wins_in_row_order() -> None:
    rows = [
        {"namespace": "train", "metric": "loss", "value": 1.0},
        {"namespace": "train", "metric": "loss", "value": 0.25},
    ]

    assert metric_map(rows) == {"train/loss": 0.25}
    assert metric_map(rows, prefix="final") == {"final/train/loss": 0.25}
    assert metric_key("", "loss") == "loss"


def test_read_metrics_map_projects_jsonl_to_public_keys(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 0, "namespace": "runtime", "metrics": {"wall_time_sec": 4.5}}) + "\n")

    assert read_metrics_map(path) == {"runtime/wall_time_sec": 4.5}


def test_duration_from_status_handles_missing_and_negative_spans() -> None:
    status = {"start_time": "2026-01-01T00:00:00+00:00", "end_time": "2026-01-01T00:01:30+00:00"}
    reversed_status = {"start_time": status["end_time"], "end_time": status["start_time"]}

    assert duration_from_status(status) == 90.0
    assert duration_from_status({"start_time": status["start_time"]}) is None
    assert duration_from_status({"start_time": "not-a-time", "end_time": status["end_time"]}) is None
    assert duration_from_status(reversed_status) is None
    assert duration_from_status(reversed_status, clamp_negative=True) == 0.0


def test_duration_from_status_file_accepts_dir_or_file_path(tmp_path: Path) -> None:
    status = {"start_time": "2026-01-01T00:00:00+00:00", "end_time": "2026-01-01T00:00:10+00:00"}
    (tmp_path / "status.json").write_text(json.dumps(status))

    assert duration_from_status_file(tmp_path) == 10.0
    assert duration_from_status_file(tmp_path / "status.json") == 10.0
    assert duration_from_status_file(tmp_path / "absent") is None


def test_status_of_reads_attempt_status(tmp_path: Path) -> None:
    assert status_of(tmp_path) == "missing_status"
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed"}))
    assert status_of(tmp_path) == "completed"
    (tmp_path / "status.json").write_text(json.dumps([1]))
    assert status_of(tmp_path) == "unknown"
