"""Collector behavior: absence stays visible and identity does the joining.

The failures guarded here are live ones:

absence is not zero and not blank
    A blank cell parses to NaN and is silently dropped, so a median over two of
    nine rows reads exactly like a median over nine. Every cell states its own
    presence and every aggregate states its coverage.

an ungated run is not a passing run
    Before H-F1 predeclares the tolerances, every value gate reports ``absent``
    with its observed value retained. That is correct. A required availability
    flag that is false still FAILS, so an all-absent row is visibly bad rather
    than quietly green.

identity, not position
    Rows join on run id, checkpoint hash, config hash, seed and stratum. A
    delivered card that is not the requested one fails the row, and four chains
    over one checkpoint that hash differently fail together.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

STUDY_DIR = Path(__file__).resolve().parent


def _load_study_module(name: str) -> ModuleType:
    """Load one study module by path (the study directory is not a package)."""

    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect = _load_study_module("collect")
report = _load_study_module("report")

# Taken from the subject, so the sentinel and the exception classes asserted here
# are the same objects the collector uses. A second importlib load would create a
# second ABSENT singleton, and "is absent" would silently stop meaning anything.
plan = collect.plan_stage
launch = collect.launch_stage
absence = collect.absence

GRID: dict[str, Any] = {
    "study": "tpen_he_v1",
    "train_config": "experiments/atomistic/he-v1/configs/train.yaml",
    "eval_config": "experiments/atomistic/he-v1/configs/eval.yaml",
    "seeds": [0],
    "checkpoint_steps": [10],
    "eval_chains": 2,
    "eval_chain_seed_base": 900000,
    "train_resources": {
        "partition": "kozinsky_gpu",
        "stratum": "a100",
        "timeout_min": 600,
        "cpus": 16,
        "mem_gb": 128,
        "gpus": 1,
    },
    "eval_resources": {
        "partition": "seas_gpu",
        "stratum": "h200",
        "timeout_min": 120,
        "cpus": 8,
        "mem_gb": 64,
        "gpus": 1,
    },
    "gate_spec": {},
}

PLAN_ATTEMPT = "20260815T120000"

HEALTHY_METRICS: dict[str, Any] = {
    "cusp_available": True,
    "cusp_finite_fit_count": 6,
    "cusp_finite_measurement_count": 84,
    "cusp_expected_slope": -2.0,
    "cusp_one_sided_slope_mean": -1.98,
    "cusp_one_sided_slope_abs_error_mean": 0.02,
    "cusp_one_sided_slope_abs_error_max": 0.05,
    "tail_available": True,
    "tail_outer_measurement_count": 6,
    "tail_finite_measurement_count": 84,
    "tail_outer_slope_mean": -2.6,
    "tail_outer_slope_min": -2.9,
    "tail_outer_slope_max": -2.2,
    "tail_negative_slope_fraction": 1.0,
    "tail_outer_radius_min": 4.0,
    "tail_outer_radius_max": 8.0,
}

ENERGY_METRICS: dict[str, Any] = {
    "local_energy_mean": -2.85,
    "local_energy_stderr": 0.004,
    "local_energy_variance": 0.02,
    "local_energy_n_finite": 1024,
    "local_energy_nonfinite_count": 0,
    "reference_energy": -2.903724377034119598,
    "energy_error": 0.0537,
    "energy_abs_error": 0.0537,
}


def _write_plan(tmp_path: Path) -> dict[str, Any]:
    config = plan.validate_grid_config(copy.deepcopy(GRID))
    manifest = plan.build_manifest(
        config=config,
        rows=plan.expand_rows(config),
        attempt_id=PLAN_ATTEMPT,
        results_root=str(tmp_path),
        grid_config_path=None,
        grid_config_sha256=None,
        created_at="2026-08-15T12:00:00-04:00",
    )
    plan.write_plan(manifest, results_root=tmp_path)
    return manifest


def _materialize_row(
    tmp_path: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    metrics: dict[str, Any] | None = None,
    delivered_device: str | None = None,
    matches: bool = True,
    status: str = "completed",
    checkpoint_payload: str = "weights",
) -> Path:
    """Write the artifacts one finished row would have left behind."""

    result_dir = launch.row_result_dir(tmp_path, row, PLAN_ATTEMPT)
    run_dir = result_dir / row["row_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    stratum = row["resources"]["stratum"]
    default_device = "NVIDIA H200" if stratum == "h200" else "NVIDIA A100-SXM4-80GB"
    (result_dir / "allocation_receipt.json").write_text(
        json.dumps(
            {
                "row_id": row["row_id"],
                "job_id": f"job-{row['index']}",
                "hostname": "holygpu8a16101",
                "requested_stratum": stratum,
                "requested_constraint": row["resources"]["constraint"],
                "delivered_device": default_device if delivered_device is None else delivered_device,
                "delivered_device_status": "present",
                "delivered_matches_requested": matches,
                "mismatch_reason": None if matches else "delivered device does not match",
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "row.json").write_text(
        json.dumps({"row": row, "plan_attempt_id": PLAN_ATTEMPT, "launch_attempt_id": "L1"}),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_id": row["row_id"]}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("model: {}\n", encoding="utf-8")
    payload = dict(metrics if metrics is not None else {**HEALTHY_METRICS, **ENERGY_METRICS})
    lines = [
        json.dumps(
            {
                "step": 0,
                "namespace": "eval/he_radial_profiles",
                "metrics": {
                    key: value
                    for key, value in payload.items()
                    if key.startswith(("cusp_", "tail_"))
                },
            }
        ),
        json.dumps(
            {
                "step": 0,
                "namespace": "eval/mcmc_energy",
                "metrics": {
                    key: value
                    for key, value in payload.items()
                    if not key.startswith(("cusp_", "tail_"))
                },
            }
        ),
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if row["kind"] == "eval":
        checkpoint_dir = launch.checkpoint_dir_for_eval_row(
            tmp_path, row, PLAN_ATTEMPT, manifest=manifest
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "manifest.json").write_text(checkpoint_payload, encoding="utf-8")
        (checkpoint_dir / "COMPLETE").write_text("", encoding="utf-8")
    return run_dir


@pytest.fixture()
def study(tmp_path: Path) -> dict[str, Any]:
    manifest = _write_plan(tmp_path)
    for row in manifest["rows"]:
        _materialize_row(tmp_path, manifest, row)
    return {"manifest": manifest, "root": tmp_path}


def _collect(study: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return collect.collect(
        results_root=study["root"],
        plan_attempt_id=PLAN_ATTEMPT,
        collect_attempt_id="C1",
        gate_spec=kwargs.pop("gate_spec", {}),
        gate_spec_source=kwargs.pop("gate_spec_source", "plan_manifest"),
        **kwargs,
    )


def test_complete_rows_collect_clean(study: dict[str, Any]) -> None:
    """A finished, correctly pinned row with no failing gate passes."""

    collected = _collect(study)
    assert collected["n_rows"] == len(study["manifest"]["rows"])
    assert collected["n_fail"] == 0, [row["reasons"] for row in collected["rows"]]


def test_missing_metric_is_absent_not_zero(study: dict[str, Any]) -> None:
    """A metric nobody emitted must never render as 0.0 or as an empty cell."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    metrics = {k: v for k, v in {**HEALTHY_METRICS, **ENERGY_METRICS}.items()}
    metrics.pop("local_energy_mean")
    _materialize_row(study["root"], manifest, eval_row, metrics=metrics)

    collected = _collect(study)
    row = next(r for r in collected["rows"] if r["identity"]["row_id"] == eval_row["row_id"])
    cell = row["metrics"]["local_energy_mean"]
    assert cell["status"] == "absent"
    assert cell["value"] is None
    assert absence.render(absence.cell_value(cell)) == "absent"


def test_nonfinite_metric_is_absent_not_a_number(study: dict[str, Any]) -> None:
    """NaN is a broken measurement, not a value an aggregate may consume."""

    assert absence.is_absent(absence.present_or_absent(float("nan")))
    assert absence.is_absent(absence.present_or_absent(float("inf")))
    assert absence.present_or_absent(0.0) == 0.0


def test_zero_is_preserved_as_zero(study: dict[str, Any]) -> None:
    """The mirror image: a measured zero must not be mistaken for absence."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    metrics = {**HEALTHY_METRICS, **ENERGY_METRICS, "energy_error": 0.0}
    _materialize_row(study["root"], manifest, eval_row, metrics=metrics)

    collected = _collect(study)
    row = next(r for r in collected["rows"] if r["identity"]["row_id"] == eval_row["row_id"])
    assert row["metrics"]["energy_error"] == {"status": "present", "value": 0.0}
    assert absence.render(0.0) == "0.0"


def test_aggregates_carry_their_coverage(study: dict[str, Any]) -> None:
    """A median over one of two rows may not read like a median over two."""

    manifest = study["manifest"]
    eval_rows = [row for row in manifest["rows"] if row["kind"] == "eval"]
    metrics = {k: v for k, v in {**HEALTHY_METRICS, **ENERGY_METRICS}.items()}
    metrics.pop("local_energy_mean")
    _materialize_row(study["root"], manifest, eval_rows[0], metrics=metrics)

    collected = _collect(study)
    summary = collected["summaries"]["local_energy_mean"]
    assert summary["n_rows"] == len(eval_rows)
    assert summary["n_present"] == len(eval_rows) - 1
    assert summary["n_absent"] == 1


def test_aggregate_over_no_values_is_absent_not_zero() -> None:
    """An aggregate with nothing behind it is absent."""

    summary = absence.summarize_values([absence.ABSENT, None]).to_dict()
    assert summary["mean"] == {"status": "absent", "value": None}
    assert summary["n_present"] == 0


def test_all_gate_metrics_are_retained_whether_or_not_they_gated(study: dict[str, Any]) -> None:
    """The collector keeps every key the gates read, values and evidence alike."""

    collected = _collect(study)
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    for key in collect.GATE_METRIC_KEYS:
        assert key in eval_row["metrics"]
    assert set(collect.GATE_METRIC_KEYS) <= set(collected["metric_keys"])


def test_undeclared_tolerances_report_absent_with_the_value_retained(
    study: dict[str, Any],
) -> None:
    """Pre-H-F1 this is the correct state: absent gates, observed values kept."""

    collected = _collect(study, gate_spec={})
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    by_name = {gate["name"]: gate for gate in eval_row["gates"]}
    value_gate = by_name["cusp_one_sided_slope_abs_error_mean_at_most"]
    assert value_gate["status"] == "absent"
    assert value_gate["value"] == {"status": "present", "value": 0.02}
    assert value_gate["threshold"] == {"status": "absent", "value": None}
    assert collected["gate_spec_declared"] is False


def test_declared_tolerance_decides_the_gate(study: dict[str, Any]) -> None:
    """With a threshold present the same row gates for real."""

    collected = _collect(
        study,
        gate_spec={"nuclear_charge": 2.0, "cusp_one_sided_slope_abs_error_mean_max": 0.01},
    )
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    by_name = {gate["name"]: gate for gate in eval_row["gates"]}
    assert by_name["cusp_one_sided_slope_abs_error_mean_at_most"]["status"] == "fail"
    assert eval_row["status"] == "fail"


def test_unavailable_fit_fails_its_required_flag(study: dict[str, Any]) -> None:
    """An all-absent row is still visibly bad: the availability gate fails."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    metrics = {
        **{k: v for k, v in HEALTHY_METRICS.items() if not k.startswith("cusp_")},
        **ENERGY_METRICS,
        "cusp_available": False,
        "cusp_finite_fit_count": 0,
        "cusp_finite_measurement_count": 0,
    }
    _materialize_row(study["root"], manifest, eval_row, metrics=metrics)
    collected = _collect(study, gate_spec={"require_cusp_available": True})
    row = next(r for r in collected["rows"] if r["identity"]["row_id"] == eval_row["row_id"])
    by_name = {gate["name"]: gate for gate in row["gates"]}
    assert by_name["cusp_available"]["status"] == "fail"
    assert by_name["cusp_one_sided_slope_mean_near_charge"]["status"] == "absent"
    assert row["status"] == "fail"


def test_delivered_stratum_mismatch_fails_the_row(study: dict[str, Any]) -> None:
    """A row that ran on the wrong card is not comparable and is not a footnote."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    _materialize_row(
        study["root"],
        manifest,
        eval_row,
        delivered_device="NVIDIA A100-SXM4-80GB",
        matches=False,
    )
    collected = _collect(study)
    row = next(r for r in collected["rows"] if r["identity"]["row_id"] == eval_row["row_id"])
    assert row["status"] == "fail"
    assert any("does not match requested stratum" in reason for reason in row["reasons"])


def test_missing_allocation_receipt_is_a_failure(study: dict[str, Any]) -> None:
    """Unverified is not verified: no receipt means the card was never checked."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    receipt = launch.row_result_dir(study["root"], eval_row, PLAN_ATTEMPT) / "allocation_receipt.json"
    receipt.unlink()
    collected = _collect(study)
    row = next(r for r in collected["rows"] if r["identity"]["row_id"] == eval_row["row_id"])
    assert row["status"] == "fail"
    assert any("never verified" in reason for reason in row["reasons"])


def test_unfinished_row_is_a_failure(study: dict[str, Any]) -> None:
    """A row with no completed status did not finish, and rows may not resume."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    _materialize_row(study["root"], manifest, eval_row, status="failed")
    collected = _collect(study)
    row = next(r for r in collected["rows"] if r["identity"]["row_id"] == eval_row["row_id"])
    assert row["status"] == "fail"
    assert any("run status is" in reason for reason in row["reasons"])


def test_identity_carries_config_and_checkpoint_hashes(study: dict[str, Any]) -> None:
    """The join keys are content hashes, not paths."""

    collected = _collect(study)
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    identity = eval_row["identity"]
    assert identity["config_sha256"]["status"] == "present"
    assert identity["checkpoint_sha256"]["status"] == "present"
    assert identity["run_id"]["value"] == identity["row_id"]
    assert identity["launch_attempt_id"]["value"] == "L1"


def test_chains_restoring_different_checkpoints_fail_together(study: dict[str, Any]) -> None:
    """Two chains over one checkpoint must have restored the same bytes."""

    manifest = study["manifest"]
    eval_rows = [row for row in manifest["rows"] if row["kind"] == "eval"]
    # Both chains of one (seed, step) share a checkpoint directory, so rewriting
    # it under one chain is exactly the "same step, different bytes" case.
    checkpoint_dir = launch.checkpoint_dir_for_eval_row(
        study["root"], eval_rows[0], PLAN_ATTEMPT, manifest=manifest
    )
    original = collect.directory_sha256(checkpoint_dir)
    (checkpoint_dir / "manifest.json").write_text("other weights", encoding="utf-8")
    assert collect.directory_sha256(checkpoint_dir) != original

    rows = [
        {
            "identity": {
                "row_id": "eval-a",
                "kind": "eval",
                "seed": 0,
                "checkpoint_step": absence.cell(10),
                "checkpoint_sha256": absence.cell("aaa"),
            },
            "status": "pass",
            "reasons": [],
        },
        {
            "identity": {
                "row_id": "eval-b",
                "kind": "eval",
                "seed": 0,
                "checkpoint_step": absence.cell(10),
                "checkpoint_sha256": absence.cell("bbb"),
            },
            "status": "pass",
            "reasons": [],
        },
    ]
    collect.cross_check_checkpoint_identity(rows)
    assert all(row["status"] == "fail" for row in rows)
    assert all("differing checkpoint" in row["reasons"][0] for row in rows)


def test_skipped_checkpoint_hash_is_absent_not_matching(study: dict[str, Any]) -> None:
    """Skipping the hash records absence; it never implies the bytes agreed."""

    collected = _collect(study, hash_checkpoints=False)
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    assert eval_row["identity"]["checkpoint_sha256"] == {"status": "absent", "value": None}
    assert collected["checkpoint_hashing"] is False


def test_duplicate_identities_are_rejected() -> None:
    """Two rows claiming one identity would double-count in every aggregate."""

    row = {
        "identity": {
            "row_id": "eval-a",
            "plan_attempt_id": "P",
            "run_id": absence.cell("run-1"),
            "kind": "eval",
        },
        "status": "pass",
        "reasons": [],
    }
    with pytest.raises(collect.CollectError, match="share identity"):
        collect.require_unique_identities([row, copy.deepcopy(row)])


def test_metrics_logged_under_two_namespaces_are_not_guessed(tmp_path: Path) -> None:
    """Picking one silently would attribute a number to the wrong task."""

    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "namespace": "eval/a", "metrics": {"shared": 1.0}}),
                json.dumps({"step": 0, "namespace": "eval/b", "metrics": {"shared": 2.0}}),
                json.dumps({"step": 0, "namespace": "eval/c", "metrics": {"unique": 3.0}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    namespaced, flat, ambiguous = collect.read_metrics_jsonl(path)
    assert ambiguous == ["shared"]
    assert "shared" not in flat
    assert flat["unique"] == 3.0
    assert namespaced["eval/a/shared"] == 1.0


def test_rows_csv_never_writes_a_blank_cell(study: dict[str, Any]) -> None:
    """A blank parses to NaN and disappears; ``absent`` does not."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    metrics = {k: v for k, v in {**HEALTHY_METRICS, **ENERGY_METRICS}.items()}
    metrics.pop("local_energy_mean")
    _materialize_row(study["root"], manifest, eval_row, metrics=metrics)

    collected = _collect(study)
    directory = collect.write_collected(collected, results_root=study["root"])
    with (directory / collect.ROWS_CSV).open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert table
    for record in table:
        for key, value in record.items():
            assert value != "", f"blank cell for {key}"
    target = next(r for r in table if r["row_id"] == eval_row["row_id"])
    assert target["local_energy_mean"] == "absent"
    train = next(r for r in table if r["kind"] == "train")
    assert train["checkpoint_sha256"] == "absent"


def test_gates_csv_lists_every_outcome(study: dict[str, Any]) -> None:
    """Seventeen outcomes per evaluation row, absent ones included."""

    collected = _collect(study)
    directory = collect.write_collected(collected, results_root=study["root"])
    with (directory / collect.GATES_CSV).open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    eval_rows = [row for row in collected["rows"] if row["identity"]["kind"] == "eval"]
    assert len(table) == 17 * len(eval_rows)
    assert {record["status"] for record in table} <= {"pass", "fail", "absent"}


def test_report_renders_absence_and_coverage(study: dict[str, Any]) -> None:
    """The report says ``absent`` and names its coverage and its estimator."""

    manifest = study["manifest"]
    eval_rows = [row for row in manifest["rows"] if row["kind"] == "eval"]
    metrics = {k: v for k, v in {**HEALTHY_METRICS, **ENERGY_METRICS}.items()}
    metrics.pop("local_energy_mean")
    _materialize_row(study["root"], manifest, eval_rows[0], metrics=metrics)

    collected = _collect(study)
    text = report.render(collected)
    assert "absent" in text
    assert f"{len(eval_rows) - 1}/{len(eval_rows)} rows" in text
    assert "IID stderr (not an MCSE)" in text
    assert "No tolerance was predeclared" in text


def test_report_lists_a_mismatched_row_as_a_failure(study: dict[str, Any]) -> None:
    """A wrong card is in the failures section, not in a footnote."""

    manifest = study["manifest"]
    eval_row = next(row for row in manifest["rows"] if row["kind"] == "eval")
    _materialize_row(
        study["root"], manifest, eval_row, delivered_device="NVIDIA A100-SXM4-80GB", matches=False
    )
    collected = _collect(study)
    text = report.render(collected)
    failures = text.split("## Failures", 1)[1]
    assert eval_row["row_id"] in failures
    assert "does not match requested stratum" in failures


def test_report_round_trips_through_disk(study: dict[str, Any]) -> None:
    """The written report is the rendered table, read back from the collected file."""

    collected = _collect(study)
    collect.write_collected(collected, results_root=study["root"])
    reread = report.read_collected(study["root"], "C1")
    path = report.write_report(reread, results_root=study["root"], attempt_id="R1")
    assert path.read_text(encoding="utf-8") == report.render(collected)
