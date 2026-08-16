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

a metric name is not a metric
    ``full_model_antisymmetry`` and ``spatial_exchange_symmetry`` share
    ``TransformConsistencySummary``, so four metric names exist twice per
    evaluation row and mean different things under each task. A request may name
    its namespace; an unqualified request for a colliding name still fails and
    names the namespaces; and a collision nobody asked about is not a failure at
    all. The measured regression these tests pin: at merged dev every one of the
    three smoke rows failed collection on collisions in metrics nothing had
    requested.
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

#: The four names ``TransformConsistencySummary`` emits under BOTH
#: ``eval/full_model_antisymmetry`` and ``eval/spatial_exchange_symmetry`` in the
#: real He evaluation config. Values are the ones the 2026-08-15 smoke measured
#: at its 25-step checkpoint. Under full label exchange the triplet fraction is
#: identically 1.0 by construction -- ``Psi -> -Psi`` gives ``u = 0`` and sign
#: ratio ``-1``, so ``f = (1 - s*sech(u))/2 = 1`` -- and that is the healthy
#: value, not contamination. Only the spatial-exchange numbers are singlet
#: purity.
TRANSFORM_CONSISTENCY_KEYS: tuple[str, ...] = (
    "logabs_max_abs_error",
    "logabs_mean_abs_error",
    "triplet_fraction_max_under_psi_orig_sq",
    "triplet_fraction_mean_under_psi_orig_sq",
)

FULL_MODEL_METRICS: dict[str, Any] = {
    "logabs_max_abs_error": 0.0,
    "logabs_mean_abs_error": 0.0,
    "triplet_fraction_max_under_psi_orig_sq": 1.0,
    "triplet_fraction_mean_under_psi_orig_sq": 1.0,
    "sign_failure_count": 0,
}

SPATIAL_EXCHANGE_METRICS: dict[str, Any] = {
    "logabs_max_abs_error": 0.6049,
    "logabs_mean_abs_error": 0.1908,
    "triplet_fraction_max_under_psi_orig_sq": 0.07934,
    "triplet_fraction_mean_under_psi_orig_sq": 0.01506,
    "sign_failure_count": 0,
}

#: The two colliding evaluation namespaces, as the real config names them.
FULL_MODEL_NS = "eval/full_model_antisymmetry"
SPATIAL_NS = "eval/spatial_exchange_symmetry"

#: What the real eval rows log: the two tasks above, sharing one summary class.
TRANSFORM_NAMESPACES: dict[str, dict[str, Any]] = {
    FULL_MODEL_NS: FULL_MODEL_METRICS,
    SPATIAL_NS: SPATIAL_EXCHANGE_METRICS,
}


def qualified(namespace: str, key: str) -> str:
    """Return the collector's qualified form of one metric request."""

    return f"{namespace}{collect.NAMESPACE_SEPARATOR}{key}"


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
    extra_namespaces: dict[str, dict[str, Any]] | None = None,
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
    # Extra namespaces reproduce the real config's shape: several tasks logging
    # into one metrics.jsonl, some of them sharing metric NAMES.
    for namespace, payload_metrics in (extra_namespaces or {}).items():
        lines.append(
            json.dumps({"step": 0, "namespace": namespace, "metrics": dict(payload_metrics)})
        )
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
    # The namespaces travel with the collision, so a failure can name them.
    assert ambiguous == {"shared": ["eval/a", "eval/b"]}
    assert "shared" not in flat
    assert flat["unique"] == 3.0
    assert namespaced["eval/a/shared"] == 1.0


def test_one_name_logged_twice_with_one_value_is_not_ambiguous(tmp_path: Path) -> None:
    """Two namespaces agreeing on a value leave nothing to disambiguate."""

    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "namespace": "eval/a", "metrics": {"agreed": 7.0}}),
                json.dumps({"step": 0, "namespace": "eval/b", "metrics": {"agreed": 7.0}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _namespaced, flat, ambiguous = collect.read_metrics_jsonl(path)
    assert ambiguous == {}
    assert flat["agreed"] == 7.0


@pytest.fixture()
def transform_study(tmp_path: Path) -> dict[str, Any]:
    """A study whose evaluation rows log the real config's colliding tasks.

    Both transform tasks share ``TransformConsistencySummary``, so every one of
    ``TRANSFORM_CONSISTENCY_KEYS`` exists twice per evaluation row with
    different values. The train row logs neither task, exactly as a training
    run does not run the evaluation stack.
    """

    manifest = _write_plan(tmp_path)
    for row in manifest["rows"]:
        _materialize_row(
            tmp_path,
            manifest,
            row,
            extra_namespaces=TRANSFORM_NAMESPACES if row["kind"] == "eval" else None,
        )
    return {"manifest": manifest, "root": tmp_path}


def test_unrequested_collision_does_not_fail_the_row(transform_study: dict[str, Any]) -> None:
    """The measured regression: at merged dev this failed all three smoke rows.

    Four names collide on every evaluation row because two tasks share a summary
    class. Nothing requested them. A collector that fails the row anyway collects
    zero usable rows from a study whose physics was sound.
    """

    collected = _collect(transform_study)
    assert collected["n_fail"] == 0, [row["reasons"] for row in collected["rows"]]
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    # The collision is still reported -- as diagnostics, not as a verdict.
    assert eval_row["ambiguous_metric_keys"] == sorted(TRANSFORM_CONSISTENCY_KEYS)
    assert eval_row["ambiguous_metric_namespaces"]["logabs_max_abs_error"] == [
        FULL_MODEL_NS,
        SPATIAL_NS,
    ]


def test_qualified_key_resolves_to_exactly_one_namespace(
    transform_study: dict[str, Any],
) -> None:
    """Both sides of the collision are collectable, each under its own name."""

    spatial = qualified(SPATIAL_NS, "triplet_fraction_mean_under_psi_orig_sq")
    full_model = qualified(FULL_MODEL_NS, "triplet_fraction_mean_under_psi_orig_sq")
    collected = _collect(transform_study, extra_metric_keys=[spatial, full_model])
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    assert eval_row["metrics"][spatial] == {"status": "present", "value": 0.01506}
    # 1.0 under full label exchange is definitional for an antisymmetric wave
    # function, not contamination; it is retained under its own name so it can
    # never be mistaken for the singlet-purity number above.
    assert eval_row["metrics"][full_model] == {"status": "present", "value": 1.0}
    assert eval_row["status"] == "pass"


def test_all_four_colliding_config_keys_collect_under_qualification(
    transform_study: dict[str, Any],
) -> None:
    """Every key the real He config collides on is collectable from both tasks."""

    requests = [
        qualified(namespace, key)
        for namespace in (FULL_MODEL_NS, SPATIAL_NS)
        for key in TRANSFORM_CONSISTENCY_KEYS
    ]
    collected = _collect(transform_study, extra_metric_keys=requests)
    assert collected["n_fail"] == 0, [row["reasons"] for row in collected["rows"]]
    eval_rows = [row for row in collected["rows"] if row["identity"]["kind"] == "eval"]
    for eval_row in eval_rows:
        for key in TRANSFORM_CONSISTENCY_KEYS:
            assert eval_row["metrics"][qualified(FULL_MODEL_NS, key)] == {
                "status": "present",
                "value": FULL_MODEL_METRICS[key],
            }
            assert eval_row["metrics"][qualified(SPATIAL_NS, key)] == {
                "status": "present",
                "value": SPATIAL_EXCHANGE_METRICS[key],
            }
    # Coverage is per qualified column, so the two tasks aggregate separately.
    for key in TRANSFORM_CONSISTENCY_KEYS:
        summary = collected["summaries"][qualified(SPATIAL_NS, key)]
        assert summary["n_present"] == len(eval_rows)
        assert summary["mean"]["value"] == SPATIAL_EXCHANGE_METRICS[key]


def test_unqualified_colliding_request_still_fails_and_names_its_namespaces(
    transform_study: dict[str, Any],
) -> None:
    """The refusal to guess survives: only the way to express intent is new."""

    collected = _collect(
        transform_study,
        extra_metric_keys=["triplet_fraction_mean_under_psi_orig_sq"],
    )
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    assert eval_row["status"] == "fail"
    reason = next(
        reason
        for reason in eval_row["reasons"]
        if "triplet_fraction_mean_under_psi_orig_sq" in reason
    )
    assert FULL_MODEL_NS in reason and SPATIAL_NS in reason
    assert eval_row["metrics"]["triplet_fraction_mean_under_psi_orig_sq"] == {
        "status": "absent",
        "value": None,
    }


def test_qualified_key_matching_no_namespace_is_a_failure(
    transform_study: dict[str, Any],
) -> None:
    """A mis-typed namespace must not render as a column of honest absence."""

    with pytest.raises(collect.CollectError, match="namespaces no row logged"):
        _collect(
            transform_study,
            extra_metric_keys=[
                qualified("eval/spatial_exchange_symetry", "logabs_mean_abs_error")
            ],
        )


def test_qualified_key_missing_under_a_logged_namespace_fails_that_row(
    transform_study: dict[str, Any],
) -> None:
    """The task ran and the metric is still not there: that is not absence."""

    collected = _collect(
        transform_study,
        extra_metric_keys=[qualified(SPATIAL_NS, "triplet_fraction_p95")],
    )
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    assert eval_row["status"] == "fail"
    assert any("emitted no 'triplet_fraction_p95'" in reason for reason in eval_row["reasons"])


def test_eval_only_qualified_key_leaves_the_train_row_absent_not_failed(
    transform_study: dict[str, Any],
) -> None:
    """A train row runs no evaluation task; an eval column may not fail it."""

    request = qualified(SPATIAL_NS, "triplet_fraction_mean_under_psi_orig_sq")
    collected = _collect(transform_study, extra_metric_keys=[request])
    train_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "train")
    assert train_row["status"] == "pass", train_row["reasons"]
    assert train_row["metrics"][request] == {"status": "absent", "value": None}


def test_gate_spec_namespace_binding_gates_only_that_namespace(tmp_path: Path) -> None:
    """H-F1's mechanism: a tolerance reads one task and never the other.

    ``tail_negative_slope_fraction`` is logged here under two namespaces with
    values that fall on opposite sides of the declared floor, which is the shape
    of the singlet-purity hazard: an unbound tolerance would land on whichever
    namespace happened to be picked, and gate the wrong task's number.
    """

    manifest = _write_plan(tmp_path)
    for row in manifest["rows"]:
        _materialize_row(
            tmp_path,
            manifest,
            row,
            extra_namespaces={"eval/decoy_task": {"tail_negative_slope_fraction": 0.0}},
        )
    study = {"manifest": manifest, "root": tmp_path}

    # Unbound, the colliding name is excluded from the gate view entirely: the
    # gate reports absent rather than gating a guess.
    unbound = _collect(study, gate_spec={"tail_negative_slope_fraction_min": 0.9})
    eval_row = next(row for row in unbound["rows"] if row["identity"]["kind"] == "eval")
    by_name = {gate["name"]: gate for gate in eval_row["gates"]}
    assert by_name["tail_negative_slope_fraction_at_least"]["status"] == "absent"

    # Bound to the healthy task, the same threshold decides on that task's value.
    bound = _collect(
        study,
        gate_spec={
            "tail_negative_slope_fraction_min": 0.9,
            collect.METRIC_NAMESPACE_SPEC_KEY: {
                "tail_negative_slope_fraction": "eval/he_radial_profiles"
            },
        },
    )
    eval_row = next(row for row in bound["rows"] if row["identity"]["kind"] == "eval")
    by_name = {gate["name"]: gate for gate in eval_row["gates"]}
    gate = by_name["tail_negative_slope_fraction_at_least"]
    assert gate["status"] == "pass"
    assert gate["value"] == {"status": "present", "value": 1.0}
    assert bound["metric_namespaces"] == {
        "tail_negative_slope_fraction": "eval/he_radial_profiles"
    }

    # Bound to the decoy, the same threshold fails on the decoy's value. Same
    # spec, same row, different task: the binding is what decides.
    decoyed = _collect(
        study,
        gate_spec={
            "tail_negative_slope_fraction_min": 0.9,
            collect.METRIC_NAMESPACE_SPEC_KEY: {
                "tail_negative_slope_fraction": "eval/decoy_task"
            },
        },
    )
    eval_row = next(row for row in decoyed["rows"] if row["identity"]["kind"] == "eval")
    by_name = {gate["name"]: gate for gate in eval_row["gates"]}
    assert by_name["tail_negative_slope_fraction_at_least"]["status"] == "fail"
    assert by_name["tail_negative_slope_fraction_at_least"]["value"] == {
        "status": "present",
        "value": 0.0,
    }


def test_bindings_alone_are_not_a_declared_tolerance(transform_study: dict[str, Any]) -> None:
    """Saying WHICH namespace a metric comes from declares no threshold."""

    collected = _collect(
        transform_study,
        gate_spec={
            collect.METRIC_NAMESPACE_SPEC_KEY: {
                "tail_negative_slope_fraction": "eval/he_radial_profiles"
            }
        },
    )
    assert collected["gate_spec_declared"] is False
    assert collected["gate_spec"] == {}
    eval_row = next(row for row in collected["rows"] if row["identity"]["kind"] == "eval")
    by_name = {gate["name"]: gate for gate in eval_row["gates"]}
    assert by_name["tail_negative_slope_fraction_at_least"]["status"] == "absent"


def test_gate_binding_is_not_passed_through_to_the_gates() -> None:
    """The gates reject unknown spec keys; the binding must never reach them."""

    thresholds, bindings = collect.split_gate_spec(
        {
            "nuclear_charge": 2.0,
            collect.METRIC_NAMESPACE_SPEC_KEY: {"m": "eval/a"},
        }
    )
    assert thresholds == {"nuclear_charge": 2.0}
    assert bindings == {"m": "eval/a"}
    # The unstripped spec is exactly what gates.py is built to reject.
    with pytest.raises(ValueError, match="unknown atom gate spec keys"):
        collect.gates.evaluate_atom_gates(
            {}, spec={collect.METRIC_NAMESPACE_SPEC_KEY: {"m": "eval/a"}}
        )


def test_malformed_namespace_binding_is_rejected() -> None:
    """A binding that is not '<metric>: <namespace>' is a typo, not a setting."""

    with pytest.raises(collect.CollectError, match="must be a mapping"):
        collect.split_gate_spec({collect.METRIC_NAMESPACE_SPEC_KEY: ["eval/a"]})
    with pytest.raises(collect.CollectError, match="must be a namespace string"):
        collect.split_gate_spec({collect.METRIC_NAMESPACE_SPEC_KEY: {"m": 3}})
    with pytest.raises(collect.CollectError, match="METRIC=NAMESPACE"):
        collect.parse_metric_namespace_arguments(["tail_negative_slope_fraction"])
    assert collect.parse_metric_namespace_arguments(["m=eval/a"]) == {"m": "eval/a"}


def test_qualified_columns_reach_the_csv_and_the_report(
    transform_study: dict[str, Any],
) -> None:
    """A reader must see which task produced a number without the config."""

    spatial = qualified(SPATIAL_NS, "triplet_fraction_mean_under_psi_orig_sq")
    full_model = qualified(FULL_MODEL_NS, "triplet_fraction_mean_under_psi_orig_sq")
    collected = _collect(
        transform_study,
        extra_metric_keys=[spatial, full_model],
        gate_spec={
            collect.METRIC_NAMESPACE_SPEC_KEY: {
                "tail_negative_slope_fraction": "eval/he_radial_profiles"
            }
        },
    )
    directory = collect.write_collected(collected, results_root=transform_study["root"])
    with (directory / collect.ROWS_CSV).open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    eval_record = next(record for record in table if record["kind"] == "eval")
    assert eval_record[spatial] == "0.01506"
    assert eval_record[full_model] == "1.0"
    train_record = next(record for record in table if record["kind"] == "train")
    assert train_record[spatial] == "absent"

    text = report.render(collected)
    assert f"`{spatial}`" in text
    assert f"`{full_model}`" in text
    assert (
        "`tail_negative_slope_fraction` read from namespace `eval/he_radial_profiles`"
        in text
    )


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
    """EVERY declared gate appears per evaluation row, absent ones included.

    The count is derived from the gate table rather than restated, so adding a
    gate cannot leave this test asserting a stale number. It previously read
    ``17``; adding the three singlet-purity gates made that literal wrong while
    the property it was protecting -- every declared gate reaches the CSV --
    remained true.
    """

    collected = _collect(study)
    directory = collect.write_collected(collected, results_root=study["root"])
    with (directory / collect.GATES_CSV).open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    eval_rows = [row for row in collected["rows"] if row["identity"]["kind"] == "eval"]
    # Taken from the subject, for the same reason ABSENT is: a second import of
    # `gates` would be a different module object and could disagree.
    n_gates = len(collect.gates.evaluate_atom_gates({}, spec={}))
    assert n_gates > 0
    assert len(table) == n_gates * len(eval_rows)
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
