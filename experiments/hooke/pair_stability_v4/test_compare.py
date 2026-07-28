"""Tests for the fail-closed V4-0 parity comparator."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import compare  # noqa: E402
import control_audit  # noqa: E402
import layout  # noqa: E402
import reference  # noqa: E402
import roots  # noqa: E402
from test_control_audit import _write_valid_control_evidence  # noqa: E402
from test_reference import (  # noqa: E402
    _completed_lineage,
    _v3_reference_source,
)


@pytest.fixture(autouse=True)
def _test_reference_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep reference fixtures in this test's V4-owned temporary namespace."""

    monkeypatch.setattr(reference, "REFERENCE_OWNER_ROOT", tmp_path.absolute())


def test_content_self_compare_and_public_acceptance_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the private content boundary permits self-comparison."""

    source, source_attempts = _v3_reference_source(
        tmp_path / "source",
        monkeypatch,
        lineage="lineage-a",
    )
    fixture_specs = tuple(
        SimpleNamespace(
            logical_path=compare._logical_path(
                path.relative_to(source),
                source_attempts,
            ),
            media_type=reference._media_type(path.relative_to(source)),
        )
        for path in reference.enumerate_inventory(
            source,
            attempts=source_attempts,
        )
    )
    map_path = _write_exact_fixture_map(
        tmp_path / "fixture-map.json",
        fixture_specs,
    )
    monkeypatch.setattr(
        reference,
        "COMPARISON_LAYOUT_PATH",
        map_path,
    )
    monkeypatch.setattr(
        reference,
        "COMPARISON_LAYOUT_LOGICAL_PATH",
        "fixture-map.json",
    )
    monkeypatch.setattr(
        compare,
        "COMPARISON_LAYOUT_LOGICAL_PATH",
        "fixture-map.json",
    )
    monkeypatch.setattr(compare, "REPO_ROOT", tmp_path)
    frozen = reference.freeze_reference(
        source,
        (tmp_path / "frozen-reference").absolute(),
        attempts=source_attempts,
    )
    descriptor, artifacts = reference.load_reference(frozen)
    fixture_layout = layout.load_layout_map(map_path)
    source_paths = {
        artifact.logical_path: source / artifact.source_path
        for artifact in artifacts
    }
    assert descriptor["comparison_contract"]["layout_map_sha256"] == (
        fixture_layout.sha256
    )

    assert (
        compare._compare_reference_contents(
            descriptor,
            artifacts,
            source,
            source_paths,
            candidate_attempts=source_attempts,
            layout=fixture_layout,
        )
        == ()
    )

    with pytest.raises(ValueError, match="same root"):
        compare.compare_reference(
            frozen,
            source,
            candidate_attempts=source_attempts,
            layout_map_path=map_path,
        )

    candidate, candidate_attempts = _completed_lineage(
        tmp_path / "candidate",
        lineage="lineage-b",
    )
    ungated = compare.compare_reference(
        frozen,
        candidate,
        candidate_attempts=candidate_attempts,
        layout_map_path=map_path,
    )
    assert ungated and all(item.kind == "control_audit" for item in ungated)
    runtime_local = dict(descriptor["runtime_closure"])
    runtime_fanout = {
        **runtime_local,
        "python_executable": "/tmp/.venv-submitit/bin/python",
        "uv_project_environment": "/tmp/.venv-submitit",
    }
    _write_valid_control_evidence(
        candidate,
        candidate_attempts,
        source_receipts=(
            dict(descriptor["legacy_closure"]),
            runtime_local,
            runtime_fanout,
            dict(descriptor["config_closure"]),
        ),
    )
    with pytest.raises(ValueError, match="same lineage"):
        compare.compare_reference(
            frozen,
            candidate,
            candidate_attempts=source_attempts,
            layout_map_path=map_path,
        )

    assert (
        compare.compare_reference(
            frozen,
            candidate,
            candidate_attempts=candidate_attempts,
            layout_map_path=map_path,
        )
        == ()
    )
    manual_candidate, manual_attempts = _completed_lineage(
        tmp_path / "manual-candidate",
        lineage="lineage-manual",
    )
    default_manual = compare.compare_reference(
        frozen,
        manual_candidate,
        candidate_attempts=manual_attempts,
        layout_map_path=map_path,
    )
    assert default_manual and all(
        item.kind == "control_audit" for item in default_manual
    )
    assert (
        compare.compare_reference(
            frozen,
            manual_candidate,
            candidate_attempts=manual_attempts,
            layout_map_path=map_path,
            comparison_mode=compare.FROZEN_V3_MANUAL_COMPARISON_MODE,
        )
        == ()
    )
    manual_provenance = compare.comparison_provenance(
        frozen,
        manual_candidate,
        candidate_attempts=manual_attempts,
        layout_map_path=map_path,
        comparison_mode=compare.FROZEN_V3_MANUAL_COMPARISON_MODE,
    )
    assert manual_provenance["comparison_mode"] == "frozen-v3-manual"
    assert manual_provenance["candidate"]["control"][
        "verification_status"
    ] == "operator_trusted_manual"
    assert manual_provenance["candidate"]["control"][
        "controller_closure"
    ] == "not_checked_operator_trusted_manual"
    manual_report = compare.write_comparison_report(
        manual_candidate,
        "manual-comparison",
        (),
        provenance=manual_provenance,
        comparison_mode=compare.FROZEN_V3_MANUAL_COMPARISON_MODE,
    )
    manual_payload = json.loads(manual_report.read_text())
    assert manual_payload["comparison_mode"] == "frozen-v3-manual"
    assert manual_payload["provenance"]["comparison_mode"] == (
        "frozen-v3-manual"
    )
    manual_grid = (
        manual_candidate
        / "00_grid"
        / manual_attempts["grid"]
        / "manifest.json"
    )
    manual_grid_payload = json.loads(manual_grid.read_text())
    manual_grid_payload["n_jobs"] = 0
    manual_grid.write_text(json.dumps(manual_grid_payload))
    manual_audit_differences = compare.compare_reference(
        frozen,
        manual_candidate,
        candidate_attempts=manual_attempts,
        layout_map_path=map_path,
        comparison_mode=compare.FROZEN_V3_MANUAL_COMPARISON_MODE,
    )
    assert manual_audit_differences and all(
        item.kind == "candidate_audit" for item in manual_audit_differences
    )
    closure_candidate, closure_attempts = _completed_lineage(
        tmp_path / "closure-candidate",
        lineage="lineage-c",
    )
    changed_config = json.loads(json.dumps(descriptor["config_closure"]))
    changed_config["files"][0]["sha256"] = "0" * 64
    changed_config["closure_sha256"] = control_audit._receipt_digest(
        changed_config["files"]
    )
    _write_valid_control_evidence(
        closure_candidate,
        closure_attempts,
        source_receipts=(
            dict(descriptor["legacy_closure"]),
            dict(descriptor["runtime_closure"]),
            runtime_fanout,
            changed_config,
        ),
    )
    closure_differences = compare.compare_reference(
        frozen,
        closure_candidate,
        candidate_attempts=closure_attempts,
        layout_map_path=map_path,
    )
    assert closure_differences and all(
        item.kind == "closure" for item in closure_differences
    )
    provenance = compare.comparison_provenance(
        frozen,
        candidate,
        candidate_attempts=candidate_attempts,
        layout_map_path=map_path,
    )
    assert provenance["layout_map"]["sha256"] == fixture_layout.sha256
    assert provenance["reference"]["artifact_count"] == len(artifacts)
    assert provenance["candidate"]["artifact_count"] == len(artifacts)
    assert provenance["candidate"]["evidence_status"] == "verified"
    assert set(provenance["candidate"]["slurm_evidence"]["stages"]) == {
        "01_train",
        "02_validation",
        "06_final_train",
        "07_final_eval",
    }

    summary_path = (
        candidate
        / "03_collect"
        / candidate_attempts["collection"]
        / "summary.csv"
    )
    original_summary = summary_path.read_bytes()
    lines = original_summary.decode().splitlines(keepends=True)
    header = lines[0].rstrip("\r\n").split(",")
    header[1] = header[0]
    lines[0] = ",".join(header) + "\n"
    summary_path.write_text("".join(lines))
    assert compare.compare_reference(
        frozen,
        candidate,
        candidate_attempts=candidate_attempts,
        layout_map_path=map_path,
    )
    summary_path.write_bytes(original_summary)

    lines = original_summary.decode().splitlines(keepends=True)
    lines[1], lines[2] = lines[2], lines[1]
    summary_path.write_text("".join(lines))
    assert compare.compare_reference(
        frozen,
        candidate,
        candidate_attempts=candidate_attempts,
        layout_map_path=map_path,
    )
    summary_path.write_bytes(original_summary)

    changed_metric = _rewrite_csv(
        original_summary,
        lambda rows, fields: rows[0].__setitem__(
            next(
                field
                for field in fields
                if field.startswith("eval/")
                and not field.startswith("eval/perf/")
                and "/status/" not in field
            ),
            "2.01",
        ),
    )
    summary_path.write_bytes(changed_metric)
    assert compare.compare_reference(
        frozen,
        candidate,
        candidate_attempts=candidate_attempts,
        layout_map_path=map_path,
    )
    summary_path.write_bytes(original_summary)

    grid = json.loads(
        (
            candidate
            / "00_grid"
            / candidate_attempts["grid"]
            / "manifest.json"
        ).read_text()
    )
    changed_seed = _rewrite_csv(
        original_summary,
        lambda rows, _fields: rows[0].__setitem__(
            str(grid["scan_seed_axis"]),
            "999",
        ),
    )
    summary_path.write_bytes(changed_seed)
    assert compare.compare_reference(
        frozen,
        candidate,
        candidate_attempts=candidate_attempts,
        layout_map_path=map_path,
    )
    summary_path.write_bytes(original_summary)

    mutated_map = json.loads(map_path.read_text())
    mutated_map["entries"][0]["logical_role"] += "_changed"
    mutated_map_path = tmp_path / "mutated-map.json"
    mutated_map_path.write_text(json.dumps(mutated_map))
    with pytest.raises(ValueError, match="comparison contract"):
        compare.compare_reference(
            frozen,
            candidate,
            candidate_attempts=candidate_attempts,
            layout_map_path=mutated_map_path,
        )

    wrong_root = tmp_path / "v3-root"
    wrong_root.mkdir()
    (wrong_root / ".pair_stability_v4-root.json").write_text(
        json.dumps(
            {
                "schema_version": "pair-stability-v4/root/v1",
                "study": "pair_stability_v3",
                "canonical_root": str(wrong_root),
                "lineage_id": "lineage-c",
                "purpose": "experiment",
                "created_at": "2026-01-01T00:00:00-05:00",
            }
        )
    )
    wrong_attempts = {
        key: "lineage-c" for key in compare.ATTEMPT_KEYS
    }
    with pytest.raises(ValueError, match="sentinel study"):
        compare.compare_reference(
            frozen,
            wrong_root,
            candidate_attempts=wrong_attempts,
            layout_map_path=map_path,
        )


def test_token_approval_is_category_scoped() -> None:
    """Attempt-only policy cannot normalize root, study, or config digests."""

    categories = sorted(layout.TOKEN_SUBSTITUTIONS)
    tokens = {
        side: {category: () for category in categories}
        for side in ("reference", "candidate")
    }
    tokens["reference"]["attempt_ids"] = (("attempt-left", "<ATTEMPT>"),)
    tokens["candidate"]["attempt_ids"] = (("attempt-right", "<ATTEMPT>"),)
    for category, left, right, normalized in (
        ("results_root", "root-left", "root-right", "<ROOT>"),
        ("study_identity", "study-left", "study-right", "<STUDY>"),
        (
            "config_digests",
            "digest-left",
            "digest-right",
            "<DIGEST>",
        ),
    ):
        tokens["reference"][category] = ((left, normalized),)
        tokens["candidate"][category] = ((right, normalized),)

    left = compare._normalize_string(
        "attempt-left root-left study-left digest-left",
        tokens=tokens,
        approved=("attempt_ids",),
        side="reference",
    )
    right = compare._normalize_string(
        "attempt-right root-right study-right digest-right",
        tokens=tokens,
        approved=("attempt_ids",),
        side="candidate",
    )

    assert left == "<ATTEMPT> root-left study-left digest-left"
    assert right == "<ATTEMPT> root-right study-right digest-right"
    assert left != right


def test_layout_contract_has_only_semantic_expansions(
    tmp_path: Path,
) -> None:
    """Media-wide catchalls and wildcard/pattern policies fail validation."""

    assert set(layout.EXPANSION_TOKENS) == {
        "single",
        "scan_runs",
        "final_runs",
    }
    value = {
        "schema_version": layout.LAYOUT_SCHEMA_VERSION,
        "comparator_schema_version": layout.COMPARATOR_SCHEMA_VERSION,
        "tolerances": {"rel_tol": 1e-9, "abs_tol": 1e-12},
        "entries": [
            {
                "logical_role": "bad",
                "reference_logical_path": "<logical_path>",
                "candidate_logical_path": "<logical_path>",
                "expansion": "json_artifacts",
                "format": "json",
                "approved_token_substitutions": [],
                "volatile_json_pointers": ["/created_*"],
                "volatile_csv_columns": [],
                "float_tolerant_json_pointers": [],
                "float_string_tolerant_json_pointers": [],
                "float_tolerant_csv_columns": [],
                "json_record_arrays": [],
                "presence_only": False,
            }
        ],
    }
    target = tmp_path / "invalid-map.json"
    target.write_text(json.dumps(value))

    with pytest.raises(ValueError):
        layout.load_layout_map(target)


def test_json_record_array_is_fail_closed_and_ordered(
    tmp_path: Path,
) -> None:
    """Record policies ignore only named fields and preserve row semantics."""

    loaded, policy = _record_policy(tmp_path)
    left = {
        "overall": "1.0",
        "configs": [
            {
                "config_id": "a",
                "science": "1.0",
                "runtime": "10",
                "science_seed_n": "2",
                "decision": "first",
            },
            {
                "config_id": "b",
                "science": "2.0",
                "runtime": "20",
                "science_seed_n": "2",
                "decision": "second",
            },
        ],
    }
    accepted = json.loads(json.dumps(left))
    accepted["overall"] = "1.0000000001"
    accepted["configs"][0]["science"] = "1.0000000001"
    accepted["configs"][0]["runtime"] = "999"
    assert _structured_differences(left, accepted, loaded, policy) == ()

    mutations = []
    missing = json.loads(json.dumps(left))
    del missing["configs"][0]["science"]
    mutations.append(missing)
    extra = json.loads(json.dumps(left))
    extra["configs"][0]["unexpected"] = "x"
    mutations.append(extra)
    mutations.append({"overall": "1.0", "configs": left["configs"][:1]})
    reordered = json.loads(json.dumps(left))
    reordered["configs"].reverse()
    mutations.append(reordered)
    for field, value in (
        ("science_seed_n", "3"),
        ("config_id", "changed"),
        ("decision", "changed"),
    ):
        changed = json.loads(json.dumps(left))
        changed["configs"][0][field] = value
        mutations.append(changed)

    for changed in mutations:
        assert _structured_differences(left, changed, loaded, policy)


@pytest.mark.parametrize(
    ("left_value", "right_value"),
    [
        ("1.0", "1.01"),
        ("1.0", "-1.0"),
        ("", "1.0"),
        ("NaN", "NaN"),
        ("Inf", "Inf"),
        ("1e9999", "1e9999"),
        ("1.0", 1.0),
    ],
)
def test_float_string_tolerance_rejects_invalid_or_distant_values(
    tmp_path: Path,
    left_value: object,
    right_value: object,
) -> None:
    """Only close, strict, finite decimal strings receive tolerance."""

    loaded, policy = _record_policy(tmp_path)
    left = {
        "overall": "1.0",
        "configs": [
            {
                "config_id": "a",
                "science": left_value,
                "runtime": "1",
                "science_seed_n": "2",
                "decision": "same",
            }
        ],
    }
    right = json.loads(json.dumps(left))
    right["configs"][0]["science"] = right_value

    assert _structured_differences(left, right, loaded, policy)


def test_undeclared_numeric_string_remains_exact(tmp_path: Path) -> None:
    """Numeric-looking strings outside the literal allowlist are exact."""

    loaded, policy = _record_policy(tmp_path)
    left = {
        "overall": "1.0",
        "configs": [
            {
                "config_id": "a",
                "science": "1.0",
                "runtime": "1",
                "science_seed_n": "2",
                "decision": "1.0000000000",
            }
        ],
    }
    right = json.loads(json.dumps(left))
    right["configs"][0]["decision"] = "1.0000000001"

    differences = _structured_differences(left, right, loaded, policy)

    assert len(differences) == 1
    assert differences[0].kind == "value"


def test_csv_policy_preserves_header_order_rows_seeds_and_tolerance(
    tmp_path: Path,
) -> None:
    """CSV comparison streams rows and applies only literal column policy."""

    loaded, policy = _csv_policy(tmp_path)
    left = (
        b"config_id,seed,metric,runtime,count\n"
        b"a,0,1.0,10,2\n"
        b"b,1,2.0,20,2\n"
    )
    accepted = (
        b"config_id,seed,metric,runtime,count\n"
        b"a,0,1.0000000001,999,2\n"
        b"b,1,2.0000000001,888,2\n"
    )
    assert _csv_differences(left, accepted, loaded, policy) == ()

    rejected = (
        b"config_id,seed,metric,runtime,count\n"
        b"a,0,1.01,10,2\n"
        b"b,1,2.0,20,2\n",
        b"config_id,seed,metric,runtime,count\n"
        b"b,1,2.0,20,2\n"
        b"a,0,1.0,10,2\n",
        b"config_id,seed,metric,runtime,count\n"
        b"a,9,1.0,10,2\n"
        b"b,1,2.0,20,2\n",
        b"config_id,seed,metric,runtime,count\n"
        b"a,0,1.0,10,3\n"
        b"b,1,2.0,20,2\n",
        b"config_id,config_id,metric,runtime,count\n"
        b"a,0,1.0,10,2\n"
        b"b,1,2.0,20,2\n",
        b"config_id,seed,metric,runtime,count\n"
        b"a,0,NaN,10,2\n"
        b"b,1,2.0,20,2\n",
        b"config_id,seed,metric,runtime,count\n"
        b"a,0,Inf,10,2\n"
        b"b,1,2.0,20,2\n",
    )
    for right in rejected:
        assert _csv_differences(left, right, loaded, policy)


def test_streamed_gzip_reference_matches_raw_candidate(
    tmp_path: Path,
) -> None:
    """Logical bytes compare identically across frozen gzip/raw storage."""

    loaded, policy = _csv_policy(tmp_path)
    raw = b"config_id,seed,metric,runtime,count\na,0,1.0,10,2\n"
    stored = tmp_path / "stored.csv.gz"
    with stored.open("wb") as binary:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=binary,
            compresslevel=9,
            mtime=0,
        ) as handle:
            handle.write(raw)
    candidate = tmp_path / "candidate.csv"
    candidate.write_bytes(raw)
    artifact = reference.ReferenceArtifact(
        logical_role="table",
        logical_path="table.csv",
        source_path="table.csv",
        stored_path=stored.name,
        media_type="text/csv",
        encoding="gzip",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        stored_sha256=hashlib.sha256(stored.read_bytes()).hexdigest(),
        raw_size=len(raw),
        stored_size=stored.stat().st_size,
        reference_dir=tmp_path,
    )
    sink = compare._DifferenceSink()
    compare._compare_artifact(
        artifact,
        candidate,
        policy=policy,
        layout=loaded,
        tokens=_empty_tokens(),
        sink=sink,
    )
    assert sink.values == []


def test_json_key_order_is_semantic_but_jsonl_row_order_is_exact(
    tmp_path: Path,
) -> None:
    """Object key order is ignored while append-only record order is not."""

    loaded, policy = _record_policy(tmp_path)
    left = {
        "overall": "1",
        "configs": [
            {
                "config_id": "a",
                "science": "1",
                "runtime": "1",
                "science_seed_n": "2",
                "decision": "same",
            }
        ],
    }
    reordered_keys = {
        "configs": [
            {
                "decision": "same",
                "science_seed_n": "2",
                "runtime": "1",
                "science": "1",
                "config_id": "a",
            }
        ],
        "overall": "1",
    }
    assert (
        _structured_differences(left, reordered_keys, loaded, policy)
        == ()
    )

    jsonl_layout, jsonl_policy = _jsonl_policy(tmp_path)
    left_rows = b'{"id":"a","value":1}\n{"id":"b","value":2}\n'
    right_rows = b'{"id":"b","value":2}\n{"id":"a","value":1}\n'
    sink = compare._DifferenceSink()
    compare._compare_jsonl_streams(
        io.BytesIO(left_rows),
        io.BytesIO(right_rows),
        policy=jsonl_policy,
        layout=jsonl_layout,
        tokens=_empty_tokens(),
        sink=sink,
    )
    assert sink.values


def test_layout_materialization_rejects_missing_extra_and_duplicate_entries(
    tmp_path: Path,
) -> None:
    """The static mapping must remain a concrete one-to-one bijection."""

    loaded, _policy = _csv_policy(tmp_path)
    with pytest.raises(ValueError, match="inventory mismatch"):
        layout.materialize_layout(
            loaded,
            expansions={},
            reference_paths={"table.csv"},
            candidate_paths={"table.csv", "extra.csv"},
        )
    with pytest.raises(ValueError, match="inventory mismatch"):
        layout.materialize_layout(
            loaded,
            expansions={},
            reference_paths=set(),
            candidate_paths={"table.csv"},
        )

    value = _record_map_value()
    value["entries"].append(dict(value["entries"][0]))
    target = tmp_path / "duplicate-entry.json"
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicated"):
        layout.load_layout_map(target)


def test_difference_report_is_bounded_and_records_provenance(
    tmp_path: Path,
) -> None:
    """Machine-readable output has a deterministic outcome and hard bound."""

    sink = compare._DifferenceSink(limit=3)
    for index in range(8):
        sink.add(
            artifact=f"artifact-{index}",
            location="/",
            kind="value",
            message="changed",
            reference="x" * 500,
            candidate="y" * 500,
        )
    assert len(sink.values) == 3
    assert sink.values[-1].kind == "truncated"
    assert len(sink.values[0].reference) <= compare.MAX_VALUE_CHARS

    candidate = roots.initialize_root(
        (tmp_path / "candidate").absolute(),
        lineage_id="lineage-a",
    )
    destination = compare.write_comparison_report(
        candidate,
        "comparison-a",
        sink.values,
        provenance={"layout_map_sha256": "a" * 64},
    )
    report = json.loads(destination.read_text())
    assert report["schema_version"] == layout.COMPARATOR_SCHEMA_VERSION
    assert report["comparison_mode"] == compare.CANONICAL_COMPARISON_MODE
    assert report["outcome"] == "failed"
    assert report["n_differences"] == 3
    assert report["provenance"] == {
        "layout_map_sha256": "a" * 64,
        "comparison_mode": compare.CANONICAL_COMPARISON_MODE,
    }
    with pytest.raises(FileExistsError, match="already exists"):
        compare.write_comparison_report(
            candidate,
            "comparison-a",
            sink.values,
            provenance={"layout_map_sha256": "a" * 64},
        )
    with pytest.raises(ValueError, match="lineage/attempt ids"):
        compare.write_comparison_report(
            candidate,
            "../escape",
            sink.values,
            provenance={"layout_map_sha256": "a" * 64},
        )
    with pytest.raises(ValueError, match="disagrees with provenance"):
        compare.write_comparison_report(
            candidate,
            "comparison-manual-mismatch",
            sink.values,
            provenance={"comparison_mode": "frozen-v3-manual"},
        )
    escaped = candidate / "_v4" / "comparison" / "escaped"
    escaped.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes v4 root"):
        compare.write_comparison_report(
            candidate,
            "escaped",
            sink.values,
            provenance={"layout_map_sha256": "a" * 64},
        )


def test_compare_cli_exit_codes_are_zero_one_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success, parity mismatch, and usage/operational errors stay distinct."""

    attempts = json.dumps(
        {key: "lineage-b" for key in compare.ATTEMPT_KEYS}
    )
    argv = [
        "--reference",
        "/tmp/reference",
        "--candidate-root",
        "/tmp/candidate",
        "--candidate-attempts",
        attempts,
    ]
    monkeypatch.setattr(compare, "compare_reference", lambda *a, **k: ())
    assert compare.main(argv) == 0

    captured: dict[str, object] = {}

    def manual_success(*_args: object, **kwargs: object) -> tuple[()]:
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(compare, "compare_reference", manual_success)
    assert compare.main(
        [
            *argv,
            "--comparison-mode",
            compare.FROZEN_V3_MANUAL_COMPARISON_MODE,
        ]
    ) == 0
    assert captured["comparison_mode"] == "frozen-v3-manual"

    difference = compare.Difference(
        artifact="table.csv",
        location="/rows/1/metric",
        kind="float",
        message="changed",
    )
    monkeypatch.setattr(
        compare,
        "compare_reference",
        lambda *a, **k: (difference,),
    )
    assert compare.main(argv) == 1

    with pytest.raises(SystemExit) as exc_info:
        compare.main([*argv[:-1], "{not-json"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("mutation", ["duplicate", "overlap", "unknown"])
def test_json_record_array_map_rejects_ambiguous_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Record declarations are literal, unique, and unambiguous."""

    raw = _record_map_value()
    record = raw["entries"][0]["json_record_arrays"][0]
    if mutation == "duplicate":
        record["volatile_fields"] = ["runtime", "runtime"]
    elif mutation == "overlap":
        record["volatile_fields"] = ["runtime", "science"]
    else:
        record["unknown_policy"] = []
    target = tmp_path / f"{mutation}.json"
    target.write_text(json.dumps(raw))

    with pytest.raises(ValueError):
        layout.load_layout_map(target)


def test_json_record_array_absent_field_and_pointer_fail(
    tmp_path: Path,
) -> None:
    """Every declared array and governed field must be exercised."""

    loaded, policy = _record_policy(tmp_path)
    missing_field = {
        "overall": "1.0",
        "configs": [
            {
                "config_id": "a",
                "runtime": "1",
                "science_seed_n": "2",
                "decision": "same",
            }
        ],
    }
    missing_array = {"overall": "1.0", "other": []}

    assert any(
        difference.kind == "policy"
        for difference in _structured_differences(
            missing_field,
            missing_field,
            loaded,
            policy,
        )
    )
    assert any(
        difference.kind == "policy"
        for difference in _structured_differences(
            missing_array,
            missing_array,
            loaded,
            policy,
        )
    )


def test_default_selection_policy_is_literal_and_complete() -> None:
    """The checked-in report partition is static and preserves exact fields."""

    loaded = layout.load_layout_map()
    assert len(loaded.entries) == 87
    entry = next(
        item
        for item in loaded.entries
        if item.reference_logical_path
        == "04_select/{selection}/selection_report.json"
    )
    assert entry.float_string_tolerant_json_pointers == (
        "/overall_metric_value",
        "/secondary_metric_value",
    )
    assert tuple(
        policy.array_pointer for policy in entry.json_record_arrays
    ) == ("/champions", "/configs")
    champions, configs = entry.json_record_arrays
    assert champions.volatile_fields == ()
    assert len(champions.float_string_tolerant_fields) == 21
    assert len(configs.volatile_fields) == 168
    assert len(configs.float_string_tolerant_fields) == 207
    exact_fields = {
        "config_id",
        "n_diagnostics_seed_median",
        "runtime/cuda_device_count_seed_mean",
        "runtime/cuda_device_count_seed_median",
        "runtime/cuda_device_count_seed_stderr",
    }
    governed = {
        *configs.volatile_fields,
        *configs.float_string_tolerant_fields,
    }
    assert exact_fields.isdisjoint(governed)
    assert all(not field.endswith("_seed_n") for field in governed)


def _write_exact_fixture_map(
    destination: Path,
    artifacts: tuple[object, ...],
) -> Path:
    format_by_media = {
        "application/json": "json",
        "application/x-ndjson": "jsonl",
        "application/yaml": "yaml",
        "text/csv": "csv",
        "text/markdown": "markdown",
        "text/x-shellscript": "text",
        "text/plain": "text",
    }
    entries = []
    for index, artifact in enumerate(artifacts):
        volatile_pointers: list[str] = []
        if artifact.logical_path == "00_grid/{grid}/manifest.json":
            volatile_pointers.append("/created_at")
        if artifact.logical_path.endswith("/execution_records.jsonl"):
            volatile_pointers.append("/launcher_job_id")
        if artifact.logical_path.endswith("/submission.json"):
            volatile_pointers.append("/launcher_job_id")
        entries.append(
            {
                "logical_role": f"fixture_artifact_{index:04d}",
                "reference_logical_path": artifact.logical_path,
                "candidate_logical_path": artifact.logical_path,
                "expansion": "single",
                "format": format_by_media[artifact.media_type],
                "approved_token_substitutions": sorted(
                    layout.TOKEN_SUBSTITUTIONS
                ),
                "volatile_json_pointers": sorted(volatile_pointers),
                "volatile_csv_columns": [],
                "float_tolerant_json_pointers": [],
                "float_string_tolerant_json_pointers": [],
                "float_tolerant_csv_columns": [],
                "json_record_arrays": [],
                "presence_only": False,
            }
        )
    destination.write_text(
        json.dumps(
            {
                "schema_version": layout.LAYOUT_SCHEMA_VERSION,
                "comparator_schema_version": layout.COMPARATOR_SCHEMA_VERSION,
                "tolerances": {"rel_tol": 1e-9, "abs_tol": 1e-12},
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def _record_map_value() -> dict[str, object]:
    return {
        "schema_version": layout.LAYOUT_SCHEMA_VERSION,
        "comparator_schema_version": layout.COMPARATOR_SCHEMA_VERSION,
        "tolerances": {"rel_tol": 1e-9, "abs_tol": 1e-12},
        "entries": [
            {
                "logical_role": "selection",
                "reference_logical_path": "report.json",
                "candidate_logical_path": "report.json",
                "expansion": "single",
                "format": "json",
                "approved_token_substitutions": [],
                "volatile_json_pointers": [],
                "volatile_csv_columns": [],
                "float_tolerant_json_pointers": [],
                "float_string_tolerant_json_pointers": ["/overall"],
                "float_tolerant_csv_columns": [],
                "json_record_arrays": [
                    {
                        "array_pointer": "/configs",
                        "volatile_fields": ["runtime"],
                        "float_string_tolerant_fields": ["science"],
                    }
                ],
                "presence_only": False,
            }
        ],
    }


def _record_policy(
    tmp_path: Path,
) -> tuple[layout.LayoutMap, layout.ArtifactPolicy]:
    target = tmp_path / "record-map.json"
    target.write_text(json.dumps(_record_map_value()))
    loaded = layout.load_layout_map(target)
    policies = layout.materialize_layout(
        loaded,
        expansions={},
        reference_paths={"report.json"},
        candidate_paths={"report.json"},
    )
    return loaded, policies[0]


def _csv_policy(
    tmp_path: Path,
) -> tuple[layout.LayoutMap, layout.ArtifactPolicy]:
    value = _record_map_value()
    entry = value["entries"][0]
    entry.update(
        {
            "logical_role": "table",
            "reference_logical_path": "table.csv",
            "candidate_logical_path": "table.csv",
            "format": "csv",
            "volatile_json_pointers": [],
            "volatile_csv_columns": ["runtime"],
            "float_tolerant_json_pointers": [],
            "float_string_tolerant_json_pointers": [],
            "float_tolerant_csv_columns": ["metric"],
            "json_record_arrays": [],
        }
    )
    target = tmp_path / "csv-map.json"
    target.write_text(json.dumps(value))
    loaded = layout.load_layout_map(target)
    policy = layout.materialize_layout(
        loaded,
        expansions={},
        reference_paths={"table.csv"},
        candidate_paths={"table.csv"},
    )[0]
    return loaded, policy


def _jsonl_policy(
    tmp_path: Path,
) -> tuple[layout.LayoutMap, layout.ArtifactPolicy]:
    value = _record_map_value()
    entry = value["entries"][0]
    entry.update(
        {
            "logical_role": "events",
            "reference_logical_path": "events.jsonl",
            "candidate_logical_path": "events.jsonl",
            "format": "jsonl",
            "volatile_json_pointers": [],
            "volatile_csv_columns": [],
            "float_tolerant_json_pointers": [],
            "float_string_tolerant_json_pointers": [],
            "float_tolerant_csv_columns": [],
            "json_record_arrays": [],
        }
    )
    target = tmp_path / "jsonl-map.json"
    target.write_text(json.dumps(value))
    loaded = layout.load_layout_map(target)
    policy = layout.materialize_layout(
        loaded,
        expansions={},
        reference_paths={"events.jsonl"},
        candidate_paths={"events.jsonl"},
    )[0]
    return loaded, policy


def _structured_differences(
    left: object,
    right: object,
    loaded: layout.LayoutMap,
    policy: layout.ArtifactPolicy,
) -> tuple[compare.Difference, ...]:
    sink = compare._DifferenceSink()
    compare._compare_structured_value(
        left,
        right,
        artifact="report.json",
        policy=policy,
        layout=loaded,
        tokens=_empty_tokens(),
        sink=sink,
    )
    return tuple(sink.values)


def _csv_differences(
    left: bytes,
    right: bytes,
    loaded: layout.LayoutMap,
    policy: layout.ArtifactPolicy,
) -> tuple[compare.Difference, ...]:
    sink = compare._DifferenceSink()
    compare._compare_csv_streams(
        io.BytesIO(left),
        io.BytesIO(right),
        policy=policy,
        layout=loaded,
        tokens=_empty_tokens(),
        sink=sink,
    )
    return tuple(sink.values)


def _empty_tokens() -> dict[str, dict[str, tuple[object, ...]]]:
    return {
        side: {
            category: ()
            for category in sorted(layout.TOKEN_SUBSTITUTIONS)
        }
        for side in ("reference", "candidate")
    }


def _rewrite_csv(
    original: bytes,
    mutate,
) -> bytes:
    source = io.StringIO(original.decode(), newline="")
    reader = csv.DictReader(source)
    fields = list(reader.fieldnames or ())
    rows = list(reader)
    mutate(rows, fields)
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode()
