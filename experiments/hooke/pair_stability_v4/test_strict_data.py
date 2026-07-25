"""Regression tests for V4-0 unambiguous structured-evidence parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import compare  # noqa: E402
import dispatch  # noqa: E402
import layout  # noqa: E402
import reference  # noqa: E402
import routes  # noqa: E402
from strict_data import StrictDataError, iter_jsonl, loads_json, loads_yaml  # noqa: E402
from test_reference import _v3_reference_source  # noqa: E402


@pytest.mark.parametrize(
    "payload",
    (
        '{"outer":{"x":1,"x":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":1e999}',
    ),
)
def test_json_rejects_nested_duplicates_and_nonfinite_values(payload: str) -> None:
    """Ambiguity fails before any comparator volatility rule can apply."""

    with pytest.raises(StrictDataError):
        loads_json(payload)
    with pytest.raises(StrictDataError):
        compare._strict_json_loads(payload)


def test_yaml_and_jsonl_reject_ambiguous_rows(tmp_path: Path) -> None:
    """All accepted structured formats share duplicate/nonfinite rejection."""

    with pytest.raises(StrictDataError):
        loads_yaml("value: 1\nvalue: 2\n")
    with pytest.raises(StrictDataError):
        loads_yaml("value: .inf\n")
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id":1,"id":2}\n')
    with pytest.raises(StrictDataError):
        tuple(iter_jsonl(path))


def test_dispatch_manifest_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Live guarded config/manifest intake is strict, not only comparison."""

    path = tmp_path / "manifest.json"
    path.write_text('{"jobs":[],"jobs":[]}\n')
    with pytest.raises(ValueError, match="invalid JSON object"):
        dispatch._read_json_object(path)


def test_config_copy_verifier_rejects_duplicate_yaml_before_semantic_compare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public V4 config receipt path cannot accept duplicate YAML keys."""

    v3 = tmp_path / "pair_stability_v3"
    v4 = tmp_path / "pair_stability_v4"
    for root in (v3, v4):
        (root / "configs").mkdir(parents=True)
    smoke = (
        "study: pair_stability_v3\n"
        "config: experiments/hooke/pair_stability_v3/configs/pair_stability.yaml\n"
        "validation_config: experiments/hooke/pair_stability_v3/configs/pair_validation.yaml\n"
        "results_root: experiments/hooke/pair_stability_v3/results\n"
    )
    model = (
        "study:\n  name: pair_stability_v3\n"
        "run:\n  root: experiments/hooke/pair_stability_v3/results/01_train\n"
    )
    validation = model.replace("01_train", "02_validation")
    for name, content in {
        "smoke.yaml": smoke,
        "pair_stability.yaml": model,
        "pair_validation.yaml": validation,
    }.items():
        (v3 / "configs" / name).write_text(content)
        (v4 / "configs" / name).write_text(
            content.replace("pair_stability_v3", "pair_stability_v4").replace("V3", "V4")
        )
    (v4 / "configs" / "pair_stability.yaml").write_text(
        "study:\n  name: pair_stability_v4\nstudy:\n  name: duplicate\n"
    )
    monkeypatch.setattr(routes, "V3_STUDY_DIR", v3)
    monkeypatch.setattr(routes, "STUDY_DIR", v4)
    errors = routes.verify_v4_config_copies(tmp_path)
    assert any("cannot read config pair pair_stability.yaml" in error for error in errors)


def test_comparator_rejects_nonfinite_volatile_value_before_ignoring_it(
    tmp_path: Path,
) -> None:
    """A volatile pointer cannot turn malformed JSON into accepted parity."""

    policy = layout.ArtifactPolicy(
        logical_role="test",
        reference_logical_path="artifact.json",
        candidate_logical_path="artifact.json",
        format="json",
        approved_token_substitutions=(),
        volatile_json_pointers=("/volatile",),
        volatile_csv_columns=(),
        float_tolerant_json_pointers=(),
        float_string_tolerant_json_pointers=(),
        float_tolerant_csv_columns=(),
        json_record_arrays=(),
        presence_only=False,
    )
    contract = layout.LayoutMap(
        schema_version=layout.LAYOUT_SCHEMA_VERSION,
        comparator_schema_version=layout.COMPARATOR_SCHEMA_VERSION,
        rel_tol=1e-9,
        abs_tol=1e-12,
        entries=(),
        source_path=tmp_path / "layout.json",
        sha256="a" * 64,
    )
    with pytest.raises(StrictDataError):
        compare._compare_small_content(
            b'{"volatile": 1}',
            b'{"volatile": NaN}',
            policy=policy,
            layout=contract,
            tokens={},
            sink=compare._DifferenceSink(),
        )


def test_reference_freeze_rejects_ambiguous_task_input_before_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference read-set contract parses JSONL before its audit call."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    tasks = (
        source
        / "01_train"
        / "stage_plans"
        / attempts["train"]
        / "tasks.jsonl"
    )
    tasks.write_text('{"task_id":"a","task_id":"b"}\n')
    with pytest.raises(StrictDataError, match="duplicate JSON object key"):
        reference.freeze_reference(
            source,
            (tmp_path / "reference").absolute(),
            attempts=attempts,
        )
