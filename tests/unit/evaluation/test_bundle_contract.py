"""Regression guards for evaluation bundle ownership."""

from __future__ import annotations

from dataclasses import MISSING, fields
from pathlib import Path
from typing import get_type_hints

import pytest

from tpen.evaluation.bundle import (
    EvaluationBundle,
    GeneratedConfigurations,
    TransformComparisonValues,
    TransformKind,
    TransformName,
)
from tpen.evaluation.results import ArtifactRecord


def test_evaluation_bundle_fields_are_intentional() -> None:
    assert {field.name for field in fields(EvaluationBundle)} <= {
        "generated",
        "wavefunction",
        "local_energy",
        "derivatives",
        "electron_nucleus_radial",
        "helium_atlas",
        "trace",
        "transform",
        "trace_comparison",
        "feature_trace",
        "readout_trace",
        "factor_response",
    }


def test_records_and_atlas_carriers_coexist_as_typed_bundle_fields() -> None:
    assert [field.name for field in fields(GeneratedConfigurations)] == [
        "batch",
        "metadata",
        "trajectory_records",
    ]
    assert "helium_atlas" in {
        field.name for field in fields(EvaluationBundle)
    }


def test_transform_comparison_identity_fields_are_required_and_typed() -> None:
    transform_fields = {field.name: field for field in fields(TransformComparisonValues)}
    identity_fields = {
        "sample_index",
        "original_positions",
        "transformed_positions",
        "original_logabs",
        "transformed_logabs",
        "transform_name",
        "transform_kind",
        "finite",
    }
    assert identity_fields <= set(transform_fields)
    assert all(transform_fields[name].default is MISSING for name in identity_fields)
    annotations = get_type_hints(TransformComparisonValues)
    assert annotations["transform_name"] is TransformName
    assert annotations["transform_kind"] is TransformKind


def test_artifact_record_fields_mean_actual_artifact() -> None:
    assert {field.name for field in fields(ArtifactRecord)} == {
        "name",
        "kind",
        "path",
        "metadata",
    }


def test_artifact_record_metadata_is_json_scalar_only() -> None:
    ArtifactRecord(name="records", kind="csv", path=Path("records.csv"), metadata={"rows": 2})

    with pytest.raises(TypeError, match="JSON scalar"):
        ArtifactRecord(
            name="records",
            kind="csv",
            path=Path("records.csv"),
            metadata={"bad": Path("nested")},
        )
