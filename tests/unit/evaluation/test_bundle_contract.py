"""Regression guards for evaluation bundle ownership."""

from __future__ import annotations

from dataclasses import fields

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.results import ArtifactRecord


def test_evaluation_bundle_fields_are_intentional() -> None:
    assert {field.name for field in fields(EvaluationBundle)} <= {
        "generated",
        "wavefunction",
        "local_energy",
        "derivatives",
        "trace",
        "transform",
        "trace_comparison",
        "feature_trace",
        "readout_trace",
    }


def test_artifact_record_fields_mean_actual_artifact() -> None:
    assert {field.name for field in fields(ArtifactRecord)} == {
        "name",
        "kind",
        "path",
        "metadata",
    }
