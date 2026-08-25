"""Contracts for range-conditioned and rare-event trajectory statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import (
    ElectronBatch,
    two_electron_atomic_geometry,
    two_electron_atomic_geometry_reference,
)
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries import ConditionedLocalEnergySummary
from tpen.evaluation.trajectory_records import (
    TrajectoryRecordArtifact,
    TrajectoryRecordBatch,
    TrajectoryRecordStreamWriter,
)
from tpen.statistics.conditioned import (
    MAX_CCDF_THRESHOLD_COUNT,
    MAX_EVENT_RECORD_CAP,
    MAX_JOINT_STRATA,
    ConditionedStatisticsReport,
    produce_conditioned_local_energy_statistics,
)
from tpen.statistics.trajectory import ObservableTrajectory


def _positions(n_draws: int, n_walkers: int) -> torch.Tensor:
    positions = torch.zeros(n_draws, n_walkers, 2, 3, dtype=torch.float64)
    for draw in range(n_draws):
        for walker in range(n_walkers):
            positions[draw, walker, 0, 0] = 0.8 + 0.02 * (walker % 3)
            positions[draw, walker, 1, 1] = 1.0 + 0.01 * (draw % 5)
    return positions


def _artifact(
    tmp_path: Path,
    values: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    logabs: torch.Tensor | None = None,
    term_energies: dict[str, torch.Tensor] | None = None,
    name: str = "trajectory.csv",
) -> TrajectoryRecordArtifact:
    values = values.to(torch.float64)
    n_draws, n_walkers = values.shape
    positions = _positions(n_draws, n_walkers) if positions is None else positions.to(torch.float64)
    logabs = torch.full_like(values, -2.0) if logabs is None else logabs.to(torch.float64)
    if term_energies is None:
        term_energies = {
            "kinetic": 2.0 * values,
            "potential": -values,
        }
    atoms = AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )

    def record(draw: int) -> TrajectoryRecordBatch:
        return TrajectoryRecordBatch(
            draw_index=torch.full((n_walkers,), draw, dtype=torch.int64),
            walker_index=torch.arange(n_walkers, dtype=torch.int64),
            positions=positions[draw],
            local_energy=values[draw],
            term_energies={term: rows[draw] for term, rows in term_energies.items()},
            logabs=logabs[draw],
            sign=torch.ones(n_walkers, dtype=torch.float64),
            finite_mask=torch.isfinite(values[draw]),
        )

    writer = TrajectoryRecordStreamWriter(
        tmp_path / name,
        observable="local_energy",
        n_draws=n_draws,
        n_walkers=n_walkers,
        term_names=tuple(term_energies),
        atomic_configuration=atoms,
        first_draw=record(0),
    )
    for draw in range(1, n_draws):
        writer.append(record(draw))
    return writer.finalize(
        ObservableTrajectory(
            observable="local_energy",
            values=values,
            draw_stride=1,
            burn_in_draws=0,
        )
    )


def _range_edges() -> dict[str, list[float]]:
    return {
        "minimum_electron_nuclear_radius": [0.5, 1.5],
        "electron_electron_distance": [0.5, 2.0],
        "maximum_electron_nuclear_radius": [0.5, 1.5],
        "hyperradius": [1.0, 2.0],
        "cos_theta12": [-0.5, 0.5],
        "logabs": [-10.0, -1.0],
    }


def _produce(artifact: TrajectoryRecordArtifact, **overrides) -> ConditionedStatisticsReport:
    options = {
        "range_edges": _range_edges(),
        "joint_strata": (
            {
                "name": "compact_orthogonal",
                "bounds": {
                    "maximum_electron_nuclear_radius": [0.0, 2.0],
                    "cos_theta12": [-0.25, 0.25],
                },
            },
        ),
        "quantiles": (0.1, 0.5, 0.9),
        "quantile_sample_cap": 7,
        "quantile_seed": 314159,
        "deviation_ccdf_thresholds": (0.5, 2.0),
        "top_k": 2,
        "max_event_records": 2,
        "cancellation_ratio_threshold": 2.5,
        "cancellation_term_l1_threshold": 0.0,
        "low_logabs_threshold": -3.0,
        "chunk_size": 11,
    }
    options.update(overrides)
    return produce_conditioned_local_energy_statistics(artifact, **options)


def _bin(record: dict, quantity: str, bin_id: str) -> dict:
    return next(
        item
        for item in record["range_conditioned"][quantity]["bins"]
        if item["id"] == bin_id
    )


def test_two_electron_geometry_matches_slow_reference_and_marks_undefined_angle() -> None:
    positions = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
            [[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    atoms = AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
        atomic_configuration=atoms,
    )

    actual = two_electron_atomic_geometry(batch)
    reference = two_electron_atomic_geometry_reference(batch)

    for name in (
        "minimum_electron_nuclear_radius",
        "electron_electron_distance",
        "maximum_electron_nuclear_radius",
        "hyperradius",
        "cos_theta12",
    ):
        assert torch.allclose(
            getattr(actual, name),
            getattr(reference, name),
            rtol=1.0e-14,
            atol=1.0e-14,
            equal_nan=True,
        )
    assert actual.angle_defined.tolist() == [True, False, True]
    assert actual.angle_undefined_at_coalescence.tolist() == [False, True, False]
    assert math.isnan(float(actual.cos_theta12[1]))
    assert actual.cos_theta12[0].item() == pytest.approx(0.0)


def test_structural_overflow_preserves_probability_and_variance_reconciliation(
    tmp_path: Path,
) -> None:
    values = torch.arange(30, dtype=torch.float64).reshape(15, 2)
    positions = _positions(15, 2)
    positions[4, 1, 1] = torch.tensor([0.0, 5.0, 0.0])
    artifact = _artifact(tmp_path, values, positions=positions)

    record = _produce(artifact).to_dict()
    partition = record["range_conditioned"]["maximum_electron_nuclear_radius"]
    overflow = _bin(record, "maximum_electron_nuclear_radius", "overflow")

    assert overflow["support"] == 1
    assert partition["reconciliation"]["probability_sum"] == 1.0
    assert partition["reconciliation"]["finite_count_sum"] == 30
    assert partition["reconciliation"]["second_moment_contribution_sum"] == pytest.approx(
        record["global"]["second_moment_about_mean"]
    )
    assert partition["structural_bins"] == ["underflow", "overflow", "nonfinite"]


def test_seeded_bounded_quantiles_are_byte_identical(tmp_path: Path) -> None:
    values = torch.arange(80, dtype=torch.float64).reshape(40, 2)
    artifact = _artifact(tmp_path, values)

    first = json.dumps(_produce(artifact).to_dict(), sort_keys=True, allow_nan=False)
    second = json.dumps(_produce(artifact).to_dict(), sort_keys=True, allow_nan=False)

    assert first == second
    record = json.loads(first)
    quantile = _bin(record, "logabs", "range_000")["observables"]["local_energy"][
        "flattened_quantiles"
    ]
    assert quantile["configured_seed"] == 314159
    assert quantile["sample_count"] == 7
    assert quantile["population_count"] == 80
    assert quantile["exact"] is False


def test_draw_level_ratio_mcse_detects_autocorrelation_without_second_pooling(
    tmp_path: Path,
) -> None:
    n_draws = 256
    n_walkers = 4
    generator = torch.Generator().manual_seed(1234)
    noise = torch.randn(n_draws, n_walkers, generator=generator, dtype=torch.float64)
    values = torch.zeros_like(noise)
    for draw in range(1, n_draws):
        values[draw] = 0.9 * values[draw - 1] + noise[draw]
    artifact = _artifact(tmp_path, values)

    record = _produce(artifact, quantile_sample_cap=32).to_dict()
    statistics = _bin(record, "logabs", "range_000")["observables"]["local_energy"]
    mcse = statistics["conditional_mean_mcse"]
    iid = statistics["flattened_iid_stderr"]

    assert mcse["status"] == "available"
    assert mcse["correlated_axis"] == "retained_draw"
    assert mcse["walker_reduction"] == "within_draw_sum_before_autocorrelation"
    assert mcse["tau_int"] > 2.0
    assert mcse["value"] > 2.0 * iid
    assert mcse["value"] / iid == pytest.approx(math.sqrt(mcse["tau_int"]), rel=0.35)


def test_sparse_empty_and_nonfinite_bins_have_explicit_states(tmp_path: Path) -> None:
    values = torch.arange(32, dtype=torch.float64).reshape(16, 2)
    values[0, 0] = float("nan")
    positions = _positions(16, 2)
    positions[:3, 0, 0] = torch.zeros(3, dtype=torch.float64)
    artifact = _artifact(tmp_path, values, positions=positions)

    record = _produce(artifact).to_dict()
    underflow = _bin(record, "minimum_electron_nuclear_radius", "underflow")
    undefined_angle = _bin(record, "cos_theta12", "nonfinite")

    assert underflow["support"] == 3
    assert underflow["finite_status"] == "partially_nonfinite"
    assert underflow["observables"]["local_energy"]["conditional_mean_mcse"][
        "status"
    ] == "unresolved"
    assert "minimum 8" in underflow["observables"]["local_energy"][
        "conditional_mean_mcse"
    ]["reason"]
    assert undefined_angle["support"] == 3
    empty = _bin(record, "maximum_electron_nuclear_radius", "underflow")
    assert empty["finite_status"] == "empty"
    assert empty["observables"]["local_energy"]["conditional_mean"] is None
    assert empty["observables"]["local_energy"]["conditional_mean_mcse"]["status"] == "empty"
    assert record["rare_events"]["nonfinite"]["total_count"] >= 1


def test_two_pass_identity_and_attribution_tampering_fail_loudly(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, torch.arange(32, dtype=torch.float64).reshape(16, 2))
    report = _produce(artifact)
    record = report.to_dict()

    assert record["source"]["statistics_pass"]["csv_sha256"] == artifact.csv_sha256
    assert record["source"]["rare_events_pass"]["csv_sha256"] == artifact.csv_sha256
    assert record["source"]["statistics_pass"]["byte_count"] == artifact.byte_count
    assert record["source"]["rare_events_pass"]["byte_count"] == artifact.byte_count

    identity_tampered = report.to_dict()
    identity_tampered["source"]["rare_events_pass"]["csv_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rare_events_pass.csv_sha256"):
        ConditionedStatisticsReport(identity_tampered).validate()

    first_bin = record["range_conditioned"]["logabs"]["bins"][0]
    first_bin["variance_attribution"]["second_moment_contribution"] += 1.0
    with pytest.raises(ValueError, match="second_moment_contribution_sum"):
        ConditionedStatisticsReport(record).validate()


def test_fixed_bin_and_cap_contracts_reject_unsafe_configuration(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, torch.arange(32, dtype=torch.float64).reshape(16, 2))
    missing = _range_edges()
    del missing["logabs"]

    with pytest.raises(ValueError, match="define exactly"):
        _produce(artifact, range_edges=missing)
    with pytest.raises(TypeError, match="terminal snapshots are not accepted"):
        _produce(artifact.final_draw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hard cap"):
        _produce(
            artifact,
            joint_strata=tuple(
                {"name": f"stratum-{index}", "bounds": {"logabs": [-3.0, -1.0]}}
                for index in range(MAX_JOINT_STRATA + 1)
            ),
        )
    with pytest.raises(ValueError, match="top_k"):
        _produce(artifact, top_k=MAX_EVENT_RECORD_CAP + 1)
    with pytest.raises(ValueError, match="deviation_ccdf_thresholds.*hard cap"):
        _produce(
            artifact,
            deviation_ccdf_thresholds=tuple(
                float(index) for index in range(MAX_CCDF_THRESHOLD_COUNT + 1)
            ),
        )


def test_rare_event_records_are_full_and_bounded(tmp_path: Path) -> None:
    values = torch.tensor(
        [[0.1, 10.0], [0.2, -8.0], [0.3, 7.0], [0.4, float("inf")]] * 4,
        dtype=torch.float64,
    )
    logabs = torch.full_like(values, -5.0)
    artifact = _artifact(tmp_path, values, logabs=logabs)

    rare = _produce(artifact, top_k=1, max_event_records=1).to_dict()["rare_events"]

    assert rare["top_k_absolute_deviations"]["record_count"] == 1
    assert rare["top_k_absolute_deviations"]["truncated"] is True
    assert rare["nonfinite"]["record_count"] == 1
    assert rare["nonfinite"]["truncated"] is True
    assert rare["cancellation"]["record_count"] == 1
    assert rare["low_amplitude"]["record_count"] == 1
    selected = rare["top_k_absolute_deviations"]["records"][0]
    assert "row" in selected and "derived_geometry" in selected
    assert "position/electron_1/axis_2" in selected["row"]


def test_json_summary_requires_typed_trajectory_and_emits_no_headline_metric(
    tmp_path: Path,
) -> None:
    values = torch.arange(32, dtype=torch.float64).reshape(16, 2)
    artifact = _artifact(tmp_path, values)
    atoms = artifact.atomic_configuration
    batch = ElectronBatch(
        positions=artifact.final_draw.positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
        atomic_configuration=atoms,
    )
    context = EvaluationContext(
        namespace="eval/mcmc_energy",
        artifact_level="records",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=1,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )
    summary = ConditionedLocalEnergySummary(
        range_edges=_range_edges(),
        quantile_seed=17,
        deviation_ccdf_thresholds=(1.0,),
        quantile_sample_cap=5,
        top_k=1,
        max_event_records=1,
    )
    bundle = EvaluationBundle(
        generated=GeneratedConfigurations(
            batch=batch,
            metadata={},
            trajectory_records=artifact,
        )
    )

    result = summary.summarize(bundle=bundle, context=context, namespace=context.namespace)

    assert result.metrics == {}
    (output,) = result.artifacts
    assert output.metadata["headline_estimator"] is False
    payload = json.loads(output.path.read_text(encoding="utf-8"))
    assert payload["schema"] == "conditioned_local_energy/v1"
    assert payload["source"]["two_pass_identity_confirmed"] is True

    missing = EvaluationBundle(
        generated=GeneratedConfigurations(batch=batch, metadata={}, trajectory_records=None)
    )
    with pytest.raises(ValueError, match="terminal snapshot"):
        summary.summarize(bundle=missing, context=context, namespace=context.namespace)
