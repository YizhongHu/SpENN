"""JSON artifact summary for conditioned local-energy diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tpen.evaluation.bundle import EvaluationBundle
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, SummaryResult
from tpen.evaluation.trajectory_records import TrajectoryRecordArtifact
from tpen.statistics.conditioned import (
    CONDITIONED_LOCAL_ENERGY_SCHEMA,
    DEFAULT_MIN_OCCUPIED_DRAWS,
    produce_conditioned_local_energy_statistics,
)

DEFAULT_CONDITIONED_STATISTICS_FILENAME = "conditioned_local_energy.json"
"""Canonical task-local JSON artifact name."""


class ConditionedLocalEnergySummary:
    """Stream the typed retained trajectory into a diagnostic JSON artifact.

    Parameters
    ----------
    range_edges : mapping of str to sequence of float
        Predeclared cut points for minimum electron-nuclear radius, ``r12``,
        maximum radius, hyperradius, ``cos(theta12)``, and ``logabs``.
        Underflow, overflow, and non-finite bins are structural and always
        emitted.
    quantile_seed : int
        Required deterministic seed for bounded descriptive quantile samples.
    deviation_ccdf_thresholds : sequence of float
        Predeclared absolute local-energy deviation thresholds.
    joint_strata : sequence of mappings, optional
        Hard-capped rectangular multi-variable conditions.
    quantiles : sequence of float, optional
        Descriptive flattened quantile probabilities.
    quantile_sample_cap : int, optional
        Per-condition, per-observable bounded sample count.
    top_k : int, optional
        Full records retained for the largest absolute deviations.
    max_event_records : int, optional
        Full records retained for each other rare-event category.
    cancellation_ratio_threshold : float, optional
        Predeclared cancellation-ratio threshold.
    cancellation_term_l1_threshold : float, optional
        Predeclared minimum absolute Hamiltonian-term sum.
    cancellation_energy_floor : float, optional
        Fixed cancellation-ratio denominator floor.
    low_logabs_threshold : float, optional
        Predeclared low-amplitude threshold.
    min_occupied_draws : int, optional
        Minimum occupied retained draws before conditional MCSE is available.
    chunk_size : int, optional
        Maximum trajectory rows held in memory at once.
    filename : str or pathlib.Path, optional
        JSON name relative to the evaluation task output directory.
    """

    name = "conditioned_local_energy"
    required_fields = frozenset()

    def __init__(
        self,
        *,
        range_edges: Mapping[str, Sequence[float]],
        quantile_seed: int,
        deviation_ccdf_thresholds: Sequence[float],
        joint_strata: Sequence[Mapping[str, Any]] = (),
        quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
        quantile_sample_cap: int = 4096,
        top_k: int = 100,
        max_event_records: int = 100,
        cancellation_ratio_threshold: float = 100.0,
        cancellation_term_l1_threshold: float = 10.0,
        cancellation_energy_floor: float = 1.0e-12,
        low_logabs_threshold: float = -20.0,
        min_occupied_draws: int = DEFAULT_MIN_OCCUPIED_DRAWS,
        chunk_size: int = 8192,
        filename: Path | str = DEFAULT_CONDITIONED_STATISTICS_FILENAME,
    ) -> None:
        self.range_edges = dict(range_edges)
        self.quantile_seed = int(quantile_seed)
        self.deviation_ccdf_thresholds = tuple(deviation_ccdf_thresholds)
        self.joint_strata = tuple(joint_strata)
        self.quantiles = tuple(quantiles)
        self.quantile_sample_cap = int(quantile_sample_cap)
        self.top_k = int(top_k)
        self.max_event_records = int(max_event_records)
        self.cancellation_ratio_threshold = float(cancellation_ratio_threshold)
        self.cancellation_term_l1_threshold = float(cancellation_term_l1_threshold)
        self.cancellation_energy_floor = float(cancellation_energy_floor)
        self.low_logabs_threshold = float(low_logabs_threshold)
        self.min_occupied_draws = int(min_occupied_draws)
        self.chunk_size = int(chunk_size)
        self.filename = Path(filename)
        if self.filename.is_absolute() or self.filename.name != str(self.filename):
            raise ValueError("conditioned statistics filename must be one task-local file name")

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Write one deterministic artifact from the typed retained trajectory."""

        del namespace
        artifact = bundle.generated.trajectory_records
        if not isinstance(artifact, TrajectoryRecordArtifact):
            raise ValueError(
                "ConditionedLocalEnergySummary requires generated.trajectory_records; "
                "a terminal snapshot cannot satisfy the retained-trajectory contract"
            )
        report = produce_conditioned_local_energy_statistics(
            artifact,
            range_edges=self.range_edges,
            joint_strata=self.joint_strata,
            quantiles=self.quantiles,
            quantile_sample_cap=self.quantile_sample_cap,
            quantile_seed=self.quantile_seed,
            deviation_ccdf_thresholds=self.deviation_ccdf_thresholds,
            top_k=self.top_k,
            max_event_records=self.max_event_records,
            cancellation_ratio_threshold=self.cancellation_ratio_threshold,
            cancellation_term_l1_threshold=self.cancellation_term_l1_threshold,
            cancellation_energy_floor=self.cancellation_energy_floor,
            low_logabs_threshold=self.low_logabs_threshold,
            min_occupied_draws=self.min_occupied_draws,
            chunk_size=self.chunk_size,
        )
        path = context.task_output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        return SummaryResult(
            metrics={},
            artifacts=(
                ArtifactRecord(
                    name="conditioned_local_energy",
                    kind="json",
                    path=path,
                    metadata={
                        "schema": CONDITIONED_LOCAL_ENERGY_SCHEMA,
                        "source_csv_sha256": artifact.csv_sha256,
                        "source_rows": artifact.row_count,
                        "two_pass_identity_confirmed": True,
                        "headline_estimator": False,
                    },
                ),
            ),
        )


__all__ = [
    "DEFAULT_CONDITIONED_STATISTICS_FILENAME",
    "ConditionedLocalEnergySummary",
]
