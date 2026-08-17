"""Consume the trajectory-statistics producer and emit its sidecar receipt.

This summary is a *consumer*. It hands the collected ``[draw, walker]``
trajectory to :func:`tpen.statistics.produce_trajectory_statistics`, writes the
returned receipt to an append-only JSONL sidecar, and projects it into metrics.
It never estimates ``tau_int``, ``ess`` or ``mcse`` itself, and it never
re-derives or second-guesses the producer's statistical decisions: the producer
owns those, and a consumer that recomputed them could disagree with the record
it just wrote.

Two reporting rules are load-bearing.

**Both estimators are published, each explicitly labelled.** ``{prefix}_mcse``
is the correlation-aware bar; ``{prefix}_stderr_iid`` is the naive
``sigma / sqrt(N)`` over the same trajectory. The IID value is never deleted,
never relabelled as an MCSE, and never silently substituted for one. Publishing
both over the *same* samples is what makes ``{prefix}_mcse_inflation`` --- their
ratio ``mcse / stderr_iid`` --- a meaningful number a reader can check. The
separate snapshot metric ``local_energy_stderr`` from
:func:`tpen.evaluation.summaries.local_energy.summarize_values` is a different
sample and is left untouched.

``mcse_inflation`` is **not** ``sqrt(tau_int)``, and the difference is
structural rather than numerical. ``mcse = stderr_iid * sqrt(tau_int)`` is the
shortcut :func:`tpen.statistics.produce_trajectory_statistics` rejects by name,
because it assumes every chain shares one variance and one tau. What is
actually computed is

- ``stderr_iid = sqrt(Var_pooled / N)``, from the *pooled* variance;
- ``mcse^2 = sum_i (N_i/N)^2 * s_i^2 * tau_i / N_i``, from *per-chain* variances;
- ``ess = sum_i (N_i / tau_i)`` and ``tau_int = N / ess``.

``tau_int`` is therefore an N-weighted **harmonic** mean of the per-chain tau,
dominated by the best-mixed chains, while ``mcse`` is dominated by the chains
with the largest ``s_i^2 * tau_i``. On homogeneous chains the two coincide and
the ratio *does* equal ``sqrt(tau_int)``, which is exactly when the shortcut's
assumptions hold. On heterogeneous chains they diverge, and ``tau_int < 1``
alongside ``mcse_inflation > 1`` is a consistent observation rather than a
contradiction. ``tests/unit/evaluation/test_trajectory_statistics_consumer.py``
pins both halves of that statement; do not "fix" the estimator to make the two
numbers agree.

**Missing is never zero.** ``absent`` and ``unresolved`` receipts carry no
payload, so the numeric metrics are *omitted* rather than filled with zeros, and
the status and its reason travel on the emitted artifact record. A zero MCSE
would read as "there is no correlation"; the whole point of the status field is
that "we do not know" must stay distinguishable from "we measured nothing".
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tpen.checkpoint.hashing import file_sha256, resolved_config_hash
from tpen.evaluation.bundle import EvaluationBundle
from tpen.evaluation.generators.trajectory import TRAJECTORY_METADATA_KEY
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, JsonScalar, MetricScalar, SummaryResult
from tpen.statistics import (
    ObservableTrajectory,
    TrajectoryStatisticsIdentity,
    TrajectoryStatisticsReceipt,
    TrajectoryStatisticsSidecar,
    produce_trajectory_statistics,
)
from tpen.statistics.producer import absent_receipt

DEFAULT_SIDECAR_NAME = "trajectory_statistics.jsonl"
"""Sidecar filename written into the task output directory by default."""


class TrajectoryStatisticsSummary:
    """Produce, persist, and project one trajectory-statistics receipt.

    Parameters
    ----------
    stage : str
        Pipeline stage recorded in the join identity.
    run_id : str
        Run identifier recorded in the join identity.
    attempt_id : str
        Attempt identifier. A requeued attempt is a different measurement and
        must not join onto its predecessor.
    checkpoint_path : pathlib.Path or str
        Checkpoint file whose *contents* are hashed to produce
        ``checkpoint_sha256``. The identity is content-addressed, never
        path-derived, so a receipt still joins after a run tree is moved.
    evaluator_id : str
        Versioned consumer identity, for example ``"he_mcse/v1"``.
    config_sha256 : str, optional
        Resolved-configuration hash. Supply this or `config`.
    config : object, optional
        Resolved configuration to hash with
        :func:`tpen.checkpoint.hashing.resolved_config_hash`. Ignored when
        `config_sha256` is given.
    observable : str, optional
        Observable name; must match the collected trajectory.
    sidecar_path : pathlib.Path or str, optional
        Sidecar location. Defaults to `DEFAULT_SIDECAR_NAME` inside the task
        output directory. A relative path is resolved against that directory.
    prefix : str, optional
        Metric name prefix.
    verify_checkpoint_contents : bool, optional
        Pass `checkpoint_path` through to the producer so it re-binds the
        claimed hash to the file's contents. Defaults to ``True``.
    producer_options : mapping, optional
        Extra keyword arguments forwarded verbatim to
        :func:`~tpen.statistics.produce_trajectory_statistics` (for example
        ``min_draws_per_chain``). Estimator policy stays with the producer.

    Raises
    ------
    ValueError
        If neither `config_sha256` nor `config` is supplied.
    """

    name = "trajectory_statistics"
    required_fields = frozenset()

    def __init__(
        self,
        *,
        stage: str,
        run_id: str,
        attempt_id: str,
        checkpoint_path: Path | str,
        evaluator_id: str,
        config_sha256: str | None = None,
        config: Any = None,
        observable: str = "local_energy",
        sidecar_path: Path | str | None = None,
        prefix: str = "local_energy",
        verify_checkpoint_contents: bool = True,
        producer_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.stage = str(stage)
        self.run_id = str(run_id)
        self.attempt_id = str(attempt_id)
        self.checkpoint_path = Path(checkpoint_path)
        self.evaluator_id = str(evaluator_id)
        if config_sha256 is None and config is None:
            raise ValueError(
                "TrajectoryStatisticsSummary requires config_sha256 or config; the "
                "join identity admits no blanks"
            )
        self._config_sha256 = None if config_sha256 is None else str(config_sha256)
        self._config = config
        self.observable = str(observable)
        self.sidecar_path = None if sidecar_path is None else Path(sidecar_path)
        self.prefix = str(prefix)
        self.verify_checkpoint_contents = bool(verify_checkpoint_contents)
        self.producer_options = dict(producer_options or {})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Produce the receipt, append it to the sidecar, and project metrics."""

        identity = self._identity()
        trajectory = _trajectory_from(bundle)

        if trajectory is None:
            # An explicit `absent` row, not a missing row: it records that the
            # question was asked and nothing was collected, which is what stops
            # a downstream reader treating the gap as a zero.
            receipt = absent_receipt(
                identity,
                reason=(
                    "no observable trajectory was published by the generator under "
                    f"metadata key {TRAJECTORY_METADATA_KEY!r}; a snapshot generator "
                    "retains no draw axis, so no autocorrelation can be estimated"
                ),
            )
        else:
            receipt = produce_trajectory_statistics(
                trajectory,
                identity,
                checkpoint_path=self.checkpoint_path if self.verify_checkpoint_contents else None,
                **self.producer_options,
            )

        sidecar_path = self._sidecar_path(context)
        sidecar = TrajectoryStatisticsSidecar(sidecar_path)
        sidecar.append(receipt)

        metrics = _receipt_metrics(receipt, prefix=self.prefix)
        artifact = ArtifactRecord(
            name=f"{self.prefix}_trajectory_statistics",
            kind="trajectory_statistics_sidecar",
            path=sidecar_path,
            metadata=_receipt_artifact_metadata(receipt),
        )
        return SummaryResult(metrics=metrics, artifacts=(artifact,))

    def _identity(self) -> TrajectoryStatisticsIdentity:
        """Build the seven-part join key, hashing the checkpoint's contents."""

        return TrajectoryStatisticsIdentity(
            stage=self.stage,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            # Content-addressed on purpose. A path-derived key would break
            # silently the moment a run tree is moved or re-collected, and a
            # silent join failure is worse than a loud one because it yields a
            # plausible, wrong pairing.
            checkpoint_sha256=file_sha256(self.checkpoint_path),
            config_sha256=self._resolved_config_sha256(),
            observable=self.observable,
            evaluator_id=self.evaluator_id,
        )

    def _resolved_config_sha256(self) -> str:
        if self._config_sha256 is not None:
            return self._config_sha256
        return resolved_config_hash(self._config)

    def _sidecar_path(self, context: EvaluationContext) -> Path:
        if self.sidecar_path is None:
            return Path(context.task_output_dir) / DEFAULT_SIDECAR_NAME
        if self.sidecar_path.is_absolute():
            return self.sidecar_path
        return Path(context.task_output_dir) / self.sidecar_path


def _trajectory_from(bundle: EvaluationBundle) -> ObservableTrajectory | None:
    """Return the generator's published trajectory, or ``None`` when absent."""

    value = bundle.generated.metadata.get(TRAJECTORY_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, ObservableTrajectory):
        raise TypeError(
            f"metadata[{TRAJECTORY_METADATA_KEY!r}] must be an ObservableTrajectory, "
            f"got {type(value).__name__}"
        )
    return value


def _receipt_metrics(
    receipt: TrajectoryStatisticsReceipt,
    *,
    prefix: str,
) -> dict[str, MetricScalar]:
    """Project a receipt into metrics, omitting rather than zero-filling.

    Metrics are numeric-only (`MetricScalar` admits no strings), so the status
    string and its reason travel on the artifact record instead. The boolean
    ``{prefix}_trajectory_statistics_available`` is always present so a consumer
    can tell "not measured" from "not reported", and the numeric statistics
    appear only when the producer actually resolved them.
    """

    available = receipt.status == "available"
    metrics: dict[str, MetricScalar] = {
        f"{prefix}_trajectory_statistics_available": available,
    }
    if receipt.shape is not None:
        metrics[f"{prefix}_trajectory_walkers"] = receipt.shape.walker_count
        metrics[f"{prefix}_trajectory_draws_per_walker"] = receipt.shape.draw_count
        metrics[f"{prefix}_trajectory_total_draws"] = receipt.shape.total_draws
        metrics[f"{prefix}_trajectory_draw_stride"] = receipt.shape.draw_stride
        metrics[f"{prefix}_trajectory_burn_in_draws"] = receipt.shape.burn_in_draws
    if receipt.mixing is not None and receipt.mixing.r_hat is not None:
        metrics[f"{prefix}_trajectory_split_r_hat"] = float(receipt.mixing.r_hat)

    # Truncation diagnostics, ABOVE the payload guard on purpose. `plateau_reached`
    # is the quality indicator for the study's PRIMARY uncertainty claim: if
    # Geyer's initial positive sequence did not terminate inside the data, the
    # tail was cut at the window edge and the MCSE is UNDERSTATED. Without it on
    # the row a reader cannot tell a well-estimated bar from a truncated one.
    # `producer.py` builds `PlateauDiagnostics` unconditionally and passes it into
    # every `unresolved(...)` return, so an unresolved receipt carries these
    # fields too -- and unresolved is exactly when the reader most needs them,
    # because it is the case where tau and ESS were withheld. Projecting them
    # after the guard below would drop the diagnostic precisely where it explains
    # the outcome. Pure projection of what the receipt already carries; no new
    # statistic and no new computation.
    #
    # `geyer_pair_count` deliberately diverges from the receipt's `pair_count`:
    # this is a new metric, never emitted or cited in a merged receipt, so
    # ADR-E003's ban on second spellings is not engaged, and a bare `pair_count`
    # beside a dozen other counts on an eval row does not say what it counts.
    if receipt.plateau is not None:
        metrics[f"{prefix}_plateau_reached"] = receipt.plateau.plateau_reached
        # `max_lag` travels with `truncation_lag` because the latter is not
        # interpretable alone: truncating at lag 7 is healthy in a window of 15
        # and pathological in a window of 400, and a reader should not have to
        # cross-reference a config that may since have changed.
        metrics[f"{prefix}_max_lag"] = receipt.plateau.max_lag
        # Absent rather than zero-filled: no pair was ever summed, and a 0 would
        # read as "truncated at lag zero" instead of "never got that far".
        if receipt.plateau.truncation_lag is not None:
            metrics[f"{prefix}_truncation_lag"] = receipt.plateau.truncation_lag
        if receipt.plateau.pair_count is not None:
            metrics[f"{prefix}_geyer_pair_count"] = receipt.plateau.pair_count

    payload = receipt.payload
    if payload is None:
        # No payload, no numbers. Filling zeros here would turn "we do not know"
        # into "there is no correlation".
        return metrics

    total_draws = receipt.shape.total_draws if receipt.shape is not None else None
    metrics[f"{prefix}_mcse"] = payload.mcse
    metrics[f"{prefix}_ess"] = payload.ess
    metrics[f"{prefix}_tau_int"] = payload.tau_int
    metrics[f"{prefix}_trajectory_mean"] = payload.mean
    metrics[f"{prefix}_trajectory_variance"] = payload.variance
    if total_draws:
        # Arithmetic on published values, not a re-estimation: sigma/sqrt(N)
        # over the same trajectory the MCSE was computed from. Reported
        # alongside the MCSE and explicitly labelled IID so a reader can see the
        # inflation factor instead of having to infer which bar is which.
        stderr_iid = (payload.variance / total_draws) ** 0.5
        metrics[f"{prefix}_stderr_iid"] = stderr_iid
        if stderr_iid > 0.0:
            # NOT sqrt(tau_int), and deliberately so. `mcse` sums PER-CHAIN
            # variance*tau terms while `stderr_iid` uses the POOLED variance and
            # `tau_int` is an N-weighted HARMONIC mean of the per-chain tau. The
            # producer rejects the shortcut that would make them agree because
            # it "assumes every chain shares one variance and one tau, so it is
            # wrong exactly when the walkers differ". Heterogeneous chains can
            # therefore show tau_int < 1 alongside inflation > 1: the harmonic
            # mean is pulled down by the best-mixed chains while the MCSE is
            # dominated by the highest-variance ones. Do not "fix" this into
            # agreement -- that would substitute the wrong estimator.
            metrics[f"{prefix}_mcse_inflation"] = payload.mcse / stderr_iid
    return metrics


def _receipt_artifact_metadata(receipt: TrajectoryStatisticsReceipt) -> dict[str, JsonScalar]:
    """Carry status, reason, and estimator identity onto the artifact record.

    `MetricScalar` is numeric, so this is the channel by which a non-numeric
    ``absent``/``unresolved`` outcome and its reason survive projection into a
    report rather than being rendered as a blank.
    """

    metadata: dict[str, JsonScalar] = {
        "status": receipt.status,
        "reason": receipt.reason,
        "observable": receipt.identity.observable,
        "stage": receipt.identity.stage,
        "run_id": receipt.identity.run_id,
        "attempt_id": receipt.identity.attempt_id,
        "checkpoint_sha256": receipt.identity.checkpoint_sha256,
        "config_sha256": receipt.identity.config_sha256,
        "evaluator_id": receipt.identity.evaluator_id,
        "estimator_id": receipt.estimator_id,
        "estimator_version": receipt.estimator_version,
        "tau_convention": receipt.tau_convention,
        "recorded_at_utc": receipt.recorded_at_utc,
        "warning_count": len(receipt.warnings),
    }
    if receipt.warnings:
        metadata["warnings"] = " | ".join(receipt.warnings)
    return metadata


__all__ = ["DEFAULT_SIDECAR_NAME", "TrajectoryStatisticsSummary"]
