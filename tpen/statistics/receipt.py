"""Immutable trajectory-statistics receipts and their join identity.

A receipt is the wire record a downstream consumer joins against. It is keyed
by a complete seven-part identity and carries either a full payload or a
reason, never a half-filled row.

The identity is deliberately content-addressed rather than path-addressed.
``checkpoint_sha256`` names what the model *was*, not where its file happened to
sit, so a receipt survives a run tree being moved, re-collected, or archived.
``observable`` is mandatory because autocorrelation is observable-specific --
joining an energy IAT onto a gradient measurement is a silent category error --
and ``evaluator_id`` is versioned so a changed estimator cannot join across
changed semantics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from tpen.statistics.mixing import MixingDiagnostics

__all__ = [
    "IDENTITY_FIELDS",
    "ChainStatistics",
    "PlateauDiagnostics",
    "TrajectoryShape",
    "TrajectoryStatisticsIdentity",
    "TrajectoryStatisticsPayload",
    "TrajectoryStatisticsReceipt",
    "TrajectoryStatisticsStatus",
]

TrajectoryStatisticsStatus: TypeAlias = Literal["available", "absent", "unresolved"]
"""Receipt status.

``available``
    Statistics resolved; the payload is present and every value is finite.
``absent``
    No trajectory was collected for this identity at all.
``unresolved``
    A trajectory exists but the statistics did not resolve -- too few draws, no
    plateau, or chains that disagree. Numerical fields are omitted, never
    zero-filled.
"""

IDENTITY_FIELDS: tuple[str, ...] = (
    "stage",
    "run_id",
    "attempt_id",
    "checkpoint_sha256",
    "config_sha256",
    "observable",
    "evaluator_id",
)
"""The complete join key, in canonical order."""

_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_SHA256_FIELDS = frozenset({"checkpoint_sha256", "config_sha256"})


@dataclass(frozen=True)
class TrajectoryStatisticsIdentity:
    """The seven-part immutable join key for one receipt.

    Parameters
    ----------
    stage : str
        Pipeline stage that produced the trajectory.
    run_id : str
        Run identifier.
    attempt_id : str
        Attempt identifier within the run; a requeued attempt is a different
        measurement and must not silently join onto its predecessor.
    checkpoint_sha256 : str
        Lowercase 64-character sha256 of the checkpoint *contents*.
    config_sha256 : str
        Lowercase 64-character sha256 of the resolved configuration.
    observable : str
        Observable name, for example ``"local_energy"``.
    evaluator_id : str
        Versioned evaluator identity, for example ``"local_energy/v1"``.

    Raises
    ------
    ValueError
        If any field is blank, or a sha256 field is not 64 lowercase hex
        characters. Partial identities are rejected at construction so an
        incomplete key can never reach a sidecar.
    """

    stage: str
    run_id: str
    attempt_id: str
    checkpoint_sha256: str
    config_sha256: str
    observable: str
    evaluator_id: str

    def __post_init__(self) -> None:
        for field_name in IDENTITY_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"{field_name} must be non-empty; the join key admits no blanks")
            if field_name in _SHA256_FIELDS and not _SHA256_PATTERN.match(stripped):
                raise ValueError(
                    f"{field_name} must be 64 lowercase hex characters (sha256), got {stripped!r}"
                )
            object.__setattr__(self, field_name, stripped)

    def as_key(self) -> tuple[str, ...]:
        """Return the identity as a hashable tuple in canonical field order."""
        return tuple(getattr(self, name) for name in IDENTITY_FIELDS)

    def to_dict(self) -> dict[str, str]:
        """Return the identity as a plain JSON-safe mapping."""
        return {name: getattr(self, name) for name in IDENTITY_FIELDS}


@dataclass(frozen=True)
class TrajectoryShape:
    """Layout of the trajectory the statistics were computed from.

    Always present, including on unresolved receipts: knowing a run produced
    three draws is exactly what explains why it did not resolve.

    Parameters
    ----------
    walker_count : int
        Number of independent chains.
    draw_count : int
        Retained draws *per chain*.
    draw_stride : int
        Sampler steps between retained draws.
    burn_in_draws : int
        Draws discarded before the retained window.
    """

    walker_count: int
    draw_count: int
    draw_stride: int
    burn_in_draws: int

    def __post_init__(self) -> None:
        if self.walker_count < 1:
            raise ValueError(f"walker_count must be at least 1, got {self.walker_count}")
        if self.draw_count < 1:
            raise ValueError(f"draw_count must be at least 1, got {self.draw_count}")
        if self.draw_stride < 1:
            raise ValueError(f"draw_stride must be at least 1, got {self.draw_stride}")
        if self.burn_in_draws < 0:
            raise ValueError(f"burn_in_draws must be non-negative, got {self.burn_in_draws}")

    @property
    def total_draws(self) -> int:
        """Return the total sample count across all chains."""
        return self.walker_count * self.draw_count

    def to_dict(self) -> dict[str, int]:
        """Return the shape as a plain JSON-safe mapping."""
        return {
            "walker_count": self.walker_count,
            "draw_count": self.draw_count,
            "total_draws": self.total_draws,
            "draw_stride": self.draw_stride,
            "burn_in_draws": self.burn_in_draws,
        }


@dataclass(frozen=True)
class ChainStatistics:
    """One walker's own estimate, retained whether or not it resolved.

    Per-chain records are the audit trail behind a pooled number. Every chain
    appears, including the ones that failed, because a chain dropped from the
    record is indistinguishable from a chain that was never sampled.

    Parameters
    ----------
    index : int
        Column position of this walker in the trajectory.
    n_draws : int
        Retained draws contributed by this chain.
    status : str
        ``available`` or ``unresolved`` for this chain alone.
    tau_int : float or None
        This chain's integrated autocorrelation time, or ``None`` when it did
        not resolve. Never substituted by a bound or a default.
    plateau_reached : bool
        Whether this chain's own initial positive sequence terminated.
    mean : float or None
        Chain mean, present only when the chain resolved.
    variance : float or None
        Chain sample variance, present only when the chain resolved.
    reason : str or None
        Why this chain did not resolve. Set exactly when ``status`` is
        ``unresolved``.
    """

    index: int
    n_draws: int
    status: str
    tau_int: float | None
    plateau_reached: bool
    mean: float | None
    variance: float | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.status not in ("available", "unresolved"):
            raise ValueError(f"chain status must be available|unresolved, got {self.status!r}")
        resolved = self.status == "available"
        if resolved and self.tau_int is None:
            raise ValueError(f"chain {self.index} is available but carries no tau_int")
        if not resolved and self.tau_int is not None:
            raise ValueError(f"chain {self.index} is unresolved but carries a tau_int")
        if resolved == bool((self.reason or "").strip()):
            raise ValueError(f"chain {self.index} must carry a reason exactly when unresolved")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping for the sidecar."""

        return {
            "index": self.index,
            "n_draws": self.n_draws,
            "status": self.status,
            "tau_int": self.tau_int,
            "plateau_reached": self.plateau_reached,
            "mean": self.mean,
            "variance": self.variance,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> ChainStatistics:
        """Rebuild a chain record from its sidecar mapping."""

        return cls(
            index=int(record["index"]),
            n_draws=int(record["n_draws"]),
            status=str(record["status"]),
            tau_int=record.get("tau_int"),
            plateau_reached=bool(record.get("plateau_reached", False)),
            mean=record.get("mean"),
            variance=record.get("variance"),
            reason=record.get("reason"),
        )


@dataclass(frozen=True)
class PlateauDiagnostics:
    """Whether and where the autocorrelation function turned over.

    Parameters
    ----------
    plateau_reached : bool
        Whether Geyer's initial positive sequence terminated within the data.
    truncation_lag : int or None
        Largest lag included in the ``tau_int`` sum.
    pair_count : int or None
        Number of Geyer pairs summed.
    max_lag : int
        Largest lag the draw count allowed.
    """

    plateau_reached: bool
    truncation_lag: int | None
    pair_count: int | None
    max_lag: int

    def to_dict(self) -> dict[str, Any]:
        """Return the diagnostics as a plain JSON-safe mapping."""
        return {
            "plateau_reached": self.plateau_reached,
            "truncation_lag": self.truncation_lag,
            "pair_count": self.pair_count,
            "max_lag": self.max_lag,
        }


@dataclass(frozen=True)
class TrajectoryStatisticsPayload:
    """The numbers, present only when ``status == "available"``.

    Parameters
    ----------
    tau_int : float
        Integrated autocorrelation time; see
        :data:`tpen.statistics.autocorrelation.TAU_CONVENTION`.
    ess : float
        Effective sample size, ``total_draws / tau_int``.
    mcse : float
        Correlation-aware Monte-Carlo standard error of the mean.
    mean : float
        Sample mean of the observable across all draws and chains.
    variance : float
        Pooled within-chain sample variance.

    Raises
    ------
    ValueError
        If any value is non-finite, or ``tau_int``/``ess``/``variance`` is not
        strictly positive. A payload that exists must be usable.
    """

    tau_int: float
    ess: float
    mcse: float
    mean: float
    variance: float

    def __post_init__(self) -> None:
        for name in ("tau_int", "ess", "mcse", "mean", "variance"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            object.__setattr__(self, name, value)
        for name in ("tau_int", "ess", "variance"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive, got {getattr(self, name)}")
        if self.mcse < 0.0:
            raise ValueError(f"mcse must be non-negative, got {self.mcse}")

    def to_dict(self) -> dict[str, float]:
        """Return the payload as a plain JSON-safe mapping."""
        return {
            "tau_int": self.tau_int,
            "ess": self.ess,
            "mcse": self.mcse,
            "mean": self.mean,
            "variance": self.variance,
        }


@dataclass(frozen=True)
class TrajectoryStatisticsReceipt:
    """One immutable trajectory-statistics record.

    Parameters
    ----------
    identity : TrajectoryStatisticsIdentity
        The seven-part join key.
    status : {'available', 'absent', 'unresolved'}
        Outcome. ``available`` requires a payload and forbids a reason;
        ``absent`` and ``unresolved`` require a reason and forbid a payload.
    recorded_at_utc : str
        UTC ISO-8601 timestamp of when the receipt was produced.
    estimator_id : str
        Identity of the estimator that produced the numbers.
    estimator_version : str
        Version of that estimator.
    tau_convention : str
        Spelled-out IAT convention, so a consumer never has to infer whether
        the half-IAT was used.
    shape : TrajectoryShape
        Trajectory layout. Present for every status except ``absent``.
    plateau : PlateauDiagnostics or None
        Plateau diagnostics, absent only when no trajectory existed.
    mixing : MixingDiagnostics or None
        Split-Rhat diagnostics, absent only when no trajectory existed.
    payload : TrajectoryStatisticsPayload or None
        The numbers; present exactly when ``status == "available"``.
    source_artifact_sha256 : str or None
        Content address of the trajectory the statistics came from.
    reason : str or None
        Why the statistics are not available; present exactly when
        ``status != "available"``.
    warnings : tuple of str
        Non-fatal caveats. A warning never suppresses a payload and never
        substitutes for ``unresolved``.
    chains : tuple of ChainStatistics
        Every walker's own estimate, in column order, including chains that did
        not resolve. This is the audit trail behind any pooled number: the
        pooled value is a function of these, so a reader can recompute it and
        see which chain, if any, forced the whole receipt to ``unresolved``.

    Raises
    ------
    ValueError
        If the status and payload/reason combination is inconsistent.
    """

    identity: TrajectoryStatisticsIdentity
    status: TrajectoryStatisticsStatus
    recorded_at_utc: str
    estimator_id: str
    estimator_version: str
    tau_convention: str
    shape: TrajectoryShape | None = None
    plateau: PlateauDiagnostics | None = None
    mixing: MixingDiagnostics | None = None
    payload: TrajectoryStatisticsPayload | None = None
    source_artifact_sha256: str | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    chains: tuple[ChainStatistics, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("available", "absent", "unresolved"):
            raise ValueError(
                f"status must be one of available|absent|unresolved, got {self.status!r}"
            )
        # Status/payload consistency is the invariant a consumer relies on to
        # avoid reading a zero that was never measured.
        if self.status == "available":
            if self.payload is None:
                raise ValueError("status 'available' requires a payload")
            if self.reason is not None:
                raise ValueError("status 'available' must not carry a reason")
            # A payload alone is not enough. `from_dict` is the consumer-side
            # trust boundary, so a hand-built or hand-edited record could
            # otherwise present numbers as `available` while carrying no
            # plateau, no mixing diagnostics, and no trajectory digest --
            # exactly the shape the no-plateau rule exists to forbid, arriving
            # through the door that skips the producer.
            if self.plateau is None:
                raise ValueError("status 'available' requires plateau diagnostics")
            if not self.plateau.plateau_reached:
                raise ValueError(
                    "status 'available' requires a reached plateau; an unterminated "
                    "initial positive sequence is 'unresolved', never a number"
                )
            if self.mixing is None:
                raise ValueError("status 'available' requires mixing diagnostics")
            if not (self.source_artifact_sha256 or "").strip():
                raise ValueError(
                    "status 'available' requires source_artifact_sha256; statistics "
                    "must name the trajectory content they were computed from"
                )
        else:
            if self.payload is not None:
                raise ValueError(f"status {self.status!r} must not carry a payload")
            if not (self.reason or "").strip():
                raise ValueError(f"status {self.status!r} requires a non-empty reason")
        if self.status != "absent" and self.shape is None:
            raise ValueError(f"status {self.status!r} requires a trajectory shape")
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))
        object.__setattr__(self, "chains", tuple(self.chains))

    def to_dict(self) -> dict[str, Any]:
        """Return a flat JSON-safe mapping suitable for a JSONL sidecar."""

        record: dict[str, Any] = {
            **self.identity.to_dict(),
            "status": self.status,
            "recorded_at_utc": self.recorded_at_utc,
            "estimator_id": self.estimator_id,
            "estimator_version": self.estimator_version,
            "tau_convention": self.tau_convention,
            "source_artifact_sha256": self.source_artifact_sha256,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "chains": [chain.to_dict() for chain in self.chains],
        }
        record["shape"] = None if self.shape is None else self.shape.to_dict()
        record["plateau"] = None if self.plateau is None else self.plateau.to_dict()
        if self.mixing is None:
            record["mixing"] = None
        else:
            record["mixing"] = {
                "r_hat": self.mixing.r_hat,
                "n_split_chains": self.mixing.n_split_chains,
                "draws_per_split_chain": self.mixing.draws_per_split_chain,
                "reason": self.mixing.reason,
            }
        record["statistics"] = None if self.payload is None else self.payload.to_dict()
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> TrajectoryStatisticsReceipt:
        """Rebuild a receipt from the mapping produced by :meth:`to_dict`.

        Parameters
        ----------
        record : Mapping
            A previously serialised receipt.

        Returns
        -------
        TrajectoryStatisticsReceipt
            The reconstructed receipt, revalidated by ``__post_init__``.
        """

        identity = TrajectoryStatisticsIdentity(**{name: record[name] for name in IDENTITY_FIELDS})

        shape_record = record.get("shape")
        shape = (
            None
            if shape_record is None
            else TrajectoryShape(
                walker_count=int(shape_record["walker_count"]),
                draw_count=int(shape_record["draw_count"]),
                draw_stride=int(shape_record["draw_stride"]),
                burn_in_draws=int(shape_record["burn_in_draws"]),
            )
        )
        plateau_record = record.get("plateau")
        plateau = (
            None
            if plateau_record is None
            else PlateauDiagnostics(
                plateau_reached=bool(plateau_record["plateau_reached"]),
                truncation_lag=plateau_record["truncation_lag"],
                pair_count=plateau_record["pair_count"],
                max_lag=int(plateau_record["max_lag"]),
            )
        )
        mixing_record = record.get("mixing")
        mixing = (
            None
            if mixing_record is None
            else MixingDiagnostics(
                r_hat=mixing_record["r_hat"],
                n_split_chains=int(mixing_record["n_split_chains"]),
                draws_per_split_chain=int(mixing_record["draws_per_split_chain"]),
                reason=mixing_record["reason"],
            )
        )
        statistics_record = record.get("statistics")
        payload = (
            None
            if statistics_record is None
            else TrajectoryStatisticsPayload(
                tau_int=float(statistics_record["tau_int"]),
                ess=float(statistics_record["ess"]),
                mcse=float(statistics_record["mcse"]),
                mean=float(statistics_record["mean"]),
                variance=float(statistics_record["variance"]),
            )
        )
        return cls(
            identity=identity,
            status=record["status"],
            recorded_at_utc=record["recorded_at_utc"],
            estimator_id=record["estimator_id"],
            estimator_version=record["estimator_version"],
            tau_convention=record["tau_convention"],
            shape=shape,
            plateau=plateau,
            mixing=mixing,
            payload=payload,
            source_artifact_sha256=record.get("source_artifact_sha256"),
            reason=record.get("reason"),
            warnings=tuple(record.get("warnings") or ()),
            chains=tuple(
                ChainStatistics.from_dict(chain) for chain in (record.get("chains") or ())
            ),
        )
