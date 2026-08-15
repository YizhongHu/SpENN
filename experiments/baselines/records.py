"""Common results record for the NN-QMC baseline comparison.

One record is one ``(system, code, run)`` row of the scorecard in
``experiments/baselines/README.md`` section 3. Every code in the comparison --
TPEN, FermiNet, LapNet, DeepQMC -- writes this same shape, so the collector
needs no per-code knowledge.

The design rule is that an unknown quantity is ``None``, never a placeholder
number. A record that silently defaults ``wall_clock_seconds`` to ``0.0`` would
turn a missing measurement into a false efficiency claim, so construction
validates instead of coercing.

Notes
-----
Atomic units throughout: energies in hartree, variances in hartree squared.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

# The three efficiency denominators of README section 3 are `steps`,
# `samples`, and `wall_clock_seconds`; all three are kept because reporting only
# one hides either optimizer quality or kernel engineering.
_NON_NEGATIVE_FLOAT_FIELDS = (
    "energy_stderr_hartree",
    "local_energy_variance_hartree2",
    "wall_clock_seconds",
)
_NON_NEGATIVE_INT_FIELDS = (
    "steps",
    "samples",
    "parameter_count",
    "n_gpus",
)
_REQUIRED_TEXT_FIELDS = ("system_id", "code")

#: How the energy estimate was produced. Not free text: a training-tail average
#: and a fixed-parameter inference pass are different quantities, and a table
#: that mixes them silently is wrong in a way no reader can detect.
_ESTIMATORS = ("training_tail", "inference")
_OPTIONAL_TEXT_FIELDS = (
    "code_commit",
    "ansatz",
    "optimizer",
    "dtype",
    "device_type",
    "gpu_model",
    "run_id",
    "run_dir",
    "collected_at",
    "notes",
)


class RecordValidationError(ValueError):
    """A results record violated the schema."""


@dataclasses.dataclass(frozen=True)
class BaselineRecord:
    """One measured ``(system, code, run)`` row of the comparison scorecard.

    Parameters
    ----------
    system_id : str
        Key into ``experiments/baselines/systems.yaml``.
    code : str
        Which implementation produced the row, e.g. ``"tpen"``, ``"ferminet"``.
    code_commit : str or None, optional
        Commit SHA (or release tag) of that implementation. Required for a
        reproducible claim; ``None`` marks the row as not yet reproducible.
    ansatz : str or None, optional
        Ansatz variant within the code, e.g. ``"psiformer"``.
    energy_hartree : float or None, optional
        Variational energy estimate.
    energy_stderr_hartree : float or None, optional
        Monte Carlo standard error on ``energy_hartree``. An energy without an
        error bar cannot be compared against a reference.
    local_energy_variance_hartree2 : float or None, optional
        Local-energy variance, the near-free ansatz-quality signal.
    steps : int or None, optional
        Optimizer steps taken (efficiency denominator 1).
    samples : int or None, optional
        Cumulative local-energy evaluations (efficiency denominator 2,
        hardware-free).
    wall_clock_seconds : float or None, optional
        Measured wall clock for the run (efficiency denominator 3).
    estimator : str
        How ``energy_hartree`` was produced: ``"training_tail"`` for an average
        over the final steps of optimization, or ``"inference"`` for a
        fixed-parameter evaluation pass. Required, and validated against that
        set. These are different quantities -- FermiNet's published table uses
        inference while a training-tail average carries optimization noise and
        any clipping bias -- so a table that mixes them without saying so is
        wrong in a way a reader cannot detect.
    device_type : str or None, optional
        Accelerator family the run executed on, e.g. ``"cpu"``, ``"cuda"``,
        ``"rocm"``, ``"xpu"``. Recorded separately from ``gpu_model`` because a
        CPU row is a legitimate record rather than missing data, and because
        float64 reduction order differs between CPU and GPU: without this field
        a last-digit energy difference cannot be told apart from a device
        artifact.
    gpu_model : str or None, optional
        Accelerator model string; wall clock is meaningless without it. One
        Slurm partition can mix cards (Cannon's ``seas_gpu`` holds both A100-80GB
        and H200 nodes), so this is per-run, not per-partition.
    n_gpus : int or None, optional
        Number of GPUs the run used.
    dtype : str or None, optional
        Numeric dtype, e.g. ``"float64"``. A known confounder (README section 4).
    optimizer : str or None, optional
        Optimizer name, e.g. ``"adam"``, ``"kfac"``. The largest confounder.
    parameter_count : int or None, optional
        Trainable parameter count.
    seed : int or None, optional
        Random seed, for the seed-spread axis.
    run_id : str or None, optional
        Stable identifier for the run.
    run_dir : str or None, optional
        Run directory, recorded relative to the scanned run root so no facility
        absolute path ever enters a committed artifact.
    collected_at : str or None, optional
        ISO 8601 timestamp of collection.
    notes : str or None, optional
        Free-text caveats, e.g. documented deltas from a published protocol.

    Raises
    ------
    RecordValidationError
        If a field has the wrong type, is negative where a count is required,
        or is NaN or infinite.
    """

    system_id: str
    code: str
    code_commit: str | None = None
    ansatz: str | None = None
    energy_hartree: float | None = None
    energy_stderr_hartree: float | None = None
    local_energy_variance_hartree2: float | None = None
    steps: int | None = None
    samples: int | None = None
    wall_clock_seconds: float | None = None
    estimator: str | None = None
    device_type: str | None = None
    gpu_model: str | None = None
    n_gpus: int | None = None
    dtype: str | None = None
    optimizer: str | None = None
    parameter_count: int | None = None
    seed: int | None = None
    run_id: str | None = None
    run_dir: str | None = None
    collected_at: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        for name in _REQUIRED_TEXT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RecordValidationError(f"{name} must be a non-empty string")

        for name in _OPTIONAL_TEXT_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise RecordValidationError(f"{name} must be a string or None")

        # `estimator` is required and closed-vocabulary. It carries a default of
        # None only so the dataclass field order stays stable; omitting it is an
        # error, not a permitted "unknown". A row that cannot say how its energy
        # was produced cannot be compared against one that can.
        if self.estimator not in _ESTIMATORS:
            raise RecordValidationError(
                f"estimator must be one of {_ESTIMATORS}, got {self.estimator!r}"
            )

        # `energy_hartree` is signed; the rest are magnitudes and cannot be
        # negative. All of them must be real numbers when present.
        if self.energy_hartree is not None:
            _check_finite("energy_hartree", self.energy_hartree)
        for name in _NON_NEGATIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            _check_finite(name, value)
            if float(value) < 0.0:
                raise RecordValidationError(f"{name} must be non-negative, got {value!r}")

        for name in _NON_NEGATIVE_INT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise RecordValidationError(f"{name} must be an int or None, got {value!r}")
            if value < 0:
                raise RecordValidationError(f"{name} must be non-negative, got {value!r}")

        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise RecordValidationError(f"seed must be an int or None, got {self.seed!r}")

        # An energy without an error bar is not a comparable measurement.
        if self.energy_hartree is not None and self.energy_stderr_hartree is None:
            raise RecordValidationError("energy_hartree requires energy_stderr_hartree")

    def to_json_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serializable mapping.

        Returns
        -------
        dict
            All fields in declaration order, ``None`` included so that every
            line of ``results.jsonl`` carries the same keys.
        """

        return dataclasses.asdict(self)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the schema's field names in declaration order.

        Returns
        -------
        tuple of str
            Field names.
        """

        return tuple(field.name for field in dataclasses.fields(cls))

    @classmethod
    def from_json_dict(cls, payload: Any) -> "BaselineRecord":
        """Build a record from a parsed JSON object.

        Parameters
        ----------
        payload : Any
            Mapping parsed from a ``baseline_record.json`` file.

        Returns
        -------
        BaselineRecord
            The validated record.

        Raises
        ------
        RecordValidationError
            If ``payload`` is not a mapping, carries unknown keys, or fails
            field validation. Unknown keys are rejected rather than ignored so
            that a typo in an emitter cannot silently drop a measurement.
        """

        if not isinstance(payload, dict):
            raise RecordValidationError("record must be a JSON object")
        known = set(cls.field_names())
        unknown = sorted(set(payload) - known)
        if unknown:
            raise RecordValidationError(f"unknown record fields: {unknown}")
        return cls(**payload)


def _check_finite(name: str, value: Any) -> None:
    """Raise unless ``value`` is a real, finite number.

    Parameters
    ----------
    name : str
        Field name, used in the error message.
    value : Any
        Candidate value.

    Raises
    ------
    RecordValidationError
        If the value is not a real number, or is NaN or infinite.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{name} must be a number or None, got {value!r}")
    if not math.isfinite(float(value)):
        raise RecordValidationError(f"{name} must be finite, got {value!r}")


__all__ = ["BaselineRecord", "RecordValidationError"]
