"""GPU hardware strata, partition wall limits, and the delivered-device check.

Cannon's production GPU targets are NOT one hardware stratum. ``seas_gpu``
carries both ``nvidia_h200`` and ``nvidia_a100-sxm4-80gb`` nodes, and a requeue
can change the card underneath a campaign. That already destroyed timing
comparability between two of this program's own baseline runs, so this module
makes the pin mandatory rather than advisory:

1. every GPU row is submitted with an explicit ``--constraint`` feature;
2. the requested constraint is recorded in the row's receipt; and
3. the DELIVERED device is asserted from inside the allocation.

The constraint is what was asked for. The in-job banner is what was got. A
mismatch fails the row loudly -- it is never downgraded to a note, because a
silently reassigned card produces numbers that look fine and cannot be pooled.

The numbers here were measured read-only on 2026-08-15 and are recorded in the
Cannon note ``gpu-hardware-strata-and-constraint-rule-2026-08-15``. Re-measure
before changing them; do not infer a device from hostname, partition, or
advertised GRES.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: Minutes in one day, spelled once so the wall limits below read as measured.
_DAY_MIN = 24 * 60


class StratumError(ValueError):
    """A stratum, partition, or wall-time request violates the pinning rule."""


class DeliveredDeviceMismatch(RuntimeError):
    """The allocation delivered a different card than the one constrained for.

    This is a row failure, not a warning. Timings, throughput, and GPU-seconds
    from a mismatched row are not comparable with the stratum they were planned
    under.
    """


@dataclass(frozen=True)
class Stratum:
    """One pinnable GPU hardware stratum.

    Parameters
    ----------
    name : str
        Study-facing stratum name, used in row ids and receipts.
    constraint : str
        Slurm node feature passed as ``--constraint``.
    device_name_required : tuple of str
        Lowercased substrings that must ALL appear in the delivered device
        name.
    device_name_forbidden : tuple of str
        Lowercased substrings that must appear in NONE of it. This is what
        keeps a MIG slice from satisfying a full-card stratum.
    partitions : tuple of str
        Partitions on which this stratum actually exists.
    """

    name: str
    constraint: str
    device_name_required: tuple[str, ...]
    device_name_forbidden: tuple[str, ...]
    partitions: tuple[str, ...]


#: Declared GPU strata. ``a100_mig`` is deliberately canary-only: Cannon's
#: ``gpu_test`` partition serves ``nvidia_a100_3g.20gb`` slices with no
#: distinguishing node feature, so its empty constraint must never leak into
#: the production placement validator.
STRATA: Mapping[str, Stratum] = {
    "h200": Stratum(
        name="h200",
        constraint="h200",
        device_name_required=("h200",),
        device_name_forbidden=("mig",),
        partitions=("seas_gpu",),
    ),
    "a100": Stratum(
        name="a100",
        constraint="a100",
        device_name_required=("a100-sxm4-80gb",),
        device_name_forbidden=("mig",),
        partitions=("kozinsky_gpu", "seas_gpu"),
    ),
    "a100_mig": Stratum(
        name="a100_mig",
        constraint="",
        device_name_required=("a100",),
        device_name_forbidden=(),
        partitions=("gpu_test",),
    ),
}

#: Measured wall limits. A partition absent from this mapping is not a partition
#: this driver may size a row against: `production-grid-v0` forbids relying on
#: restart/resume, so a row must be sized to FINISH inside a known ceiling.
PARTITION_WALL_LIMIT_MIN: Mapping[str, int] = {
    "seas_gpu": 2 * _DAY_MIN,
    "kozinsky_gpu": 7 * _DAY_MIN,
    "gpu_test": 12 * 60,
    "test": 12 * 60,
}

#: Partitions that serve GPUs at all. A GPU row on any other partition is a
#: planning error, not something to fix at submission time.
GPU_PARTITIONS: frozenset[str] = frozenset({"seas_gpu", "kozinsky_gpu", "gpu_test"})

#: Partitions this driver will place production rows on. `gpu_test` is for
#: smokes and pilots and is capped at two concurrent jobs user-wide.
PRODUCTION_GPU_PARTITIONS: frozenset[str] = frozenset({"seas_gpu", "kozinsky_gpu"})


def stratum(name: str) -> Stratum:
    """Return one declared stratum.

    Raises
    ------
    StratumError
        If ``name`` is not a declared stratum. An unknown stratum name must not
        fall back to "unpinned"; that is exactly the failure this module exists
        to prevent.
    """

    key = str(name).strip().lower()
    if key not in STRATA:
        known = ", ".join(sorted(STRATA))
        raise StratumError(f"unknown GPU stratum {name!r}; declared strata are: {known}")
    return STRATA[key]


def constraint_for(name: str) -> str:
    """Return the Slurm ``--constraint`` feature pinning one stratum."""

    return stratum(name).constraint


def wall_limit_min(partition: str) -> int:
    """Return the measured wall limit of ``partition`` in minutes.

    Raises
    ------
    StratumError
        If the partition's limit has not been measured and recorded. Guessing a
        ceiling is how a no-restart row gets sized past the wall.
    """

    key = str(partition).strip()
    if key not in PARTITION_WALL_LIMIT_MIN:
        known = ", ".join(sorted(PARTITION_WALL_LIMIT_MIN))
        raise StratumError(
            f"no measured wall limit for partition {partition!r}; "
            f"measured partitions are: {known}"
        )
    return PARTITION_WALL_LIMIT_MIN[key]


def validate_gpu_placement(*, partition: str, stratum_name: str, timeout_min: int) -> Stratum:
    """Validate one GPU row's partition, stratum, and wall time together.

    Returns
    -------
    Stratum
        The resolved stratum, so the caller records exactly what it validated.

    Raises
    ------
    StratumError
        If the partition serves no GPUs, is not a production GPU target, does
        not carry the requested stratum, or the requested wall time exceeds the
        partition's measured ceiling. The last one matters because rows may not
        resume: a row that would be cut off at the wall is a planning defect,
        not a run to be restarted.
    """

    partition = str(partition).strip()
    resolved = stratum(stratum_name)
    if partition not in GPU_PARTITIONS:
        raise StratumError(f"partition {partition!r} serves no GPUs")
    if partition not in PRODUCTION_GPU_PARTITIONS:
        raise StratumError(
            f"partition {partition!r} is a smoke/pilot target; production grid rows "
            f"must use one of: {', '.join(sorted(PRODUCTION_GPU_PARTITIONS))}"
        )
    if partition not in resolved.partitions:
        raise StratumError(
            f"stratum {resolved.name!r} is not available on partition {partition!r}; "
            f"it exists on: {', '.join(resolved.partitions)}"
        )
    limit = wall_limit_min(partition)
    if int(timeout_min) <= 0:
        raise StratumError(f"row wall time must be positive, got {timeout_min!r}")
    if int(timeout_min) > limit:
        raise StratumError(
            f"row wall time {timeout_min} min exceeds the measured {partition} ceiling "
            f"of {limit} min; rows may not resume, so size the row to finish"
        )
    return resolved


def validate_canary_gpu_placement(
    *, partition: str, stratum_name: str, timeout_min: int
) -> Stratum:
    """Validate the one policy-authorized reduced-scale canary placement.

    The canary is not allowed to choose a production fallback.  It uses the
    current Cannon ``gpu_test`` A100-MIG profile or planning fails closed.
    """

    partition = str(partition).strip()
    resolved = stratum(stratum_name)
    if partition != "gpu_test" or resolved.name != "a100_mig":
        raise StratumError(
            "He-v1 evaluation canary requires gpu_test/a100_mig; no fallback is permitted"
        )
    if partition not in resolved.partitions:
        raise StratumError(
            f"stratum {resolved.name!r} is not available on partition {partition!r}"
        )
    limit = wall_limit_min(partition)
    if int(timeout_min) <= 0 or int(timeout_min) > limit:
        raise StratumError(
            f"canary wall time must be in 1..{limit} minutes, got {timeout_min!r}"
        )
    if resolved.constraint:
        raise StratumError("gpu_test A100-MIG must not invent a Slurm node constraint")
    return resolved


def check_delivered_device(*, stratum_name: str, delivered: str | None) -> None:
    """Assert the delivered device matches the constrained stratum.

    Parameters
    ----------
    stratum_name : str
        The stratum the row was submitted under.
    delivered : str or None
        Device name read from inside the allocation. ``None`` means no device
        was visible, which fails: an unverifiable card is not a matching card.

    Raises
    ------
    DeliveredDeviceMismatch
        If no device was visible, or the visible one is not the constrained
        stratum.
    """

    resolved = stratum(stratum_name)
    if delivered is None or not str(delivered).strip():
        raise DeliveredDeviceMismatch(
            f"row requested stratum {resolved.name!r} (--constraint={resolved.constraint}) "
            "but no GPU device name was visible inside the allocation"
        )
    text = str(delivered).strip().lower()
    missing = [token for token in resolved.device_name_required if token not in text]
    if resolved.name == "a100_mig" and not (
        "mig" in text or "3g.20gb" in text or "a100 3g" in text
    ):
        missing.append("MIG/3g.20gb")
    present = [token for token in resolved.device_name_forbidden if token in text]
    if missing or present:
        raise DeliveredDeviceMismatch(
            f"delivered device {delivered!r} does not match requested stratum "
            f"{resolved.name!r} (--constraint={resolved.constraint}): "
            f"missing {missing!r}, forbidden {present!r}"
        )


def slurm_time(minutes: int) -> str:
    """Format minutes as a Slurm ``--time`` value."""

    total = int(minutes)
    if total <= 0:
        raise StratumError(f"wall time must be positive, got {minutes!r}")
    days, remainder = divmod(total, _DAY_MIN)
    hours, mins = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{mins:02d}:00"
    return f"{hours:02d}:{mins:02d}:00"


__all__ = [
    "DeliveredDeviceMismatch",
    "GPU_PARTITIONS",
    "PARTITION_WALL_LIMIT_MIN",
    "PRODUCTION_GPU_PARTITIONS",
    "STRATA",
    "Stratum",
    "StratumError",
    "check_delivered_device",
    "constraint_for",
    "slurm_time",
    "stratum",
    "validate_canary_gpu_placement",
    "validate_gpu_placement",
    "wall_limit_min",
]
