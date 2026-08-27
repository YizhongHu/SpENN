"""Facility-aware GPU strata for the He-cutover smoke."""

from __future__ import annotations

from dataclasses import dataclass

import hev1


@dataclass(frozen=True)
class PolarisStratum:
    name: str = "a100_40gb"
    required_device_substring: str = "a100-sxm4-40gb"
    queue: str = "debug"
    wall_limit_min: int = 60


POLARIS = PolarisStratum()
POLARIS_CAPACITY = PolarisStratum(queue="capacity", wall_limit_min=168 * 60)


def validate_placement(*, facility: str, partition: str, stratum: str, timeout_min: int):
    """Validate a Cannon or Polaris placement without weakening either policy."""

    if facility == "cannon":
        return hev1.strata.validate_canary_gpu_placement(
            partition=partition, stratum_name=stratum, timeout_min=timeout_min
        )
    if facility != "polaris":
        raise ValueError(f"unknown facility {facility!r}")
    placements = {item.queue: item for item in (POLARIS, POLARIS_CAPACITY)}
    placement = placements.get(partition)
    if placement is None or stratum != placement.name:
        raise ValueError("Polaris requires debug or capacity on a100_40gb")
    if timeout_min <= 0 or timeout_min > placement.wall_limit_min:
        raise ValueError(
            f"Polaris {partition} wall time must be in 1..{placement.wall_limit_min} minutes"
        )
    return placement


def check_delivered_device(*, facility: str, stratum: str, delivered: str | None) -> None:
    """Fail unless the allocation delivered the requested device stratum."""

    if facility == "cannon":
        hev1.strata.check_delivered_device(stratum_name=stratum, delivered=delivered)
        return
    if stratum != POLARIS.name or POLARIS.required_device_substring not in str(delivered).lower():
        raise RuntimeError(
            f"delivered device {delivered!r} does not match Polaris {POLARIS.name!r}; "
            f"required substring {POLARIS.required_device_substring!r}"
        )
