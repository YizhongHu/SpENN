"""Study-specific run-id grammar for the original pair-stability grid."""

from __future__ import annotations

import re
from typing import Any

from utils.naming import format_lr

GRID_AXES = ("architecture", "normalization", "lr", "channels", "seed")


def run_id_for(point: dict[str, Any]) -> str:
    """Return the deterministic run id for one grid point."""

    return (
        f"arch-{point['architecture']}"
        f"_norm-{point['normalization']}"
        f"_lr-{format_lr(point['lr'])}"
        f"_ch-{int(point['channels'])}"
        f"_seed-{int(point['seed'])}"
    )


_RUN_ID_PATTERN = re.compile(
    r"^arch-(?P<architecture>.+)_norm-(?P<normalization>[^_]+)"
    r"_lr-(?P<lr>[^_]+)_ch-(?P<channels>\d+)_seed-(?P<seed>\d+)$"
)


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Recover the grid choices encoded in a run id."""

    match = _RUN_ID_PATTERN.match(run_id)
    if match is None:
        raise ValueError(f"run id {run_id!r} does not match the pair_stability convention")
    fields = match.groupdict()
    return {
        "architecture": fields["architecture"],
        "normalization": fields["normalization"],
        "lr": fields["lr"],
        "channels": int(fields["channels"]),
        "seed": int(fields["seed"]),
    }
