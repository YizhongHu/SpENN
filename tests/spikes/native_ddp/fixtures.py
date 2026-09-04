"""Deterministic rank-sharded fixtures for the native DDP worker."""

from __future__ import annotations

import torch


def scientific_fixture(
    world_size: int, rank: int, *, kind: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return feature and energy shards, including the M2 uneven topology."""

    if kind == "m2":
        if world_size != 3:
            raise ValueError("the m2 fixture requires world_size=3")
        shards = (
            (
                ((0.2, 0.1), (-0.5, 0.3), (0.7, -0.2), (0.4, 0.9), (-0.1, 0.6)),
                (1.0, float("nan"), 2.0, -1.0, float("nan")),
            ),
            (
                ((0.6, -0.4), (-0.2, 0.8), (0.5, 0.2)),
                (float("nan"), float("nan"), float("nan")),
            ),
            (
                ((-0.4, 0.3), (0.3, 0.7), (0.9, -0.8), (-0.7, 0.2), (0.1, 0.5), (0.2, -0.3), (0.5, 0.4)),
                (3.0, -2.0, float("nan"), 1.0, 0.5, 2.5, float("nan")),
            ),
        )
    elif kind == "regular":
        if world_size != 2:
            raise ValueError("the regular fixture requires world_size=2")
        shards = (
            (
                ((0.2, 0.1), (-0.5, 0.3), (0.7, -0.2), (0.4, 0.9)),
                (1.0, 2.0, 0.5, float("nan")),
            ),
            (
                ((-0.3, 0.8), (0.4, -0.1), (1.1, 0.2)),
                (3.0, -1.0, 2.0),
            ),
        )
    elif kind == "all_invalid":
        if world_size != 2:
            raise ValueError("the all_invalid fixture requires world_size=2")
        shards = (
            (((0.2, 0.1), (-0.5, 0.3)), (float("nan"), float("inf"))),
            (((0.7, -0.2), (0.4, 0.9), (-0.3, 0.8)), (float("nan"), float("inf"), float("nan"))),
        )
    else:
        raise ValueError(f"unknown scientific fixture {kind!r}")

    features, energy = shards[rank]
    return (
        torch.tensor(features, dtype=torch.float64),
        torch.tensor(energy, dtype=torch.float64),
    )


__all__ = ["scientific_fixture"]
