"""Deterministic rank partition and exact process-local RNG sidecars."""

from __future__ import annotations

import base64
import pickle
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SeedPartition:
    """Immutable seed assignment for one exact world topology."""

    base_seed: int
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or type(self.world_size) is not int:
            raise TypeError("rank and world_size must be integers")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must belong to the seed partition")

    @property
    def rank_seed(self) -> int:
        """Derive a stable, non-overlapping rank-local seed."""

        return int(self.base_seed + 1_000_003 * self.rank)

    def make_generator(self) -> torch.Generator:
        """Create the CPU generator owned by this rank's sampler."""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.rank_seed)
        return generator


def seed_global_rngs(seed: int) -> None:
    """Seed Python, NumPy, and Torch CPU RNGs for a worker process."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def capture_global_rng_state() -> dict[str, Any]:
    """Capture all global RNG state used by the spike worker."""

    return {
        "python": _encode_pickle(random.getstate()),
        "numpy": _encode_pickle(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().tolist(),
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    """Restore the three global RNG streams without reinterpretation."""

    random.setstate(_decode_pickle(str(state["python"])))
    np.random.set_state(_decode_pickle(str(state["numpy"])))
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))


def _encode_pickle(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=5)).decode("ascii")


def _decode_pickle(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


__all__ = [
    "SeedPartition",
    "capture_global_rng_state",
    "restore_global_rng_state",
    "seed_global_rngs",
]
