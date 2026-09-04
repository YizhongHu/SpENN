"""Small typed c10d runtime surface for the native DDP spike.

The runtime is deliberately a proposal-local adapter.  It owns no application
state and exposes only the CPU/Gloo collectives needed by the spike worker.
Production TPEN does not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch.distributed as dist


@dataclass(frozen=True)
class DistributedRuntime:
    """Immutable identity and collective facade for one c10d process."""

    rank: int
    world_size: int
    process_group_timeout_seconds: float
    backend: str = "gloo"

    def __post_init__(self) -> None:
        if self.backend != "gloo":
            raise ValueError("the native spike is CPU/Gloo-only")
        if type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be an integer in the runtime world")
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if self.process_group_timeout_seconds <= 0:
            raise ValueError("process_group_timeout_seconds must be positive")

    @classmethod
    def initialize(
        cls,
        *,
        rank: int,
        world_size: int,
        rendezvous_file: Path,
        process_group_timeout_seconds: float,
    ) -> "DistributedRuntime":
        """Initialize one fresh FileStore process group and return its identity."""

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=process_group_timeout_seconds),
        )
        return cls(
            rank=rank,
            world_size=world_size,
            process_group_timeout_seconds=process_group_timeout_seconds,
        )

    def barrier(self) -> None:
        """Synchronize every rank in this runtime."""

        dist.barrier()

    def all_gather_objects(self, value: Any) -> list[Any]:
        """Gather one Python value from every rank in rank order."""

        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def broadcast_object(self, value: Any | None, *, source: int = 0) -> Any:
        """Broadcast one Python value from ``source`` and return it everywhere."""

        payload = [value if self.rank == source else None]
        dist.broadcast_object_list(payload, src=source)
        return payload[0]

    def collective_scalar_sum(self, value: float) -> float:
        """Return a scalar sum used only for non-gradient diagnostics."""

        tensor = _scalar_tensor(value)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def close(self) -> None:
        """Destroy this invocation's process group when it is still live."""

        if dist.is_initialized():
            dist.destroy_process_group()


def _scalar_tensor(value: float) -> "torch.Tensor":
    """Build a CPU scalar without exposing a device-selection surface."""

    import torch

    return torch.tensor(float(value), dtype=torch.float64)


__all__ = ["DistributedRuntime"]
