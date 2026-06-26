"""Executor interfaces for future backend replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .execution import ExecutionRecord
from .specs import StagePlan, TaskSpec


class Executor(Protocol):
    """Protocol implemented by local, Submitit, and future executors."""

    def submit(self, plan: StagePlan, tasks: Sequence[TaskSpec]) -> Sequence[ExecutionRecord]:
        """Submit ``tasks`` from ``plan`` and return execution records."""


@dataclass(frozen=True)
class ExecutorOptions:
    """Backend-neutral options understood by executor adapters."""

    backend: str
    chunk_size: int = 1
    allow_partial_failures: bool = False
    claim_rows: bool = False
