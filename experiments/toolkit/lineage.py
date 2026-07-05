"""Task-lineage sidecars threading task ids through aggregation stages.

Fan-out stages (e.g. ``02_validation``) assign each row a deterministic
task id (see :func:`specs.task_id_from_parts`). Aggregation stages downstream
(collect, select, final_plan, ...) keep their existing outputs unchanged --
those are byte-compared against ``pair_stability_v2`` by ``parity.py``, which
has no toolkit integration and can never grow a new column or key. Instead,
each aggregation stage writes a ``task_lineage.jsonl`` sidecar next to its
regular outputs, and reads the upstream stage's sidecar (when one exists) to
extend the chain.

A row's task id never requires a file read to compute: it is deterministic
from ``(stage, run_id, attempt_id)``, all of which an aggregation stage
already holds. :func:`stage_plan_task_ids` and :func:`synthesized_task_id`
verify that synthesized id against the upstream stage's real ``tasks.jsonl``
whenever one exists, to catch synthesis bugs; the check is best-effort
because v2/v3 parity runs point collect at reused ``pair_stability_v2``
validation data that has no ``stage_plans/`` directory at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .jsonio import read_jsonl, write_jsonl
from .specs import StagePlan, task_id_from_parts

LINEAGE_JSONL = "task_lineage.jsonl"


@dataclass(frozen=True)
class TaskLineageRow:
    """One aggregation row's upstream task-id lineage."""

    row_id: str
    task_ids: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {"row_id": self.row_id, "task_ids": dict(self.task_ids)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskLineageRow":
        """Return a row parsed from its JSON-safe representation."""

        return cls(row_id=str(payload["row_id"]), task_ids=dict(payload.get("task_ids", {})))


def write_task_lineage(directory: str | Path, rows: list[TaskLineageRow]) -> Path:
    """Write a stage's ``task_lineage.jsonl`` sidecar and return its path."""

    path = Path(directory) / LINEAGE_JSONL
    write_jsonl(path, (row.to_dict() for row in rows))
    return path


def read_task_lineage(directory: str | Path) -> dict[str, TaskLineageRow]:
    """Read a stage's ``task_lineage.jsonl`` sidecar, keyed by ``row_id``.

    Returns an empty mapping when the sidecar does not exist, so consuming a
    stage that predates this sidecar (or reused v2 data) is a no-op rather
    than an error.
    """

    path = Path(directory) / LINEAGE_JSONL
    if not path.is_file():
        return {}
    return {str(row["row_id"]): TaskLineageRow.from_dict(row) for row in read_jsonl(path)}


def stage_plan_task_ids(stage_plan_dir: str | Path) -> frozenset[str] | None:
    """Return the task ids recorded by a real stage plan, or ``None`` if absent."""

    try:
        plan = StagePlan.read(stage_plan_dir)
    except FileNotFoundError:
        return None
    return frozenset(task.task_id for task in plan.tasks)


def synthesized_task_id(
    *,
    stage: str,
    run_id: str,
    attempt_id: str,
    known_task_ids: frozenset[str] | None,
) -> str:
    """Return the deterministic task id, verified against ``known_task_ids`` if given."""

    task_id = task_id_from_parts(stage=stage, run_id=run_id, attempt_id=attempt_id)
    if known_task_ids is not None and task_id not in known_task_ids:
        raise ValueError(f"synthesized task_id {task_id!r} is not a known task id for stage {stage!r}")
    return task_id
