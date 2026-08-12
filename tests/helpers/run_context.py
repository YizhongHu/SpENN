"""Real `tpen.artifacts.RunContext` construction for tests that need dispatch.

A test that only calls a callback's handler directly can use a lightweight
stand-in. A test that has to prove *delivery* cannot: the routing it is trying
to exercise runs through `RunContext._dispatch_occurrence`, the very method a
stand-in overrides. These helpers build the genuine article instead, writing its
artifacts under a ``tmp_path``.

Since item ``24f91145`` the ``isinstance(state, callback.state_type)`` decision
lives one level down, in `tpen.callback.StatefulCallback.handle_occurrence`, and
is taken once per subscription GROUP rather than once per callback. Both halves
are needed to observe a delivery, which is a further reason a dispatcher
stand-in cannot substitute for the real one here.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.logging import LogRecord


class RecordingLogger:
    """Logger double capturing every `tpen.logging.LogRecord` it is handed.

    Attaching this to a real `RunContext` keeps `RunContext.log` on its
    production path, so a test observes what the configured loggers would have
    received rather than what a patched ``log`` chose to record.
    """

    def __init__(self) -> None:
        self.records: list[LogRecord] = []

    def log(self, record: LogRecord) -> None:
        """Capture one metric record."""

        self.records.append(record)

    def namespaces(self) -> set[str]:
        """Return every namespace seen so far."""

        return {record.namespace for record in self.records}

    def by_namespace(self, namespace: str) -> list[LogRecord]:
        """Return the records logged under ``namespace``, in order."""

        return [record for record in self.records if record.namespace == namespace]

    def steps(self, namespace: str) -> list[int | None]:
        """Return the step coordinate of each record under ``namespace``."""

        return [record.step for record in self.by_namespace(namespace)]

    def latest(self, namespace: str) -> dict[str, Any]:
        """Return the metrics of the most recent record under ``namespace``."""

        return dict(self.by_namespace(namespace)[-1].metrics)


def make_run_context(
    tmp_path: Path,
    *,
    callbacks: list[Any] | None = None,
    loggers: list[Any] | None = None,
    run_id: str = "helper-run",
    device: str = "cpu",
) -> RunContext:
    """Return a real `RunContext` writing artifacts under ``tmp_path``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Root for the run directory, normally pytest's ``tmp_path`` fixture.
    callbacks : list, optional
        Callbacks the context dispatches to, in order.
    loggers : list, optional
        Loggers `RunContext.log` writes to, in order.
    run_id : str, optional
        Run identifier used for both the run name and the directory name.
    device : str, optional
        Device string recorded in the metadata; the trainer reads it.
    """

    artifact_manager = ArtifactManager(
        tmp_path,
        experiment=run_id,
        sector="unit",
        run_id=run_id,
        layout="flat",
    )
    artifact_manager.make_dirs()
    metadata = RunMetadata(
        run_id=run_id,
        run_name=run_id,
        timestamp="2026-08-11T12:00:00+00:00",
        timezone="UTC",
        git_commit="test-sha",
        git_branch="test-branch",
        dirty_worktree=False,
        command="pytest",
        config_path="test.yaml",
        resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
        run_dir=str(artifact_manager.run_dir),
        device=device,
        dtype="float64",
    )
    return RunContext(
        cfg=OmegaConf.create({}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=RunClock(timezone="UTC", tzinfo=UTC),
        callbacks=[] if callbacks is None else callbacks,
        loggers=[] if loggers is None else loggers,
    )


__all__ = ["RecordingLogger", "make_run_context"]
