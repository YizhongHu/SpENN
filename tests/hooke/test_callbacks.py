"""Callback dispatch tests for configured runs."""

from __future__ import annotations

from pathlib import Path

from spenn.artifacts import RunContext
from spenn.callback import Callback, Event
from spenn.runner import Scaffold
from spenn.run import load_config, prepare_run_context

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "hooke" / "configs" / "smoke" / "scaffold.yaml"


class RecordingCallback(Callback):
    """Record handled event names for callback registry tests."""

    def __init__(self, triggers: list[str]) -> None:
        super().__init__(triggers)
        self.events: list[str] = []

    def on_run_start(self, event: Event) -> None:
        """Record run start."""

        self.events.append(event.name)

    def on_run_end(self, event: Event) -> None:
        """Record run end."""

        self.events.append(event.name)


def test_callback_registry_dispatches_matching_events(tmp_path: Path) -> None:
    """Runner callback dispatch uses subscribed event names."""

    callback = RecordingCallback(["run_start"])
    runner = Scaffold(callbacks=[callback], loggers=[])
    context = _context(tmp_path)

    runner.emit("run_start", context)
    runner.emit("run_end", context)

    assert callback.events == ["run_start"]


def test_periodic_callback_filters_by_step(tmp_path: Path) -> None:
    """Step filters run only on eligible event steps."""

    callback = RecordingCallback(["step_end"])
    callback.every_n_steps = 2
    runner = Scaffold(callbacks=[callback], loggers=[])
    context = _context(tmp_path)

    runner.emit("step_end", context, payload={"step": 0})
    runner.emit("step_end", context, payload={"step": 1})
    runner.emit("step_end", context, payload={"step": 2})

    assert callback.events == []
    assert callback.num_calls == 2


def _context(tmp_path: Path) -> RunContext:
    cfg = load_config(str(CONFIG), [f"run.root={tmp_path}"])
    return prepare_run_context(cfg, config_path=str(CONFIG), command="pytest callbacks")
