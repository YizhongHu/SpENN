"""Unit tests for the Checkpoint callback."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import tpen
from tpen.artifacts import RunContext
from tpen.callback import Checkpoint, Event
from tpen.checkpoint import checkpoint_hashes
from tpen.events import Occurrence
from tpen.training.events import TrainingIteration, UpdateCompleted
from tpen.training.state import TrainerState


class _CheckpointContext(RunContext):
    """Minimal `RunContext` carrying only what `save_checkpoint` reads.

    This subclasses `RunContext` rather than duck-typing a `SimpleNamespace`
    because `tpen.callback.Callback.handle_occurrence` annotates its context
    parameter and the suite runs typeguard over the ``tpen`` package, so the
    typed dispatch path rejects a stand-in that is not really a `RunContext`.
    The legacy ``handle`` path is unannotated and accepted either way, which is
    why a plain namespace sufficed before these tests began driving typed
    occurrences.

    The dataclass ``__init__`` is bypassed on purpose -- an artifact manager,
    clock, and logger list are irrelevant to checkpoint writing -- and
    ``run_dir`` is overridden because the base class resolves it through the
    artifact manager this stub never builds. Mirrors the established
    ``tests/unit/callback/support.py::RecordingContext`` pattern.
    """

    def __init__(self, cfg, metadata, run_dir: str) -> None:
        self.cfg = cfg
        self.metadata = metadata
        self._run_dir = Path(run_dir)

    @property
    def run_dir(self) -> Path:
        """Return the stubbed run directory."""

        return self._run_dir

    def __repr__(self) -> str:
        # The inherited dataclass repr reads fields this stub never sets, so it
        # would raise while pytest renders a failure. Keep failures readable.
        return f"_CheckpointContext(run_dir={self._run_dir!r})"


def _state(
    step: int,
    *,
    next_iteration: int | None = None,
    completed_updates: int | None = None,
) -> TrainerState:
    """Build loop state for one completed iteration.

    Both counters default to ``step + 1``, the ordinary case where every
    attempted iteration applied its optimizer update. Pass them separately to
    model a vacuum iteration, where the cursor advances but the update count
    does not.
    """

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = _Trainer(
        next_iteration=step + 1 if next_iteration is None else next_iteration,
        completed_updates=step + 1 if completed_updates is None else completed_updates,
    )
    return TrainerState(
        step=step,
        metrics={"loss": 0.5, "energy": 1.25},
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=_SamplerWithMCMCState(),
    )


class _Trainer:
    def __init__(self, *, next_iteration: int, completed_updates: int) -> None:
        self.next_iteration = int(next_iteration)
        self.completed_updates = int(completed_updates)

    def state_dict(self) -> dict[str, int]:
        # Directory numbering follows the resume cursor; cadence follows the
        # applied-update count. Both are required manifest fields.
        return {
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }


class _SamplerWithMCMCState:
    def mcmc_state_dict(self) -> dict:
        return {"has_burned_in": True}


def _event(state: TrainerState, *, context=None, name: str = "step_end") -> Event:
    return Event(
        name=name,
        context=_context() if context is None else context,
        state=state,
        payload={"step": state.step},
        step=state.step,
    )


def _iteration(
    callback: Checkpoint,
    state: TrainerState,
    context,
    *,
    applied_update: bool = True,
    occurrence_count: int = 1,
) -> None:
    """Replay one trainer iteration's callback deliveries in loop order.

    Mirrors `tpen.training.VMCTrainer.fit`: an iteration that applied its
    optimizer update emits the typed `UpdateCompleted` immediately after
    ``optimizer.step()`` returns, then the legacy ``step_end``. A vacuum
    iteration emits `UpdateSkipped` instead, which `Checkpoint` does not
    subscribe to, so ``applied_update=False`` delivers nothing typed at all.

    The same `context` object must reach both paths, exactly as at runtime:
    `tpen.callback.Callback` resets typed state when the context identity
    changes.

    Parameters
    ----------
    occurrence_count : int, optional
        Run-local occurrence coordinate carried by the typed event. Varied
        independently of ``completed_updates`` so tests can pin that cadence
        follows the durable counter rather than this one.
    """

    if applied_update:
        callback.handle_occurrence(
            Occurrence(
                event=UpdateCompleted(iteration=TrainingIteration(step=state.step)),
                count=occurrence_count,
            ),
            context,
        )
    callback.handle(_event(state, context=context))


def test_checkpoint_writes_step_directory_and_latest_pointer(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path / "checkpoints", every_n_steps=1)

    _iteration(callback, _state(2), _context())

    ckpt_dir = tmp_path / "checkpoints"
    step_dir = ckpt_dir / "step_000003"
    assert step_dir.is_dir()
    assert (step_dir / "manifest.json").exists()
    assert (step_dir / "resolved_config.yaml").exists()
    assert (step_dir / "model.pt").exists()
    assert (step_dir / "optimizer.pt").exists()
    assert (step_dir / "trainer.json").exists()
    assert (step_dir / "sampler.pt").exists()
    assert (step_dir / "rng.pt").exists()
    assert (step_dir / "COMPLETE").exists()
    assert (ckpt_dir / "latest.json").exists()
    assert not (ckpt_dir / "step_000003.tmp").exists()


def test_checkpoint_payload_contains_expected_keys(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(3)

    _iteration(callback, state, _context())

    manifest = torch.load(tmp_path / "step_000004" / "model.pt", weights_only=False)
    assert set(manifest) == set(state.model.state_dict())
    sampler_state = torch.load(tmp_path / "step_000004" / "sampler.pt", weights_only=False)
    assert sampler_state == {"has_burned_in": True}


def test_checkpoint_respects_every_n_steps_filter(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=2)
    context = _context()

    _iteration(callback, _state(0), context)
    assert not (tmp_path / "step_000001").exists()

    _iteration(callback, _state(1), context)
    assert (tmp_path / "step_000002").exists()


def test_checkpoint_cadence_counts_completed_updates(tmp_path) -> None:
    """Cadence is spaced by applied updates; identity stays the resume cursor."""

    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=5)

    # Five completed updates hits the cadence, but the run attempted six
    # iterations, so the directory is named for the cursor, not the count.
    _iteration(callback, _state(5, next_iteration=6, completed_updates=5), _context())

    assert (tmp_path / "step_000006").exists()
    assert not (tmp_path / "step_000005").exists()


def test_checkpoint_cadence_fires_once_per_update_completed(tmp_path) -> None:
    """One firing per `UpdateCompleted`; a skipped update fires nothing."""

    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    context = _context()

    # Iteration 0 applies its update: cursor 1, one completed update.
    _iteration(callback, _state(0, next_iteration=1, completed_updates=1), context)
    # Iteration 1 is a zero-electron vacuum. It emits `UpdateSkipped`, which
    # this callback does not subscribe to, so no typed delivery arrives and the
    # gate cannot fire -- even though the cursor advanced and every_n_steps=1
    # would otherwise admit the unchanged count.
    _iteration(
        callback,
        _state(1, next_iteration=2, completed_updates=1),
        context,
        applied_update=False,
    )
    # Iteration 2 applies an update again, so a typed delivery arrives and the
    # gate fires exactly once more.
    _iteration(callback, _state(2, next_iteration=3, completed_updates=2), context)

    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000001",
        "step_000003",
    ]


def test_checkpoint_cadence_phase_survives_resume(tmp_path) -> None:
    """Cadence follows the durable counter, not the run-local occurrence count.

    A resumed run restarts `Occurrence.count` at 1. If that were the cadence
    coordinate, the resumed run would checkpoint at different points than an
    uninterrupted run that reached the same ``completed_updates``. Gating on
    the trainer's durable counter keeps the two aligned.
    """

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = Checkpoint(
        triggers=["step_end"], output_dir=uninterrupted_dir, every_n_steps=3
    )
    context = _context()
    # Nine iterations, each applying its update: completed_updates 1..9.
    for step in range(9):
        _iteration(uninterrupted, _state(step), context, occurrence_count=step + 1)

    resumed_dir = tmp_path / "resumed"
    resumed = Checkpoint(triggers=["step_end"], output_dir=resumed_dir, every_n_steps=3)
    resumed_context = _context()
    # Restored mid-phase at completed_updates=4, so the next admitted count is
    # 6, not 4 + 3. Occurrence counts restart at 1 for the new context.
    for offset, step in enumerate(range(4, 9)):
        _iteration(resumed, _state(step), resumed_context, occurrence_count=offset + 1)

    uninterrupted_names = sorted(path.name for path in uninterrupted_dir.glob("step_*"))
    resumed_names = sorted(path.name for path in resumed_dir.glob("step_*"))
    assert uninterrupted_names == ["step_000003", "step_000006", "step_000009"]
    # Same checkpoint points, minus the ones already written before the restore.
    assert resumed_names == ["step_000006", "step_000009"]
    # Occurrence-count cadence would have fired on the resumed run's third
    # delivery, at completed_updates=7. It must not.
    assert not (resumed_dir / "step_000007").exists()


def test_checkpoint_without_cadence_still_requires_a_completed_update(tmp_path) -> None:
    """No `every_n_steps` means every completed update, not every iteration."""

    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path)
    context = _context()

    _iteration(callback, _state(0, next_iteration=1, completed_updates=1), context)
    assert (tmp_path / "step_000001").exists()

    # The base class consults the cadence hook only when `every_n_steps` is
    # set, so update selection cannot live there: a vacuum iteration would
    # otherwise write a checkpoint here.
    _iteration(
        callback,
        _state(1, next_iteration=2, completed_updates=1),
        context,
        applied_update=False,
    )
    assert not (tmp_path / "step_000002").exists()


def test_train_end_checkpoint_survives_a_final_vacuum_iteration(tmp_path) -> None:
    """Update selection must not reach the terminal trigger."""

    callback = Checkpoint(
        triggers=["step_end", "train_end"], output_dir=tmp_path, every_n_steps=1
    )
    context = _context()

    _iteration(callback, _state(0, next_iteration=1, completed_updates=1), context)
    # The run's last iteration is a zero-electron vacuum, so no periodic
    # checkpoint fires and the armed step stays behind `state.step`.
    vacuum = _state(1, next_iteration=2, completed_updates=1)
    _iteration(callback, vacuum, context, applied_update=False)
    # `train_end` carries that same mutated state. It must still write.
    callback.handle(_event(vacuum, context=context, name="train_end"))

    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000001",
        "step_000002",
    ]


def test_train_end_checkpoint_written_when_the_loop_body_never_ran(tmp_path) -> None:
    """`max_steps=0` or a fully-resumed run still writes its terminal checkpoint."""

    callback = Checkpoint(triggers=["train_end"], output_dir=tmp_path, every_n_steps=1)
    # No iteration executed: `TrainerState.step` is still -1 and no
    # `UpdateCompleted` was ever delivered, but the resume cursor is 5.
    state = _state(-1, next_iteration=5, completed_updates=5)

    callback.handle(_event(state, name="train_end"))

    assert (tmp_path / "step_000005" / "COMPLETE").exists()


def test_checkpoint_writes_train_end_checkpoint_without_step_cadence(tmp_path) -> None:
    callback = Checkpoint(triggers=["train_end"], output_dir=tmp_path)
    state = _state(3, next_iteration=4)

    callback.handle(_event(state, name="train_end"))

    assert (tmp_path / "step_000004" / "COMPLETE").exists()
    assert (tmp_path / "latest.json").exists()


def test_checkpoint_train_end_skips_existing_complete_checkpoint(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end", "train_end"], output_dir=tmp_path, every_n_steps=1)
    context = _context()
    state = _state(1, next_iteration=2)

    _iteration(callback, state, context)
    callback.handle(_event(state, context=context, name="train_end"))

    assert sorted(path.name for path in tmp_path.glob("step_*")) == ["step_000002"]


def test_train_end_updates_latest_when_cadence_misses_terminal_step(tmp_path) -> None:
    periodic = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=2)
    terminal = Checkpoint(triggers=["train_end"], output_dir=tmp_path)
    context = _context()

    _iteration(periodic, _state(1, next_iteration=2), context)
    final_state = _state(2, next_iteration=3)
    _iteration(periodic, final_state, context)
    terminal.handle(_event(final_state, context=context, name="train_end"))

    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000002",
        "step_000003",
    ]
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["checkpoint_dir"] == "step_000003"
    assert latest["step"] == 3


def _context() -> _CheckpointContext:
    """Build a real `RunContext` carrying resolved config and metadata."""

    cfg = OmegaConf.create(
        {
            "study": {"name": "test_study", "config_id": "lr=0.001_channels=4"},
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 1},
            "runtime": {"device": "cpu", "dtype": "float64"},
        }
    )
    metadata = SimpleNamespace(
        run_id="run",
        device="cpu",
        dtype="float64",
        git_commit="deadbeef",
        git_branch="main",
        dirty_worktree=False,
        command="pytest",
        extra={"python_version": "3.12.0", "torch_version": torch.__version__},
    )
    # `metadata` stays a namespace: it is only ever read attribute-wise by
    # `save_checkpoint`, which annotates its context as `Any`. Only the context
    # itself has to be a real `RunContext`, because it crosses the annotated
    # typed-occurrence dispatch boundary.
    return _CheckpointContext(cfg=cfg, metadata=metadata, run_dir="/tmp/run")


def test_checkpoint_payload_uses_structured_schema(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    context = _context()
    state = _state(1)

    callback.handle_occurrence(
        Occurrence(event=UpdateCompleted(iteration=TrainingIteration(step=1)), count=1),
        context,
    )
    callback.handle(
        Event(
            name="step_end",
            context=context,
            state=state,
            payload={"step": 1},
            step=1,
        )
    )

    manifest = json.loads((tmp_path / "step_000002" / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["kind"] == "tpen.checkpoint"
    # v2 names both counters instead of one ambiguous `step`.
    assert "step" not in manifest
    assert manifest["next_iteration"] == 2
    assert manifest["completed_updates"] == 2
    assert manifest["files"]["model"] == "model.pt"
    assert manifest["hashes"] == checkpoint_hashes(context.cfg)
    assert manifest["runtime"]["device"] == "cpu"
    assert manifest["runtime"]["dtype"] == "float64"
    assert manifest["runtime"]["torch_version"] == torch.__version__
    assert manifest["provenance"]["config_id"] == "lr=0.001_channels=4"
    assert manifest["provenance"]["study_name"] == "test_study"
    assert manifest["provenance"]["git_sha"] == "deadbeef"
    assert manifest["provenance"]["tpen_version"] == tpen.__version__


def test_checkpoint_fails_loudly_when_required_state_is_missing(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(1)
    state.trainer = None

    with pytest.raises(ValueError, match="trainer"):
        callback.handle(_event(state))


def test_checkpoint_uses_inherited_should_run_with_owned_step_hook(tmp_path) -> None:
    assert "should_run" not in Checkpoint.__dict__
    assert "_legacy_cadence_step" in Checkpoint.__dict__
    callback = Checkpoint(
        triggers=["step_end"],
        output_dir=tmp_path,
        every_n_steps=2,
    )
    context = _context()
    state = _state(1, next_iteration=2, completed_updates=2)
    callback.handle_occurrence(
        Occurrence(event=UpdateCompleted(iteration=TrainingIteration(step=1)), count=1),
        context,
    )

    # The cadence coordinate comes from the trainer's typed state alone. The
    # payload's `step` is not a cadence input and must not move the gate.
    event = Event(
        name="step_end",
        context=context,
        state=state,
        payload={"step": 1},
    )

    assert callback.should_run(event)


def test_checkpoint_cadence_ignores_payload_and_state_step(tmp_path) -> None:
    """Only `completed_updates` gates cadence -- no step-shaped fallbacks."""

    callback = Checkpoint(
        triggers=["step_end"],
        output_dir=tmp_path,
        every_n_steps=2,
    )
    context = _context()
    callback.handle_occurrence(
        Occurrence(event=UpdateCompleted(iteration=TrainingIteration(step=2)), count=2),
        context,
    )
    # The counters are deliberately diverged, so this pins which one the window
    # reads: `completed_updates=3` is odd and closes the gate, while the
    # resume cursor `next_iteration=4` would have opened it. The payload and
    # `state.step` values would also have passed a step-based gate.
    event = Event(
        name="step_end",
        context=context,
        state=_state(2, next_iteration=4, completed_updates=3),
        payload={"step": 2},
        step=2,
    )

    assert not callback.should_run(event)


def test_checkpoint_cadence_requires_a_trainer_on_state(tmp_path) -> None:
    """No `global_step`/`state.step` probing survives: the trainer is required."""

    callback = Checkpoint(
        triggers=["train_end"],
        output_dir=tmp_path,
        every_n_steps=2,
    )
    event = Event(
        name="train_end",
        context=_context(),
        state=SimpleNamespace(global_step=2, step=1, trainer=None),
    )

    with pytest.raises(ValueError, match="trainer"):
        callback.should_run(event)
