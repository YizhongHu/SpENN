"""Generic run artifact helpers for configured TPEN executions."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import socket
import subprocess
import time
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from datetime import UTC, datetime, tzinfo
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from omegaconf import DictConfig, OmegaConf

from tpen.events import DomainState, Ended, Event as TypedEvent, Occurrence, Operation, Started

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUN_DIRS = ("checkpoints", "checks", "diagnostics")
DEFAULT_RUN_TIMEZONE = "UTC"
RUN_START_ENV_ALLOWLIST = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_PARTITION",
    "CUDA_VISIBLE_DEVICES",
)

_EventT = TypeVar("_EventT", bound=TypedEvent)
_OperationT = TypeVar("_OperationT", bound=Operation)


class ArtifactManager:
    """Own the standard artifact layout for one run.

    Parameters
    ----------
    root : pathlib.Path or str
        Output root. Relative paths are interpreted relative to the
        repository root.
    experiment : str
        Experiment family name.
    sector : str
        Experiment sector or suite name.
    run_id : str
        Unique run identifier.
    layout : {"nested", "flat"}
        Directory layout. ``nested`` writes
        ``root/experiment/sector/run_id``; ``flat`` writes ``root/run_id``.
    """

    def __init__(
        self,
        root: Path | str,
        experiment: str,
        sector: str,
        run_id: str,
        *,
        layout: str = "nested",
    ) -> None:
        root_path = Path(root)
        self.root = root_path if root_path.is_absolute() else ROOT / root_path
        self.experiment = str(experiment)
        self.sector = str(sector)
        self.run_id = str(run_id)
        self.layout = str(layout)
        if self.layout not in {"nested", "flat"}:
            raise ValueError(f"unsupported artifact layout {self.layout!r}; expected 'nested' or 'flat'")

    @property
    def run_dir(self) -> Path:
        """Return the run directory path."""

        if self.layout == "flat":
            return self.root / self.run_id
        return self.root / self.experiment / self.sector / self.run_id

    def make_dirs(self) -> None:
        """Create the run directory and standard child directories."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_RUN_DIRS:
            self.path(name).mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        """Return a path under this run directory."""

        return self.run_dir.joinpath(*parts)


@dataclass
class RunMetadata:
    """Execution metadata captured for one configured run."""

    run_id: str
    run_name: str
    timestamp: str
    timezone: str
    git_commit: str
    git_branch: str
    dirty_worktree: bool
    command: str | None
    config_path: str | None
    resolved_config_path: str
    run_dir: str
    device: str
    dtype: str
    status: str = "initialized"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""

        data = asdict(self)
        extra = data.pop("extra")
        data.update(extra)
        return data


@dataclass
class RunResult:
    """Result returned by a configured runner."""

    status: str
    run_dir: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class RunClock:
    """Run-wide wall-clock convention."""

    timezone: str
    tzinfo: tzinfo

    def now(self) -> datetime:
        """Return the current wall-clock time in the run timezone."""

        return datetime.now(self.tzinfo)

    def now_iso(self) -> str:
        """Return an ISO timestamp in the run timezone."""

        return self.now().isoformat()


@dataclass
class RunContext:
    """Runtime context shared by runners, callbacks, and loggers."""

    cfg: DictConfig
    source_cfg: DictConfig
    artifact_manager: ArtifactManager
    metadata: RunMetadata
    clock: RunClock
    callbacks: list[Any] = field(default_factory=list)
    loggers: list[Any] = field(default_factory=list)
    _occurrence_counts: dict[type[TypedEvent] | type[Operation], int] = field(
        default_factory=dict, init=False, repr=False
    )
    monotonic_clock: Callable[[], float] = field(
        default=time.perf_counter, repr=False
    )

    @property
    def run_dir(self) -> Path:
        """Return the active run directory."""

        return self.artifact_manager.run_dir

    def path(self, *parts: str) -> Path:
        """Return a path under the active run directory."""

        return self.artifact_manager.path(*parts)

    def now(self) -> datetime:
        """Return the current wall-clock time in the run timezone."""

        return self.clock.now()

    def now_iso(self) -> str:
        """Return an ISO timestamp in the run timezone."""

        return self.clock.now_iso()

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
        namespace: str = "run",
    ) -> None:
        """Emit one metric record to every configured logger."""

        from tpen.logging import LogRecord

        record = LogRecord(step=step, namespace=namespace, metrics=dict(metrics))
        for logger in self.loggers:
            logger.log(record)

    def emit(
        self, event: _EventT, *, state: DomainState | None = None
    ) -> Occurrence[_EventT]:
        """Record and dispatch the next occurrence of a typed event.

        Counts are one-based and local to this context. Each concrete event
        type advances independently.

        Parameters
        ----------
        event : Event
            Typed instantaneous event to emit.
        state : DomainState or None, optional
            The emitting domain's state object, delivered to typed handlers
            that declare this domain. ``None`` means this boundary offers no
            domain state, which is how every state-free emitter behaves.
        """

        if not isinstance(event, TypedEvent):
            raise TypeError(f"event must be an Event, got {type(event).__name__}")
        if isinstance(event, (Started, Ended)):
            raise TypeError("Started and Ended are emitted only by scope(operation)")
        if state is not None and not isinstance(state, DomainState):
            raise TypeError(f"state must be a DomainState, got {type(state).__name__}")
        occurrence = self._occurrence(
            event=event, count=self._next_occurrence_count(type(event))
        )
        self._dispatch_occurrence(occurrence, state=state)
        return occurrence

    @contextmanager
    def scope(
        self, operation: _OperationT, *, state: DomainState | None = None
    ) -> Iterator[Occurrence[Started[_OperationT]]]:
        """Emit paired lifecycle records around one typed operation.

        The operation's concrete type advances its counter once on entry. The
        resulting ``Started`` and ``Ended`` records share that count. After a
        successful start dispatch, the end record is dispatched even when the
        body raises.

        Parameters
        ----------
        operation : Operation
            Typed operation to bracket.
        state : DomainState or None, optional
            The emitting domain's state object. Both boundaries carry the same
            reference, so a handler at the ended boundary observes whatever the
            scope body mutated in place. That is intended: the event says only
            when the boundary happened, and the state is read at read time.
        """

        if not isinstance(operation, Operation):
            raise TypeError(f"operation must be an Operation, got {type(operation).__name__}")
        if state is not None and not isinstance(state, DomainState):
            raise TypeError(f"state must be a DomainState, got {type(state).__name__}")
        count = self._next_occurrence_count(type(operation))
        started = self._occurrence(event=Started(operation), count=count)
        self._dispatch_occurrence(started, state=state)
        try:
            yield started
        except BaseException:
            self._dispatch_occurrence(
                self._occurrence(event=Ended(operation, succeeded=False), count=count), state=state
            )
            raise
        else:
            self._dispatch_occurrence(
                self._occurrence(event=Ended(operation, succeeded=True), count=count), state=state
            )

    def _next_occurrence_count(self, event_type: type[TypedEvent] | type[Operation]) -> int:
        count = self._occurrence_counts.get(event_type, 0) + 1
        self._occurrence_counts[event_type] = count
        return count
    def _occurrence(self, *, event: _EventT, count: int) -> Occurrence[_EventT]:
        """Create one delivery-stamped occurrence before any observer runs."""

        return Occurrence(event=event, count=count, monotonic_time=self.monotonic_clock())


    def _dispatch_occurrence(
        self, occurrence: Occurrence[Any], *, state: DomainState | None = None
    ) -> None:
        """Record and dispatch one typed occurrence in callback order.

        ``state`` never reaches the durable occurrence record: that edge says
        only when something happened, and domain data travels beside it, to
        handlers that declared the domain.
        """

        write_occurrence_artifact(self, occurrence)
        write_typed_event_artifact(self, occurrence)
        if not self.callbacks:
            return
        # Deferred because callback base imports RunContext from this module:
        # `tpen.callback.base` imports `RunContext` from this module, so a
        # module-level import here would be circular.
        from tpen.callback.base import StatefulCallback

        for callback in self.callbacks:
            if isinstance(callback, StatefulCallback):
                # The ``isinstance(state, callback.state_type)`` filter used to
                # live here, and moving it inside is the whole point of the
                # ADR-E008 amendment. A callback may now declare one subscription
                # group that wants its domain's state and another that wants a
                # state-free boundary such as the run lifecycle, and the
                # dispatcher cannot tell which of them an occurrence will match.
                # Only the callback can, because only it holds the groups.
                #
                # A callback observing a different domain is still skipped rather
                # than failed -- one run emits several domains' occurrences, and
                # only some carry state -- and because both boundaries of a scope
                # share one state, a skipped `Started` is still always matched by
                # a skipped `Ended`, so no lifecycle pair is left half-open.
                callback.handle_occurrence(occurrence, self, state)
            else:
                callback.handle_occurrence(occurrence, self)


def generate_run_id(run_name: str, *, clock: RunClock | None = None) -> str:
    """Return a timestamped run identifier."""

    run_clock = _default_run_clock() if clock is None else clock
    timestamp = run_clock.now().strftime("%Y-%m-%d_%H%M%S")
    slug = _slugify(run_name)
    return f"{timestamp}_{slug}_{uuid4().hex[:6]}"


def build_run_metadata(
    cfg: DictConfig,
    *,
    command: str | None,
    config_path: str | None,
    clock: RunClock | None = None,
) -> RunMetadata:
    """Build metadata for a resolved run config."""

    run_clock = resolve_run_clock(cfg) if clock is None else clock
    git = collect_git_metadata()
    return RunMetadata(
        run_id=str(OmegaConf.select(cfg, "run.run_id")),
        run_name=str(OmegaConf.select(cfg, "experiment.run_name", default=OmegaConf.select(cfg, "experiment.name"))),
        timestamp=run_clock.now_iso(),
        timezone=run_clock.timezone,
        git_commit=str(git["git_commit"]),
        git_branch=str(git["git_branch"]),
        dirty_worktree=bool(git["dirty_worktree"]),
        command=command,
        config_path=config_path,
        resolved_config_path=str(Path(str(OmegaConf.select(cfg, "run.dir"))) / "resolved_config.yaml"),
        run_dir=str(OmegaConf.select(cfg, "run.dir")),
        device=str(OmegaConf.select(cfg, "runtime.device", default="cpu")),
        dtype=str(OmegaConf.select(cfg, "runtime.dtype", default="float64")),
        extra=collect_hardware_metadata(
            device=str(OmegaConf.select(cfg, "runtime.device", default="cpu")),
            dtype=str(OmegaConf.select(cfg, "runtime.dtype", default="float64")),
        ),
    )


def resolve_run_clock(cfg: DictConfig) -> RunClock:
    """Resolve and validate the run-wide timezone from config.

    The value lives at ``run.timezone`` and defaults to ``UTC``. It must be an
    IANA timezone accepted by :mod:`zoneinfo`, such as ``UTC`` or
    ``America/New_York``.
    """

    value = OmegaConf.select(cfg, "run.timezone", default=DEFAULT_RUN_TIMEZONE)
    if value is None:
        value = DEFAULT_RUN_TIMEZONE
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run.timezone must be a nonempty IANA timezone name")
    timezone = value.strip()
    if timezone == DEFAULT_RUN_TIMEZONE:
        return RunClock(timezone=DEFAULT_RUN_TIMEZONE, tzinfo=UTC)
    try:
        return RunClock(timezone=timezone, tzinfo=ZoneInfo(timezone))
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"run.timezone must be a valid IANA timezone name, got {timezone!r}") from exc


def collect_git_metadata() -> dict[str, Any]:
    """Collect git commit, branch, and dirty-state metadata."""

    status = _run_git(["git", "status", "--short", "--untracked-files=all"])
    return {
        "git_commit": _run_git(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_git(["git", "branch", "--show-current"]),
        "dirty_worktree": bool(status.strip()),
    }


def collect_hardware_metadata(*, device: str, dtype: str) -> dict[str, Any]:
    """Collect hardware, runtime, and scheduler provenance once per run.

    The returned container is JSON-safe and intentionally uses only stdlib plus
    an optional lazy torch import. This keeps hardware provenance in run setup,
    not in trainers, models, samplers, diagnostics, or loggers.

    Parameters
    ----------
    device : str
        Configured runtime device.
    dtype : str
        Configured runtime floating dtype.

    Returns
    -------
    dict
        Nested ``hardware``, ``runtime``, and ``slurm`` metadata blocks.
    """

    torch_info = _collect_torch_hardware()
    hardware = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_available": _available_cpu_count(),
        "cpu_count_physical": None,
        "cuda_available": torch_info["cuda_available"],
        "cuda_device_count": torch_info["cuda_device_count"],
        "cuda_devices": torch_info["cuda_devices"],
    }
    runtime = {
        "device": device,
        "dtype": dtype,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch_version": torch_info["torch_version"],
        "torch_cuda_version": torch_info["torch_cuda_version"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return {
        "hardware": hardware,
        "runtime": runtime,
        "slurm": _collect_slurm_metadata(),
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def append_jsonl(path: Path, data: Mapping[str, Any]) -> None:
    """Append a JSON-safe object as one JSONL record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(data), sort_keys=True, allow_nan=False))
        handle.write("\n")


def write_occurrence_artifact(context: RunContext, occurrence: Occurrence[Any]) -> None:
    """Append one typed occurrence to the separate typed JSONL edge."""

    event = occurrence.event
    subject: object = event
    record: dict[str, Any] = {
        "count": occurrence.count,
        "event": _qualified_type_name(event),
        "run_id": context.metadata.run_id,
        "time": context.now_iso(),
    }
    if isinstance(event, (Started, Ended)):
        subject = event.operation
        record["operation"] = _qualified_type_name(subject)
    # Mappings exist only in this serialization adapter; core event values
    # remain typed objects.
    record["fields"] = _typed_event_fields(subject)
    append_jsonl(context.path("occurrences.jsonl"), record)


def write_typed_event_artifact(context: RunContext, occurrence: Occurrence[Any]) -> None:
    """Project typed occurrences onto the stable human-facing event stream.

    ``events.jsonl`` is an artifact schema, not a callback transport.  Its
    names remain stable for existing run tooling while routing and callback
    delivery use only typed occurrences.
    """

    from tpen.checkpoint.events import CheckpointRestored, LoadFailed, LoadStarted, LoadSucceeded
    from tpen.run_events import RunCompleted, RunFailed, RunStarted
    from tpen.training.events import (
        ModelBuilt,
        TrainingCompleted,
        TrainingIteration,
        TrainingStarted,
    )
    from tpen.events import Ended, Started

    event = occurrence.event
    name: str | None = None
    payload: dict[str, Any] = {}
    step: int | None = None
    if isinstance(event, RunStarted):
        name = "run_start"
    elif isinstance(event, RunCompleted):
        name = "run_end"
        payload = {"status": event.status}
    elif isinstance(event, RunFailed):
        name = "exception"
        payload = {"exception_type": event.exception_type, "exception_message": event.exception_message}
    elif isinstance(event, LoadStarted):
        name = "load_start"
        payload = {"path": event.path, "mode": event.mode, "strict": event.strict}
    elif isinstance(event, LoadFailed):
        name = "load_failed"
        payload = {"path": event.path, "mode": event.mode, "exception_type": event.exception_type, "message": event.message}
    elif isinstance(event, LoadSucceeded):
        name = "load_success"
        payload = {"path": event.path, **event.report.to_dict()}
    elif isinstance(event, CheckpointRestored):
        name = "checkpoint_restored"
        payload = {"restore_report": event.report.to_dict()}
    elif isinstance(event, ModelBuilt):
        name = "model_built"
    elif isinstance(event, TrainingStarted):
        name = "train_start"
    elif isinstance(event, TrainingCompleted):
        name = "train_end"
    elif isinstance(event, Started) and isinstance(event.operation, TrainingIteration):
        name = "step_start"
        step = event.operation.step
    elif isinstance(event, Ended) and isinstance(event.operation, TrainingIteration):
        name = "step_end"
        step = event.operation.step
    if name is None:
        return
    if step is not None:
        payload["step"] = step
    append_jsonl(
        context.path("events.jsonl"),
        {
            "event": name,
            "payload": _event_jsonable(payload),
            "run_id": context.metadata.run_id,
            "step": step,
            "time": context.now_iso(),
        },
    )


def write_error_artifact(
    target: RunContext | Path,
    exception: BaseException,
    *,
    phase: str | None = None,
    traceback_text: str | None = None,
    command: str | None = None,
    config_path: str | None = None,
) -> Path:
    """Write a durable failure artifact when a run directory is available."""

    if isinstance(target, RunContext):
        run_dir = target.run_dir
        now = target.now_iso()
        run_id = target.metadata.run_id
        command = target.metadata.command if command is None else command
        config_path = target.metadata.config_path if config_path is None else config_path
    else:
        run_dir = Path(target)
        now = datetime.now(UTC).isoformat()
        run_id = None
    path = run_dir / "error.json"
    write_json(
        path,
        {
            "command": command,
            "config_path": config_path,
            "exception_message": str(exception),
            "exception_type": type(exception).__name__,
            "phase": phase,
            "run_dir": str(run_dir),
            "run_id": run_id,
            "status": "failed",
            "time": now,
            "traceback": traceback_text,
        },
    )
    return path


def write_run_start_artifact(context: RunContext) -> None:
    """Write the early run-start breadcrumb before long-running work begins."""

    cfg = context.cfg
    study = OmegaConf.select(cfg, "study", default={}) or {}
    if OmegaConf.is_config(study):
        study = OmegaConf.to_container(study, resolve=True)
    if not isinstance(study, Mapping):
        study = {}
    data = {
        "run_id": context.metadata.run_id,
        "run_dir": context.metadata.run_dir,
        "study": {
            "name": study.get("name"),
            "config_id": study.get("config_id"),
        },
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "command": context.metadata.command,
        "git": {
            "sha": context.metadata.git_commit,
            "branch": context.metadata.git_branch,
            "dirty": context.metadata.dirty_worktree,
        },
        "slurm": _collect_slurm_metadata(),
        "environment": _collect_allowed_environment(),
        "start_time_unix": context.now().timestamp(),
    }
    write_json(context.path("run_start.json"), data)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"exception_type": type(value).__name__, "message": str(value)}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _event_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"exception_type": type(value).__name__, "message": str(value)}
    if isinstance(value, Mapping):
        return {str(key): _event_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_jsonable(item) for item in value]
    if isinstance(value, DictConfig):
        return _event_jsonable(OmegaConf.to_container(value, resolve=True))
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None and dtype is not None:
        return {
            "device": None if device is None else str(device),
            "dtype": str(dtype),
            "shape": [int(dim) for dim in shape],
            "type": f"{type(value).__module__}.{type(value).__name__}",
        }
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def _typed_event_fields(value: object) -> dict[str, Any]:
    """Encode public dataclass fields at the event artifact boundary."""

    if is_dataclass(value):
        return _typed_dataclass_fields(value)
    if _has_instance_state(value):
        raise TypeError(
            f"stateful typed value {_qualified_type_name(value)} must be a dataclass to serialize"
        )
    return {}


def _typed_dataclass_fields(value: object, ancestors: tuple[int, ...] = ()) -> dict[str, Any]:
    """Encode explicit public fields of one typed dataclass value.

    Parameters
    ----------
    value : object
        Dataclass instance whose declared public fields are encoded.
    ancestors : tuple of int, optional
        Identities of the dataclass instances already open on the current
        field path, used to refuse a cycle instead of exhausting the stack.
    """

    nested = (*ancestors, id(value))
    return {
        item.name: _typed_event_field_value(getattr(value, item.name), nested)
        for item in dataclass_fields(value)
        if not item.name.startswith("_")
    }


def _typed_event_field_value(value: object, ancestors: tuple[int, ...] = ()) -> Any:
    """Preserve nested typed dataclass fields without probing containers.

    Any dataclass instance is encoded field-wise, not only an ``Event`` or an
    ``Operation``. A dataclass declares its fields explicitly, so reading
    ``dataclass_fields`` is typed access to a stated contract rather than a
    probe of an arbitrary container. Without this, a domain object carried on a
    typed event -- a ``RestoreReport``, an ``EvaluationFailure`` -- would reach
    ``occurrences.jsonl`` as a bare ``{"type": ...}`` marker with every field
    silently dropped.

    A dataclass reached through a list, a tuple, or a mapping value still
    collapses to its type marker, exactly as before. No typed event carries a
    container of dataclasses, so widening the traversal would be speculative
    (ADR-E003); ``test_typed_event_does_not_recurse_into_containers`` pins that
    boundary so a later change to it is deliberate.
    """

    # A dataclass *type* is not an instance and keeps its previous encoding.
    if is_dataclass(value) and not isinstance(value, type):
        if id(value) in ancestors:
            raise TypeError(
                f"cyclic typed value {_qualified_type_name(value)} cannot be serialized"
            )
        return _typed_dataclass_fields(value, ancestors)
    return _event_jsonable(value)


def _has_instance_state(value: object) -> bool:
    """Return whether a non-dataclass value carries instance-owned state."""

    instance_dict = getattr(value, "__dict__", None)
    if instance_dict:
        return True
    for value_type in type(value).__mro__:
        slots = value_type.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"}:
                continue
            if hasattr(value, slot):
                return True
    return False


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "run"


def _default_run_clock() -> RunClock:
    return RunClock(timezone=DEFAULT_RUN_TIMEZONE, tzinfo=UTC)


def _run_git(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _collect_torch_hardware() -> dict[str, Any]:
    try:
        torch = import_module("torch")
    except ImportError:
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
        }

    cuda = getattr(torch, "cuda", None)
    cuda_available = False
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available):
        try:
            cuda_available = bool(is_available())
        except Exception:  # pragma: no cover - hardware/runtime dependent
            cuda_available = False
    device_count = 0
    device_count_fn = getattr(cuda, "device_count", None)
    if cuda_available and callable(device_count_fn):
        try:
            device_count = int(device_count_fn())
        except Exception:  # pragma: no cover - hardware/runtime dependent
            device_count = 0
    devices = []
    get_device_properties = getattr(cuda, "get_device_properties", None)
    for index in range(device_count):
        if not callable(get_device_properties):
            devices.append({"index": index, "error": "torch.cuda.get_device_properties unavailable"})
            continue
        try:
            properties = get_device_properties(index)
        except Exception as exc:  # pragma: no cover - hardware dependent
            devices.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "capability": f"{int(properties.major)}.{int(properties.minor)}",
            }
        )
    return {
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
    }


def _available_cpu_count() -> int | None:
    affinity = getattr(os, "sched_getaffinity", None)
    if not callable(affinity):
        return None
    try:
        return len(affinity(0))
    except OSError:
        return None


def _collect_slurm_metadata() -> dict[str, str]:
    keys = {
        "job_id": "SLURM_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
        "cpus_per_task": "SLURM_CPUS_PER_TASK",
        "mem_per_node": "SLURM_MEM_PER_NODE",
        "job_partition": "SLURM_JOB_PARTITION",
        "submit_dir": "SLURM_SUBMIT_DIR",
        "job_name": "SLURM_JOB_NAME",
    }
    return {name: os.environ[env] for name, env in keys.items() if env in os.environ}


def _collect_allowed_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in RUN_START_ENV_ALLOWLIST if key in os.environ}


__all__ = [
    "ArtifactManager",
    "DEFAULT_RUN_TIMEZONE",
    "REQUIRED_RUN_DIRS",
    "RunClock",
    "RunContext",
    "RunMetadata",
    "RunResult",
    "build_run_metadata",
    "collect_hardware_metadata",
    "collect_git_metadata",
    "generate_run_id",
    "resolve_run_clock",
    "write_json",
    "write_error_artifact",
    "write_occurrence_artifact",
    "write_typed_event_artifact",
    "write_run_start_artifact",
]
