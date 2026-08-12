"""Generic configured run launcher."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hydra.utils import instantiate
from hydra.errors import InstantiationException
from omegaconf import DictConfig, ListConfig, OmegaConf

from tpen.accelerator import seed_all as accelerator_seed_all
from tpen.artifacts import (
    ArtifactManager,
    RunContext,
    RunResult,
    build_run_metadata,
    generate_run_id,
    resolve_run_clock,
    write_error_artifact,
    write_run_start_artifact,
)
from tpen.callback import configure_terminal_logging
from tpen.config import register_resolvers
from tpen.dependencies import OptionalDependencyError, require_torch
from tpen.events import Event as TypedEvent
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.runner import Runner

# Register custom OmegaConf resolvers (e.g. spenn.basis_feature_dim) before any
# config is loaded or resolved on the run path.
register_resolvers()


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured TPEN runner from the command line."""

    _install_bootstrap_stderr_logger()
    args = parse_args(argv)
    command = " ".join(["run.py", *(sys.argv[1:] if argv is None else argv)])
    try:
        cfg = load_config(str(args.config), args.overrides)
    except Exception as exc:
        _print_fatal(
            exc,
            phase="bootstrap",
            traceback_text=traceback.format_exc(),
            command=command,
            config_path=str(args.config),
        )
        return 1
    try:
        _preflight_optional_dependencies(cfg)
    except OptionalDependencyError as exc:
        _print_fatal(exc, phase="bootstrap", command=command, config_path=str(args.config))
        return 1
    return run_from_config(cfg, config_path=str(args.config), command=command)


@dataclass
class _BootstrapState:
    run_dir: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse configured-run command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args(argv)


def load_config(config_path: str, overrides: Sequence[str] | None = None) -> DictConfig:
    """Load a YAML config and apply OmegaConf dotlist overrides."""

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def prepare_run_context(
    cfg: DictConfig,
    *,
    config_path: str | None = None,
    command: str | None = None,
    bootstrap: _BootstrapState | None = None,
) -> RunContext:
    """Resolve run metadata, artifact paths, callbacks, and loggers.

    Callbacks and loggers are configured at the config root and owned by the
    `RunContext`; runners dispatch into ``context.callbacks`` and log through
    ``context.log``.
    """

    run_clock = resolve_run_clock(cfg)
    source_cfg = _rerunnable_config(cfg)
    OmegaConf.update(source_cfg, "run.timezone", run_clock.timezone, merge=False, force_add=True)
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(resolved_cfg, "run.timezone", run_clock.timezone, merge=False, force_add=True)
    run_name = str(
        OmegaConf.select(
            resolved_cfg,
            "experiment.run_name",
            default=OmegaConf.select(resolved_cfg, "experiment.name", default="spenn_run"),
        )
    )
    run_id = OmegaConf.select(resolved_cfg, "run.run_id", default=None)
    if run_id is None:
        run_id = generate_run_id(run_name, clock=run_clock)
        OmegaConf.update(resolved_cfg, "run.run_id", run_id, merge=False, force_add=True)
    experiment_name = str(OmegaConf.select(resolved_cfg, "experiment.name", default="experiment"))
    sector = str(OmegaConf.select(resolved_cfg, "experiment.sector", default="default"))
    root = Path(str(OmegaConf.select(resolved_cfg, "run.root", default="outputs")))
    layout = str(OmegaConf.select(resolved_cfg, "run.layout", default="nested"))
    artifact_manager = ArtifactManager(root, experiment_name, sector, str(run_id), layout=layout)
    if bootstrap is not None:
        bootstrap.run_dir = artifact_manager.run_dir
    OmegaConf.update(resolved_cfg, "run.dir", str(artifact_manager.run_dir), merge=False, force_add=True)
    OmegaConf.resolve(resolved_cfg)
    artifact_manager.make_dirs()
    _configure_terminal_logging(resolved_cfg)

    loggers = _instantiate_sequence(OmegaConf.select(resolved_cfg, "loggers", default=[]))
    callbacks = _instantiate_sequence(OmegaConf.select(resolved_cfg, "callbacks", default=[]))
    # Fail-loud interface validation only: confirm the configured objects expose
    # the lifecycle methods, without invoking any behavior (no handle/log/finish).
    _validate_callbacks(callbacks)
    _validate_loggers(loggers)
    metadata = build_run_metadata(resolved_cfg, command=command, config_path=config_path, clock=run_clock)
    context = RunContext(
        cfg=resolved_cfg,
        source_cfg=source_cfg,
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=run_clock,
        callbacks=callbacks,
        loggers=loggers,
    )
    write_run_start_artifact(context)
    return context


def run_from_config(
    cfg: DictConfig,
    *,
    config_path: str | None = None,
    command: str | None = None,
    raise_exceptions: bool = False,
) -> int:
    """Instantiate and execute the configured runner.

    Parameters
    ----------
    cfg : DictConfig
        Resolved run configuration.
    config_path, command : str or None, optional
        Provenance recorded in run metadata.
    raise_exceptions : bool, optional
        When ``True``, re-raise the original exception after the status update,
        exception event, and logger teardown. The default ``False`` preserves
        CLI-style ``return 1`` behavior; tests and debugging can set ``True`` to
        surface the original traceback.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a handled failure (when
        ``raise_exceptions=False``).
    """

    _install_bootstrap_stderr_logger()
    bootstrap = _BootstrapState()
    context: RunContext | None = None
    runner: Runner | None = None
    try:
        context = prepare_run_context(cfg, config_path=config_path, command=command, bootstrap=bootstrap)
        _seed_runtime_rngs(context.cfg)
        # Typed first, then the legacy string, so that within one moment the
        # migrated subscribers keep the order they had when both arrived on the
        # same event: `RunTiming` started its clock before `Status` rendered its
        # start boxes, and reversing the two emits would fold that rendering
        # into the measured wall time.
        context.emit(RunStarted())
        context.emit_event("run_start")
        runner = _instantiate_runner(context)
        result = runner.run(context)
        # The harness owns the whole run lifecycle, including this boundary,
        # which the runners emitted the ``run_end`` string for. See
        # `tpen.run_events` for why one emitter rather than three. Emitted BEFORE
        # the status is copied off the result, matching the legacy ordering:
        # `Metadata` sets ``metadata.status`` itself, and a run whose evaluation
        # suite failed would otherwise have its ``failed`` status overwritten.
        context.emit(RunCompleted())
        if isinstance(result, RunResult):
            context.metadata.status = result.status
        return 0
    except Exception as exc:
        phase = _failure_phase(exc, context=context, runner=runner)
        traceback_text = traceback.format_exc()
        payload = {
            "exception": exc,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "phase": phase,
            "traceback": traceback_text,
        }
        if context is not None:
            context.metadata.status = "failed"
            _write_error_if_possible(context, exc, phase=phase, traceback_text=traceback_text)
            # ONE typed event for the two legacy strings below, which carry the
            # same payload and are distinguished by no consumer -- see `RunFailed`.
            # Guarded exactly as they are: this runs while the context may be
            # half-constructed, and an emit that raised here would mask the
            # original exception.
            _emit_typed_event_if_possible(
                context,
                RunFailed(
                    exception_type=str(payload["exception_type"]),
                    exception_message=str(payload["exception_message"]),
                ),
            )
            _emit_event_if_possible(context, "run_failed", payload=payload)
            _emit_event_if_possible(context, "exception", payload=payload)
        elif bootstrap.run_dir is not None:
            _write_error_if_possible(
                bootstrap.run_dir,
                exc,
                phase=phase,
                traceback_text=traceback_text,
                command=command,
                config_path=config_path,
            )
        _print_fatal(
            exc,
            phase=phase,
            traceback_text=traceback_text,
            run_dir=context.run_dir if context is not None else bootstrap.run_dir,
            command=command,
            config_path=config_path,
        )
        if raise_exceptions:
            raise
        return 1
    finally:
        if context is not None:
            for logger in context.loggers:
                logger.finish()


def _validate_callbacks(callbacks: list[object]) -> None:
    """Fail loudly if a configured callback lacks either dispatch method.

    This checks the interface shape only; callback behavior is invoked lazily
    through normal lifecycle events, never during setup.
    """

    for index, callback in enumerate(callbacks):
        if not callable(getattr(callback, "handle", None)):
            raise TypeError(
                f"callbacks[{index}]={type(callback).__name__} must expose callable handle(event)"
            )
        if not callable(getattr(callback, "handle_occurrence", None)):
            raise TypeError(
                f"callbacks[{index}]={type(callback).__name__} must expose callable "
                "handle_occurrence(occurrence, context)"
            )


def _validate_loggers(loggers: list[object]) -> None:
    """Fail loudly if a configured logger lacks callable ``log``/``finish``.

    This checks the interface shape only; logger behavior is invoked lazily when
    records are logged and during normal run teardown, never during setup.
    """

    for index, logger in enumerate(loggers):
        if not callable(getattr(logger, "log", None)):
            raise TypeError(
                f"loggers[{index}]={type(logger).__name__} must expose callable log(record)"
            )
        if not callable(getattr(logger, "finish", None)):
            raise TypeError(
                f"loggers[{index}]={type(logger).__name__} must expose callable finish()"
            )


def _instantiate_sequence(items: ListConfig | list | tuple | None) -> list:
    instantiated = []
    for item in items or []:
        if isinstance(item, DictConfig) and "_target_" in item:
            instantiated.append(instantiate(item))
        else:
            instantiated.append(item)
    return instantiated


def _seed_runtime_rngs(cfg: DictConfig) -> None:
    """Seed process RNGs from ``runtime.seed`` before runner construction."""

    seed = OmegaConf.select(cfg, "runtime.seed", default=None)
    if seed is None:
        return
    seed_int = int(seed)
    random.seed(seed_int)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed_int % (2**32 - 1))
    if _config_requires_torch(OmegaConf.to_container(cfg, resolve=False)):
        torch = require_torch(feature="seeded configured TPEN run")
        torch.manual_seed(seed_int)
        accelerator_seed_all(seed_int, feature="seeded configured TPEN run")


def _instantiate_runner(context: RunContext) -> Runner:
    runner_cfg = context.cfg.runner.copy()
    # Callbacks and loggers are configured at the config root and owned by the
    # RunContext; a runner must not own them.
    for forbidden in ("callbacks", "loggers"):
        if forbidden in runner_cfg:
            raise ValueError(
                f"runner config must not own {forbidden!r}; configure it at the config root."
            )
    try:
        runner = instantiate(runner_cfg)
    except InstantiationException as exc:
        # Hydra wraps constructor/configuration failures, but the run artifact
        # contract records the underlying failure identity. Preserve the
        # original exception when Hydra attached it as the cause.
        if exc.__cause__ is not None:
            raise exc.__cause__ from exc
        raise
    if not isinstance(runner, Runner):
        raise TypeError(f"runner must instantiate to tpen.runner.Runner, got {type(runner)!r}")
    return runner


def _configure_terminal_logging(cfg: DictConfig) -> None:
    terminal = OmegaConf.select(cfg, "terminal", default=None)
    if terminal is None:
        return
    configure_terminal_logging(
        enabled=bool(OmegaConf.select(terminal, "enabled", default=True)),
        level=str(OmegaConf.select(terminal, "level", default="info")),
        color=str(OmegaConf.select(terminal, "color", default="auto")),
    )


def _install_bootstrap_stderr_logger() -> None:
    """Install a minimal stderr logger for fatal bootstrap diagnostics."""

    logger = logging.getLogger("spenn.bootstrap")
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        if getattr(handler, "_spenn_bootstrap_handler", False):
            return
    handler = logging.StreamHandler(sys.stderr)
    handler._spenn_bootstrap_handler = True
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def _failure_phase(
    exc: BaseException,
    *,
    context: RunContext | None,
    runner: Runner | None,
) -> str:
    phase = getattr(exc, "_spenn_failure_phase", None)
    if phase is not None:
        return str(phase)
    if context is None or runner is None:
        return "bootstrap"
    return "run"


def _write_error_if_possible(
    target: RunContext | Path,
    exc: BaseException,
    *,
    phase: str,
    traceback_text: str,
    command: str | None = None,
    config_path: str | None = None,
) -> None:
    try:
        write_error_artifact(
            target,
            exc,
            phase=phase,
            traceback_text=traceback_text,
            command=command,
            config_path=config_path,
        )
    except Exception as artifact_exc:  # pragma: no cover - disk/runtime dependent
        logging.getLogger("spenn.bootstrap").error(
            "FATAL: failed to write error.json: %s: %s",
            type(artifact_exc).__name__,
            artifact_exc,
        )


def _emit_event_if_possible(context: RunContext, name: str, *, payload: dict[str, object]) -> None:
    try:
        context.emit_event(name, payload=payload)
    except Exception as event_exc:  # pragma: no cover - callback/runtime dependent
        logging.getLogger("spenn.bootstrap").error(
            "FATAL: failed to emit %s while reporting failure: %s: %s",
            name,
            type(event_exc).__name__,
            event_exc,
        )


def _emit_typed_event_if_possible(context: RunContext, event: TypedEvent) -> None:
    """Emit one typed event on the failure path without masking the failure.

    The typed sibling of `_emit_event_if_possible`, and it exists for the same
    reason: this runs after the run has already raised, possibly from a
    half-constructed context, so a callback or a disk error here must not replace
    the exception the user needs to see.
    """

    try:
        context.emit(event)
    except Exception as event_exc:  # pragma: no cover - callback/runtime dependent
        logging.getLogger("spenn.bootstrap").error(
            "FATAL: failed to emit %s while reporting failure: %s: %s",
            type(event).__name__,
            type(event_exc).__name__,
            event_exc,
        )


def _print_fatal(
    exc: BaseException,
    *,
    phase: str,
    traceback_text: str | None = None,
    run_dir: Path | None = None,
    command: str | None = None,
    config_path: str | None = None,
) -> None:
    """Print a fatal diagnostic to stderr regardless of terminal settings."""

    parts = [f"FATAL {phase} error: {type(exc).__name__}: {exc}"]
    load_path = getattr(exc, "_spenn_load_path", None)
    load_mode = getattr(exc, "_spenn_load_mode", None)
    if load_path is not None:
        parts.append(f"load.path: {load_path}")
    if load_mode is not None:
        parts.append(f"load.mode: {load_mode}")
    if run_dir is not None:
        parts.append(f"run_dir: {run_dir}")
    if config_path is not None:
        parts.append(f"config: {config_path}")
    if command is not None:
        parts.append(f"command: {command}")
    if traceback_text:
        parts.append(traceback_text.rstrip())
    print("\n".join(parts), file=sys.stderr, flush=True)


def _preflight_optional_dependencies(cfg: DictConfig) -> None:
    """Fail early with actionable optional-dependency errors for configured targets."""

    if _config_requires_torch(OmegaConf.to_container(cfg, resolve=False)):
        require_torch(feature="configured TPEN run")


def _config_requires_torch(value: object) -> bool:
    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str) and _target_requires_torch(target):
            return True
        return any(_config_requires_torch(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_config_requires_torch(item) for item in value)
    return False


def _target_requires_torch(target: str) -> bool:
    return target.startswith(
        (
            "torch.",
            "tpen.nn.",
            "tpen.training.",
            "tpen.sampling.",
            "tpen.physics.",
            "tpen.diagnostics.",
            "tpen.equivariance.checks.",
            "tpen.runner.Train",
            "tpen.runner.Evaluate",
        )
    )


def _rerunnable_config(cfg: DictConfig) -> DictConfig:
    snapshot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(snapshot, "run.run_id", None, merge=False, force_add=True)
    OmegaConf.update(snapshot, "run.dir", None, merge=False, force_add=True)
    return snapshot


__all__ = [
    "load_config",
    "main",
    "parse_args",
    "prepare_run_context",
    "run_from_config",
]
