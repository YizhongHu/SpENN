"""Generic configured run launcher."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from spenn.artifacts import (
    ArtifactManager,
    RunContext,
    RunResult,
    build_run_metadata,
    generate_run_id,
)
from spenn.callback import Event
from spenn.runner import Runner


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured SpENN runner from the command line."""

    args = parse_args(argv)
    command = " ".join(["run.py", *(sys.argv[1:] if argv is None else argv)])
    cfg = load_config(str(args.config), args.overrides)
    return run_from_config(cfg, config_path=str(args.config), command=command)


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
) -> RunContext:
    """Resolve run metadata, artifact paths, callbacks, and loggers.

    Callbacks and loggers are configured at the config root and owned by the
    `RunContext`; runners dispatch into ``context.callbacks`` and log through
    ``context.log``.
    """

    source_cfg = _rerunnable_config(cfg)
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    run_name = str(
        OmegaConf.select(
            resolved_cfg,
            "experiment.run_name",
            default=OmegaConf.select(resolved_cfg, "experiment.name", default="spenn_run"),
        )
    )
    run_id = OmegaConf.select(resolved_cfg, "run.run_id", default=None)
    if run_id is None:
        run_id = generate_run_id(run_name)
        OmegaConf.update(resolved_cfg, "run.run_id", run_id, merge=False, force_add=True)
    experiment_name = str(OmegaConf.select(resolved_cfg, "experiment.name", default="experiment"))
    sector = str(OmegaConf.select(resolved_cfg, "experiment.sector", default="default"))
    root = Path(str(OmegaConf.select(resolved_cfg, "run.root", default="outputs")))
    artifact_manager = ArtifactManager(root, experiment_name, sector, str(run_id))
    OmegaConf.update(resolved_cfg, "run.dir", str(artifact_manager.run_dir), merge=False, force_add=True)
    OmegaConf.resolve(resolved_cfg)
    artifact_manager.make_dirs()

    loggers = _instantiate_sequence(OmegaConf.select(resolved_cfg, "loggers", default=[]))
    callbacks = _instantiate_sequence(OmegaConf.select(resolved_cfg, "callbacks", default=[]))
    # Fail-loud interface validation only: confirm the configured objects expose
    # the lifecycle methods, without invoking any behavior (no handle/log/finish).
    _validate_callbacks(callbacks)
    _validate_loggers(loggers)
    metadata = build_run_metadata(resolved_cfg, command=command, config_path=config_path)
    return RunContext(
        cfg=resolved_cfg,
        source_cfg=source_cfg,
        artifact_manager=artifact_manager,
        metadata=metadata,
        callbacks=callbacks,
        loggers=loggers,
    )


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

    context = prepare_run_context(cfg, config_path=config_path, command=command)
    runner: Runner | None = None
    try:
        runner = _instantiate_runner(context)
        result = runner.run(context)
        if isinstance(result, RunResult):
            context.metadata.status = result.status
        return 0
    except Exception as exc:
        context.metadata.status = "failed"
        payload = {
            "exception": exc,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if runner is None:
            event = Event(name="exception", context=context, payload=payload)
            for callback in context.callbacks:
                callback.handle(event)
        else:
            runner.emit("exception", context, payload=payload)
        if raise_exceptions:
            raise
        return 1
    finally:
        for logger in context.loggers:
            logger.finish()


def _validate_callbacks(callbacks: list[object]) -> None:
    """Fail loudly if a configured callback lacks a callable ``handle(event)``.

    This checks the interface shape only; callback behavior is invoked lazily
    through normal lifecycle events, never during setup.
    """

    for index, callback in enumerate(callbacks):
        if not callable(getattr(callback, "handle", None)):
            raise TypeError(
                f"callbacks[{index}]={type(callback).__name__} must expose callable handle(event)"
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


def _instantiate_runner(context: RunContext) -> Runner:
    runner_cfg = context.cfg.runner.copy()
    # Callbacks and loggers are configured at the config root and owned by the
    # RunContext; a runner must not own them.
    for forbidden in ("callbacks", "loggers"):
        if forbidden in runner_cfg:
            raise ValueError(
                f"runner config must not own {forbidden!r}; configure it at the config root."
            )
    if "diagnostics" in runner_cfg:
        diagnostics = _instantiate_sequence(OmegaConf.select(runner_cfg, "diagnostics", default=[]))
        del runner_cfg["diagnostics"]
        runner = instantiate(runner_cfg, _partial_=True)(diagnostics=diagnostics)
    else:
        runner = instantiate(runner_cfg)
    if not isinstance(runner, Runner):
        raise TypeError(f"runner must instantiate to spenn.runner.Runner, got {type(runner)!r}")
    return runner


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
