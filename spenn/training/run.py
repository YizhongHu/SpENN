"""Generic configured run launcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from spenn.runner import Runner
from spenn.training.artifacts import (
    ArtifactManager,
    RunContext,
    RunResult,
    build_run_metadata,
    generate_run_id,
)
from spenn.training.callbacks import Event


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured SpENN runner from the command line."""

    args = parse_args(argv)
    command = " ".join(["train.py", *(sys.argv[1:] if argv is None else argv)])
    cfg = load_config(str(args.config), args.overrides)
    return run_from_config(cfg, config_path=str(args.config), command=command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse scaffold runner command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args(argv)


def load_config(config_path: str, overrides: Sequence[str]) -> DictConfig:
    """Load a YAML config and apply OmegaConf dotlist overrides."""

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def prepare_run_context(
    cfg: DictConfig,
    *,
    config_path: str | None,
    command: str | None,
) -> RunContext:
    """Resolve run metadata, artifact paths, callbacks, and loggers."""

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    run_name = str(OmegaConf.select(cfg, "experiment.run_name", default=OmegaConf.select(cfg, "experiment.name", default="spenn_run")))
    run_id = OmegaConf.select(cfg, "run.run_id", default=None)
    if run_id is None:
        run_id = generate_run_id(run_name)
        OmegaConf.update(cfg, "run.run_id", run_id, merge=False)
    experiment_name = str(OmegaConf.select(cfg, "experiment.name", default="experiment"))
    sector = str(OmegaConf.select(cfg, "experiment.sector", default="default"))
    root = Path(str(OmegaConf.select(cfg, "run.root", default="outputs")))
    artifact_manager = ArtifactManager(root, experiment_name, sector, str(run_id))
    OmegaConf.update(cfg, "run.dir", str(artifact_manager.run_dir), merge=False)
    OmegaConf.resolve(cfg)
    artifact_manager.make_dirs()

    loggers = _instantiate_sequence(OmegaConf.select(cfg, "loggers", default=[]))
    callbacks = _instantiate_sequence(OmegaConf.select(cfg, "callbacks", default=[]))
    metadata = build_run_metadata(cfg, command=command, config_path=config_path)
    context = RunContext(
        cfg=cfg,
        artifact_manager=artifact_manager,
        metadata=metadata,
        callbacks=callbacks,
        loggers=loggers,
    )
    return context


def run_from_config(
    cfg: DictConfig,
    *,
    config_path: str | None = None,
    command: str | None = None,
) -> int:
    """Instantiate and execute the configured runner."""

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
        if runner is None:
            event = Event(name="exception", context=context, payload={"exception": exc})
            for callback in context.callbacks:
                callback.handle(event)
        else:
            runner.emit("exception", context, payload={"exception": exc})
        return 1
    finally:
        for logger in context.loggers:
            logger.finish()


def run_config(cfg: DictConfig, *, forwarded_overrides: list[str] | None = None) -> dict[str, object]:
    """Compatibility wrapper around :func:`run_from_config`."""

    merged = cfg
    if forwarded_overrides:
        merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(forwarded_overrides))
    code = run_from_config(merged)
    return {"status": "ok" if code == 0 else "failed", "exit_code": code}


def _instantiate_sequence(items: ListConfig | list | tuple) -> list:
    instantiated = []
    for item in items:
        if isinstance(item, DictConfig) and "_target_" in item:
            instantiated.append(instantiate(item))
        else:
            instantiated.append(item)
    return instantiated


def _instantiate_runner(context: RunContext) -> Runner:
    runner_cfg = OmegaConf.create(OmegaConf.to_container(context.cfg.runner, resolve=False))
    runner_cfg.pop("callbacks", None)
    runner_cfg.pop("loggers", None)
    runner = instantiate(runner_cfg, callbacks=context.callbacks, loggers=context.loggers)
    if not isinstance(runner, Runner):
        raise TypeError(f"runner must instantiate to spenn.runner.Runner, got {type(runner)!r}")
    return runner


__all__ = [
    "load_config",
    "main",
    "parse_args",
    "prepare_run_context",
    "run_config",
    "run_from_config",
]
