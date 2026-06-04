"""Optional experiment tracking integrations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class NullTracker:
    """No-op run tracker.

    This class preserves a common tracking interface when external tracking is
    disabled by config.
    """

    def log_rows(self, rows: list[dict[str, object]], *, step_key: str = "step") -> None:
        """Ignore a sequence of metric rows.

        Parameters
        ----------
        rows : list of dict
            Metric rows to ignore.
        step_key : str, optional
            Column that would define the metric step.
        """

        _ = rows, step_key

    def log_metrics(self, metrics: dict[str, object]) -> None:
        """Ignore final scalar metrics.

        Parameters
        ----------
        metrics : dict
            Final metrics to ignore.
        """

        _ = metrics

    def finish(self) -> None:
        """Finish without side effects."""

        return None


class WandbTracker:
    """Weights & Biases tracker wrapper.

    Parameters
    ----------
    run : object
        Object returned by ``wandb.init``.
    """

    def __init__(self, run: Any) -> None:
        self.run = run

    def log_rows(self, rows: list[dict[str, object]], *, step_key: str = "step") -> None:
        """Log row-wise metrics.

        Parameters
        ----------
        rows : list of dict
            Metric rows to log.
        step_key : str, optional
            Key used as the W&B step when present and integral.
        """

        for row in rows:
            payload = _numeric_payload(row, exclude={step_key})
            if not payload:
                continue
            step = _wandb_step(row.get(step_key, None))
            if step is None:
                self.run.log(payload)
            else:
                self.run.log(payload, step=step)

    def log_metrics(self, metrics: dict[str, object]) -> None:
        """Log final metrics and update the run summary.

        Parameters
        ----------
        metrics : dict
            Final scalar metrics.
        """

        payload = _numeric_payload(metrics)
        if not payload:
            return
        self.run.log(payload)
        self.run.summary.update(payload)

    def finish(self) -> None:
        """Finish the W&B run."""

        self.run.finish()


def build_tracker(cfg: DictConfig, *, output_dir: Path, git: dict[str, str]) -> NullTracker | WandbTracker:
    """Build the configured run tracker.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Resolved run configuration.
    output_dir : pathlib.Path
        Run artifact directory used as the W&B local directory.
    git : dict of str to str
        Git provenance metadata to include in the logged config.

    Returns
    -------
    NullTracker or WandbTracker
        No-op tracker when ``tracking.wandb.enabled`` is false; W&B tracker
        otherwise.

    Raises
    ------
    ModuleNotFoundError
        If W&B tracking is enabled but the optional ``wandb`` dependency is not
        installed.
    """

    if not bool(OmegaConf.select(cfg, "tracking.wandb.enabled", default=False)):
        return NullTracker()
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tracking.wandb.enabled=true requires the optional W&B dependency; "
            "install with `uv sync --extra wandb` or disable tracking.wandb.enabled."
        ) from exc

    wandb_cfg = OmegaConf.select(cfg, "tracking.wandb", default={})
    tags = list(OmegaConf.select(cfg, "tracking.tags", default=[]))
    run = wandb.init(
        project=_select(wandb_cfg, "project", str(OmegaConf.select(cfg, "experiment_name", default="spenn"))),
        entity=_select(wandb_cfg, "entity", None),
        name=_select(wandb_cfg, "name", str(OmegaConf.select(cfg, "run_id", default=None))),
        id=_select(wandb_cfg, "id", str(OmegaConf.select(cfg, "run_id", default=None))),
        group=_select(wandb_cfg, "group", str(OmegaConf.select(cfg, "experiment_name", default="spenn"))),
        job_type=_select(wandb_cfg, "job_type", str(OmegaConf.select(cfg, "run.mode", default="train"))),
        mode=_select(wandb_cfg, "mode", None),
        dir=str(output_dir),
        tags=tags,
        config={"config": OmegaConf.to_container(cfg, resolve=True), "git": dict(git)},
        resume=_select(wandb_cfg, "resume", "allow"),
    )
    return WandbTracker(run)


def _select(cfg: object, key: str, default: object) -> object:
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, key, default=default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _numeric_payload(row: dict[str, object], *, exclude: set[str] | None = None) -> dict[str, float | int]:
    excluded = exclude or set()
    payload: dict[str, float | int] = {}
    for key, value in row.items():
        if key in excluded or isinstance(value, bool):
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, int):
            payload[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            payload[key] = value
    return payload


def _wandb_step(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
