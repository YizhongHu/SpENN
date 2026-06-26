"""Weights & Biases metric logger."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import LogRecord, Logger


class WandB(Logger):
    """Mirror scalar run records to Weights & Biases.

    W&B is optional and imported lazily only when the first record is logged.
    The local run directory remains the authoritative experiment record; this
    logger only projects scalar metrics into dashboard-friendly W&B names.

    Parameters
    ----------
    project : str
        W&B project name.
    entity : str or None, optional
        W&B entity.
    name : str or None, optional
        W&B run name.
    group : str or None, optional
        W&B run group.
    job_type : str or None, optional
        W&B job type.
    tags : sequence of str or None, optional
        W&B tags. Any sequence is accepted (e.g. an OmegaConf list from a
        Hydra config) and is copied into a plain list.
    mode : {"online", "offline", "disabled"}, optional
        W&B logging mode.
    config : mapping or None, optional
        Small JSON-safe config/provenance payload for W&B config.
    log_config : bool, optional
        Whether to send ``config`` to W&B.
    log_code : bool, optional
        Whether to request W&B code saving.
    mirror_scalars : bool, optional
        Whether to mirror scalar records as ``<namespace>/<key>``.
    dashboard_aliases : bool, optional
        Whether to add the small documented ``dashboard/*`` alias set.
    health_flags : bool, optional
        Whether to derive compact numeric ``health/*`` flags.
    log_artifacts : bool, optional
        Whether explicit ``log_artifact`` calls should upload paths.
    """

    _VALID_MODES = {"online", "offline", "disabled"}

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: Sequence[str] | None = None,
        mode: str = "online",
        dir: str | None = None,
        config: Mapping[str, object] | None = None,
        log_config: bool = True,
        log_code: bool = False,
        mirror_scalars: bool = True,
        dashboard_aliases: bool = True,
        health_flags: bool = True,
        log_artifacts: bool = False,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}, got {mode!r}")
        self.project = project
        self.entity = entity
        self.name = name
        self.group = group
        self.job_type = job_type
        if isinstance(tags, str):
            raise TypeError(f"tags must be a sequence of strings, not a single string: {tags!r}")
        self.tags = None if tags is None else [str(tag) for tag in tags]
        self.mode = mode
        self.dir = dir
        self.config = None if config is None else _json_safe(dict(config))
        self.log_config = bool(log_config)
        self.log_code = bool(log_code)
        self.mirror_scalars = bool(mirror_scalars)
        self.dashboard_aliases = bool(dashboard_aliases)
        self.health_flags = bool(health_flags)
        self.log_artifacts = bool(log_artifacts)
        self._wandb = None
        self._run = None
        self._finished = False

    def log(self, record: LogRecord | Mapping[str, object]) -> None:
        """Log one record to W&B when it has scalar payload entries."""

        if self.mode == "disabled":
            return
        payload = project_record_to_wandb(
            record,
            mirror_scalars=self.mirror_scalars,
            dashboard_aliases=self.dashboard_aliases,
            health_flags=self.health_flags,
        )
        if not payload:
            return
        self._ensure_run()
        self._run.log(payload)
        self._update_summary(payload)

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """Upload an explicitly requested artifact when artifact logging is enabled."""

        if self.mode == "disabled" or not self.log_artifacts:
            return
        self._ensure_run()
        artifact_name = name or Path(path).name
        artifact = self._wandb.Artifact(artifact_name, type="spenn-artifact")
        artifact_path = Path(path)
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the W&B run if it was initialized."""

        if self._run is None or self._finished:
            return
        self._run.finish()
        self._finished = True

    def _ensure_run(self) -> None:
        """Initialize W&B once before the first emitted payload."""

        if self._run is not None:
            return
        try:
            self._wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "WandBLogger was configured, but the `wandb` package is not installed. "
                "Install optional W&B support, e.g. `uv sync --extra wandb`, or remove "
                "the WandB logger from the config."
            ) from exc
        self._run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            group=self.group,
            job_type=self.job_type,
            tags=self.tags,
            mode=self.mode,
            dir=self.dir,
            config=self.config if self.log_config else None,
            save_code=self.log_code,
        )
        self._define_metrics()

    def _define_metrics(self) -> None:
        """Register explicit W&B step axes when the active run supports them."""

        define_metric = getattr(self._run, "define_metric", None)
        if not callable(define_metric):
            return
        for metric, step_metric in (
            ("train/*", "train/step"),
            ("train/sampler/*", "train/step"),
            ("train/perf/*", "train/step"),
            # Validation runs inside the training lifecycle, so it shares the
            # train step axis.
            ("validation/*", "train/step"),
            ("validation/sampler/*", "train/step"),
            ("validation/perf/*", "train/step"),
            ("eval/*", "eval/step"),
            ("eval/sampler/*", "eval/step"),
            ("eval/perf/*", "eval/step"),
            ("checks/*", "checks/train_step"),
            ("diagnostics/*", "eval/step"),
            ("dashboard/*", "train/step"),
            ("health/*", "train/step"),
        ):
            define_metric(metric, step_metric=step_metric)

    def _update_summary(self, payload: Mapping[str, object]) -> None:
        """Mirror run-level timing/status metrics to W&B summary when available."""

        summary = getattr(self._run, "summary", None)
        if summary is None:
            return
        for key, value in payload.items():
            if key.startswith("runtime/") and _is_wandb_scalar(value):
                summary[key] = value


def project_record_to_wandb(
    record: LogRecord | Mapping[str, object],
    *,
    mirror_scalars: bool = True,
    dashboard_aliases: bool = True,
    health_flags: bool = True,
) -> dict[str, object]:
    """Project one backend-neutral record into a W&B payload."""

    step, namespace, metrics = _record_fields(record)
    payload: dict[str, object] = {}
    step_fields: dict[str, object] = {}
    if step is not None:
        _add_step_fields(step_fields, namespace, step)

    raw_metrics: dict[str, object] = {}
    for key, value in metrics.items():
        if not _is_wandb_scalar(value):
            continue
        raw_key = _join_metric(namespace, key)
        raw_metrics[raw_key] = value
        if mirror_scalars:
            payload[raw_key] = value

    if dashboard_aliases:
        for source, alias in _DASHBOARD_ALIASES.items():
            if source in raw_metrics:
                payload[alias] = raw_metrics[source]
                if step is not None:
                    payload.setdefault("train/step", step)

    if health_flags:
        flags = _derive_health_flags(raw_metrics)
        if flags:
            payload.update(flags)
            if step is not None:
                payload.setdefault("train/step", step)

    if not payload:
        return {}
    return {**step_fields, **payload}


_DASHBOARD_ALIASES = {
    "train/loss": "dashboard/loss",
    "train/energy": "dashboard/energy",
    "train/energy_variance": "dashboard/energy_variance",
    "train/energy_stderr": "dashboard/energy_stderr",
    "train/sampler/acceptance_rate": "dashboard/acceptance_rate",
    "train/grad_norm": "dashboard/grad_norm",
    "train/local_energy_finite_fraction": "dashboard/local_energy_finite_fraction",
    "train/perf/step_time_sec": "dashboard/step_time_sec",
    "runtime/wall_time_sec": "dashboard/wall_time_sec",
}


def _record_fields(record: LogRecord | Mapping[str, object]) -> tuple[int | None, str, Mapping[str, object]]:
    """Return ``(step, namespace, metrics)`` for supported record shapes."""

    if isinstance(record, LogRecord):
        return record.step, record.namespace, record.metrics
    step_value = record.get("step")
    step = None if step_value is None else int(step_value)
    namespace = str(record.get("namespace", "run"))
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        return step, namespace, metrics
    key = record.get("key")
    if key is None:
        return step, namespace, {}
    return step, namespace, {str(key): record.get("value")}


def _add_step_fields(payload: dict[str, object], namespace: str, step: int) -> None:
    if namespace == "train" or namespace.startswith("train/"):
        payload["train/step"] = step
    elif namespace == "validation" or namespace.startswith("validation/"):
        payload["train/step"] = step
    elif namespace == "eval" or namespace.startswith("eval/"):
        payload["eval/step"] = step
    elif namespace.startswith("checks/"):
        payload["checks/train_step"] = step
    elif namespace.startswith("diagnostics/"):
        payload["eval/step"] = step
    elif namespace == "runtime":
        payload["runtime/step"] = step


def _join_metric(namespace: str, key: str) -> str:
    namespace = namespace.strip("/")
    key = str(key).strip("/")
    return key if not namespace else f"{namespace}/{key}"


def _derive_health_flags(raw_metrics: Mapping[str, object]) -> dict[str, float]:
    flags: dict[str, float] = {}
    data_passed = raw_metrics.get("checks/data_integrity/passed")
    if isinstance(data_passed, bool):
        flags["health/numerics_ok"] = 1.0 if data_passed else 0.0
    elif raw_metrics.get("checks/data_integrity/local_energy_nonfinite_fraction") == 0:
        flags["health/numerics_ok"] = 1.0

    sampler_passed = raw_metrics.get("checks/sampler/passed")
    if isinstance(sampler_passed, bool):
        flags["health/sampler_ok"] = 1.0 if sampler_passed else 0.0

    equivariance_values = [
        value
        for key, value in raw_metrics.items()
        if key.startswith("checks/equivariance/") and key.endswith("/passed") and isinstance(value, bool)
    ]
    if equivariance_values:
        flags["health/equivariance_ok"] = 1.0 if all(equivariance_values) else 0.0

    failed = raw_metrics.get("runtime/failed")
    if isinstance(failed, bool):
        flags["health/run_ok"] = 0.0 if failed else 1.0
    elif flags:
        flags["health/run_ok"] = 1.0 if all(value == 1.0 for value in flags.values()) else 0.0
    return flags


def _is_wandb_scalar(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return str(value)



__all__ = ["WandB", "project_record_to_wandb"]
