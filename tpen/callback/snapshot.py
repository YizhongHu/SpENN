"""Configuration snapshot callbacks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .base import Callback, Event


class ConfigSnapshot(Callback):
    """Write a re-runnable run configuration at run start."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Write ``config.yaml``."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(event.context.source_cfg, self.output_path, resolve=False)


class ResolvedConfigSnapshot(Callback):
    """Write the fully resolved run configuration at run start."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Write ``resolved_config.yaml``."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(event.context.cfg, self.output_path, resolve=True)



__all__ = ["ConfigSnapshot", "ResolvedConfigSnapshot"]
