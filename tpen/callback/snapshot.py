"""Configuration snapshot callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tpen.artifacts import RunContext
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.run_events import RunStarted

from .base import Callback
from .cadence import SubscriptionGroup


class _RunStartSnapshot(Callback):
    """Write one configuration file at the run's start boundary.

    Both snapshot callbacks observe exactly one moment and differ only in which
    config they save and whether they resolve interpolations, so the
    subscription plan lives here once. Data-free, hence a plain
    `tpen.callback.Callback`: the configs are already on the run context.

    Neither takes ``triggers`` any more (ADR-E002); a config still passing the
    key fails loudly with a duplicate-argument ``TypeError`` instead of silently
    running on both the typed and the legacy path.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination for the snapshot.
    **kwargs
        Forwarded to `tpen.callback.Callback`.
    """

    def __init__(self, output_path: str | Path, **kwargs: Any) -> None:
        # Reject the undeclared base at construction rather than at save time,
        # the same reason `StatefulCallback.__init__` and
        # `tpen.training.events.TrainingPhase` do.
        if type(self) is _RunStartSnapshot:
            raise TypeError(
                "_RunStartSnapshot is abstract; use ConfigSnapshot or ResolvedConfigSnapshot"
            )
        super().__init__(
            triggers=(),
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(RunStarted),)),
            ),
            **kwargs,
        )
        self.output_path = Path(output_path)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Save the snapshot when the run starts."""

        if not isinstance(occurrence.event, RunStarted):
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save(context)

    def _save(self, context: RunContext) -> None:
        """Write this snapshot's configuration; declared by each subclass."""

        raise NotImplementedError


class ConfigSnapshot(_RunStartSnapshot):
    """Write a re-runnable run configuration at run start."""

    def _save(self, context: RunContext) -> None:
        """Write ``config.yaml`` with interpolations left unresolved."""

        OmegaConf.save(context.source_cfg, self.output_path, resolve=False)


class ResolvedConfigSnapshot(_RunStartSnapshot):
    """Write the fully resolved run configuration at run start."""

    def _save(self, context: RunContext) -> None:
        """Write ``resolved_config.yaml`` with every interpolation resolved."""

        OmegaConf.save(context.cfg, self.output_path, resolve=True)



__all__ = ["ConfigSnapshot", "ResolvedConfigSnapshot"]
