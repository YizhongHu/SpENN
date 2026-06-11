"""Sampler health callback."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..base import Callback, Event


class SamplerHealth(Callback):
    """Expose sampler statistics under ``checks/sampler`` with optional bounds.

    Reads ``state.sampler_stats`` and logs only the stats actually available. When
    acceptance-rate bounds are configured and violated, ``passed`` is ``False``;
    it raises only if ``fail_fast`` is set.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        fail_fast: bool = False,
        min_acceptance_rate: float | None = None,
        max_acceptance_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.fail_fast = bool(fail_fast)
        self.min_acceptance_rate = None if min_acceptance_rate is None else float(min_acceptance_rate)
        self.max_acceptance_rate = None if max_acceptance_rate is None else float(max_acceptance_rate)

    def on_step_end(self, event: Event) -> None:
        """Log available sampler diagnostics and check crude bounds."""

        state = event.state
        stats = dict(getattr(state, "sampler_stats", None) or {})

        metrics: dict[str, Any] = {
            key: stats[key] for key in ("acceptance_rate", "n_walkers", "n_steps", "burn_in") if key in stats
        }

        failure: str | None = None
        acceptance_rate = stats.get("acceptance_rate")
        if acceptance_rate is not None:
            if self.min_acceptance_rate is not None and acceptance_rate < self.min_acceptance_rate:
                failure = (
                    f"acceptance_rate={acceptance_rate} below min_acceptance_rate={self.min_acceptance_rate}"
                )
            elif self.max_acceptance_rate is not None and acceptance_rate > self.max_acceptance_rate:
                failure = (
                    f"acceptance_rate={acceptance_rate} above max_acceptance_rate={self.max_acceptance_rate}"
                )

        metrics["passed"] = failure is None
        event.context.log(metrics, step=state.step, namespace="checks/sampler")
        if self.fail_fast and failure is not None:
            raise RuntimeError(f"SamplerHealth failed at step {state.step}: {failure}")



__all__ = ["SamplerHealth"]
