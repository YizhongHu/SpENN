"""Sampler health callback."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..base import Callback, Event


class SamplerHealth(Callback):
    """Expose sampler statistics under ``checks/sampler`` with optional bounds.

    Reads the typed `tpen.sampling.SamplerStats` record on ``state.sampler_stats``
    and lets that record compose the check key set, so ``checks/sampler`` names
    are spelled in exactly one place. A state carrying no diagnostics still
    reports ``passed``. When acceptance-rate bounds are configured and violated,
    ``passed`` is ``False``; it raises only if ``fail_fast`` is set.
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
        # ``Event.state`` is untyped at the legacy ingress, so the state itself
        # is probed with a default; the record it yields is then read typed.
        stats = getattr(state, "sampler_stats", None)

        metrics: dict[str, Any] = {} if stats is None else stats.as_check_metrics()

        failure = None if stats is None else self._acceptance_failure(stats.acceptance_rate)

        metrics["passed"] = failure is None
        event.context.log(metrics, step=state.step, namespace="checks/sampler")
        if self.fail_fast and failure is not None:
            raise RuntimeError(f"SamplerHealth failed at step {state.step}: {failure}")

    def _acceptance_failure(self, acceptance_rate: float) -> str | None:
        """Return a bound-violation description, or ``None`` when in bounds."""

        if self.min_acceptance_rate is not None and acceptance_rate < self.min_acceptance_rate:
            return f"acceptance_rate={acceptance_rate} below min_acceptance_rate={self.min_acceptance_rate}"
        if self.max_acceptance_rate is not None and acceptance_rate > self.max_acceptance_rate:
            return f"acceptance_rate={acceptance_rate} above max_acceptance_rate={self.max_acceptance_rate}"
        return None


__all__ = ["SamplerHealth"]
