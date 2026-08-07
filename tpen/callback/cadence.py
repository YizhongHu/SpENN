"""Occurrence-count cadence primitives for typed callback subscriptions."""

from __future__ import annotations

import random
from dataclasses import dataclass

from tpen.events import Subscription


@dataclass(frozen=True)
class Cadence:
    """Occurrence-count schedule for one subscription group.

    Parameters
    ----------
    every_n : int, optional
        Interval between eligible one-based occurrence counts.
    start : int, optional
        First eligible one-based occurrence count.
    max_calls : int or None, optional
        Maximum admitted logical coordinates across the whole group.
    probability : float, optional
        Admission probability after count-window filtering.
    seed : int or None, optional
        Seed for the group-local probability stream.
    """

    every_n: int = 1
    start: int = 1
    max_calls: int | None = None
    probability: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.every_n, int) or isinstance(self.every_n, bool):
            raise TypeError("every_n must be an integer")
        if self.every_n < 1:
            raise ValueError(f"every_n must be at least 1, got {self.every_n}")
        if not isinstance(self.start, int) or isinstance(self.start, bool):
            raise TypeError("start must be an integer")
        if self.start < 1:
            raise ValueError(f"start must be at least 1, got {self.start}")
        if self.max_calls is not None:
            if not isinstance(self.max_calls, int) or isinstance(self.max_calls, bool):
                raise TypeError("max_calls must be an integer or None")
            if self.max_calls < 0:
                raise ValueError(f"max_calls must be non-negative, got {self.max_calls}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")


class CadenceGate:
    """Mutable counter and RNG stream for one immutable `Cadence`.

    Parameters
    ----------
    cadence : Cadence
        Immutable scheduling values owned by this gate.
    """

    def __init__(self, cadence: Cadence) -> None:
        if not isinstance(cadence, Cadence):
            raise TypeError(f"cadence must be Cadence, got {type(cadence).__name__}")
        self.cadence = cadence
        self.num_calls = 0
        self._rng = random.Random(cadence.seed)

    def should_run(self, count: int) -> bool:
        """Return whether one new logical coordinate is admitted."""

        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"count must be a one-based integer, got {count!r}")
        cadence = self.cadence
        if cadence.max_calls is not None and self.num_calls >= cadence.max_calls:
            return False
        if count < cadence.start or (count - cadence.start) % cadence.every_n != 0:
            return False
        if cadence.probability <= 0.0:
            return False
        if cadence.probability < 1.0 and self._rng.random() >= cadence.probability:
            return False
        self.num_calls += 1
        return True

    def reset(self) -> None:
        """Reset the counter and recreate the RNG from the original seed."""

        self.num_calls = 0
        self._rng = random.Random(self.cadence.seed)


@dataclass(frozen=True)
class SubscriptionGroup:
    """Selectors sharing one optional occurrence cadence decision.

    Parameters
    ----------
    selectors : tuple of Subscription
        Typed deliveries sharing one logical decision.
    cadence : Cadence or None, optional
        Group-local schedule. ``None`` observes and delivers without a gate.
    """

    selectors: tuple[Subscription, ...]
    cadence: Cadence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selectors, tuple):
            raise TypeError("selectors must be a tuple of Subscription values")
        if not self.selectors:
            raise ValueError("a subscription group requires at least one selector")
        if not all(isinstance(selector, Subscription) for selector in self.selectors):
            raise TypeError("selectors must contain only Subscription values")
        if self.cadence is not None and not isinstance(self.cadence, Cadence):
            raise TypeError("cadence must be Cadence or None")


def validate_subscription_groups(groups: tuple[SubscriptionGroup, ...]) -> None:
    """Reject selectors whose deliveries overlap across cadence groups."""

    for first_index, first in enumerate(groups):
        for second_index in range(first_index + 1, len(groups)):
            second = groups[second_index]
            for first_selector in first.selectors:
                for second_selector in second.selectors:
                    if _subscriptions_overlap(first_selector, second_selector):
                        raise ValueError(
                            "typed subscription groups have overlapping deliveries: "
                            f"groups {first_index} and {second_index}"
                        )


def _subscriptions_overlap(first: Subscription, second: Subscription) -> bool:
    """Return whether two selectors can deliver the same occurrence."""

    if first.lifecycle is not second.lifecycle:
        return False
    return issubclass(first.subject, second.subject) or issubclass(
        second.subject, first.subject
    )


__all__ = [
    "Cadence",
    "CadenceGate",
    "SubscriptionGroup",
    "validate_subscription_groups",
]
