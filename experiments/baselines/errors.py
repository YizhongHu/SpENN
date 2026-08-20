"""Failure type shared by the baseline run adapters.

Lives in its own module so that :mod:`experiments.baselines.statistics` can
raise it without importing an adapter, and so that both adapters raise the same
type. Adapters re-export it for callers that import it from the adapter module.
"""

from __future__ import annotations


class AdapterError(RuntimeError):
    """A run directory could not be turned into a record.

    Raised rather than returning a partial record: a run with no usable energy
    must fail loudly, never appear as a record with a null energy or vanish
    from the collection silently.
    """


__all__ = ["AdapterError"]
