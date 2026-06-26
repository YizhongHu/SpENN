"""Passive, traceable base class for particle-permutation-equivariant modules.

`EquivariantMap` is a pure computation module plus a passive tracing wrapper.
It does not check equivariance, permute inputs/outputs, or compare values.
Subclasses implement :meth:`forward_impl`; :meth:`forward` runs it and, when a
trace is active, records the output. Equivariance is verified separately by the
runtime checkers in :mod:`spenn.equivariance.checks` and by pytest-only helpers
under ``tests/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from spenn.dependencies import require_torch_nn
from spenn.equivariance.trace import trace_equivariant

nn = require_torch_nn(feature="equivariant neural-network maps")


class EquivariantMap(nn.Module, ABC):
    """Base class for traced equivariant-state modules.

    The public `forward` is the normal execution path: it owns trace recording
    and delegates computation to the abstract :meth:`forward_impl`. Subclasses
    implement ``forward_impl``, not ``forward``, unless they have a specific
    reason to bypass tracing. Runtime checkers still call the normal ``forward``;
    ``forward_impl`` is the internal template method. ``EquivariantMap`` does not
    check equivariance, permute, or compare -- that is done by the checkers in
    :mod:`spenn.equivariance.checks` and pytest-only helpers under ``tests/``.

    Parameters
    ----------
    trace_name : str or None, optional
        Explicit stable name for this producer in an active
        `EquivarianceTrace`. Overrides PyTorch module-path resolution.
    trace_output : bool, optional
        Whether `forward` records its output to an active trace.
    """

    def __init__(
        self,
        *,
        trace_name: str | None = None,
        trace_output: bool = True,
    ) -> None:
        super().__init__()
        self.trace_name = trace_name
        self.trace_output = bool(trace_output)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run :meth:`forward_impl` and passively trace the output."""

        output = self.forward_impl(*args, **kwargs)
        if self.trace_output:
            self.trace("output", output)
        return output

    @abstractmethod
    def forward_impl(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the module output without owning trace recording."""

        raise NotImplementedError(f"{type(self).__name__}.forward_impl is not implemented")

    def trace(self, slot: str, value: Any) -> None:
        """Record a semantic equivariant value to the active trace, if any.

        Complex modules may record meaningful internal values from within
        :meth:`forward_impl` (e.g. ``self.trace("interaction", interaction)``).
        Trace only semantic equivariant data, not every temporary tensor.
        """

        trace_equivariant(value=value, slot=slot, producer=self)


__all__ = ["EquivariantMap"]
