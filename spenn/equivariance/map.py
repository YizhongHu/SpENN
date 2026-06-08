"""Passive, traceable base class for particle-permutation-equivariant modules.

`EquivariantMap` is a pure computation module plus a passive tracing wrapper.
It does not check equivariance, permute inputs/outputs, or compare values.
Subclasses implement :meth:`forward_impl`; :meth:`forward` runs it and, when a
trace is active, records the output. Equivariance is verified separately by
test-time helpers in :mod:`spenn.testing.equivariance` and, in the future, by
trace-consuming runtime checkers.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from spenn.equivariance.trace import trace_equivariant


class EquivariantMap(nn.Module):
    """Base class for modules that commute with particle permutations.

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

    def forward_impl(self, *args: Any, **kwargs: Any) -> Any:
        """Implement the module computation in subclasses."""

        raise NotImplementedError(f"{type(self).__name__}.forward_impl is not implemented")

    def trace(self, slot: str, value: Any) -> None:
        """Record a semantic equivariant value to the active trace, if any.

        Complex modules may record meaningful internal values from within
        :meth:`forward_impl` (e.g. ``self.trace("interaction", interaction)``).
        Trace only semantic equivariant data, not every temporary tensor.
        """

        trace_equivariant(value=value, slot=slot, producer=self)


__all__ = ["EquivariantMap"]
