"""Base runner target and runtime placement helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from spenn.artifacts import RunContext, RunResult
from spenn.callback.base import Event


class Runner:
    """Base runner with callback lifecycle dispatch.

    Callbacks and loggers are owned by the `RunContext` (configured at the
    config root); ``emit`` dispatches lifecycle events into ``context.callbacks``
    and runners log through ``context.log``.
    """

    def emit(
        self,
        name: str,
        context: RunContext,
        *,
        state: object | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one lifecycle event to the context's callbacks."""

        event = Event(name=name, context=context, state=state, payload={} if payload is None else payload)
        for callback in context.callbacks:
            callback.handle(event)

    def run(self, context: RunContext) -> RunResult:
        """Execute a configured run."""

        raise NotImplementedError



def _place_module_for_runtime(module: torch.nn.Module, context: RunContext) -> None:
    """Move a configured module to the run's device and floating dtype."""

    module.to(device=torch.device(context.metadata.device), dtype=_runtime_dtype(context.metadata.dtype))


def _assert_eager_initialized(module: torch.nn.Module) -> None:
    """Fail before runner use if a model still contains lazy state."""

    for name, parameter in module.named_parameters():
        if isinstance(parameter, UninitializedParameter):
            raise RuntimeError(f"model parameter {name!r} is uninitialized before runner use")
    for name, buffer in module.named_buffers():
        if isinstance(buffer, UninitializedBuffer):
            raise RuntimeError(f"model buffer {name!r} is uninitialized before runner use")



def _runtime_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, str(name))
    except AttributeError as exc:
        raise ValueError(f"Unsupported runtime dtype {name!r}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported runtime dtype {name!r}")
    if not dtype.is_floating_point:
        raise ValueError(f"Runtime dtype must be floating point, got {name!r}")
    return dtype



__all__ = ["Runner"]
