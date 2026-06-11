"""Base runner target and runtime placement helpers."""

from __future__ import annotations

import importlib
from typing import Any

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



def _place_module_for_runtime(module: Any, context: RunContext) -> None:
    """Move a configured module to the run's device and floating dtype."""

    torch = _torch()
    module.to(device=torch.device(context.metadata.device), dtype=_runtime_dtype(context.metadata.dtype))


def _assert_eager_initialized(module: Any) -> None:
    """Fail before runner use if a model still contains lazy state."""

    uninitialized_parameter, uninitialized_buffer = _uninitialized_parameter_types()
    for name, parameter in module.named_parameters():
        if isinstance(parameter, uninitialized_parameter):
            raise RuntimeError(f"model parameter {name!r} is uninitialized before runner use")
    for name, buffer in module.named_buffers():
        if isinstance(buffer, uninitialized_buffer):
            raise RuntimeError(f"model buffer {name!r} is uninitialized before runner use")


def _is_torch_module(value: object) -> bool:
    """Return whether `value` is a ``torch.nn.Module`` without import-time torch.nn coupling."""

    module_type = _torch_module_type(required=False)
    return module_type is not None and isinstance(value, module_type)



def _runtime_dtype(name: str) -> Any:
    torch = _torch()
    try:
        dtype = getattr(torch, str(name))
    except AttributeError as exc:
        raise ValueError(f"Unsupported runtime dtype {name!r}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported runtime dtype {name!r}")
    if not dtype.is_floating_point:
        raise ValueError(f"Runtime dtype must be floating point, got {name!r}")
    return dtype


def _torch() -> Any:
    return importlib.import_module("torch")


def _torch_module_type(*, required: bool) -> type[object] | None:
    try:
        nn = importlib.import_module("torch.nn")
    except ModuleNotFoundError as exc:
        if exc.name != "torch.nn":
            raise
        if required:
            raise _torch_nn_runtime_error() from exc
        return None
    return nn.Module


def _uninitialized_parameter_types() -> tuple[type[object], type[object]]:
    try:
        parameter = importlib.import_module("torch.nn.parameter")
    except ModuleNotFoundError as exc:
        if exc.name != "torch.nn" and exc.name != "torch.nn.parameter":
            raise
        raise _torch_nn_runtime_error() from exc
    return parameter.UninitializedParameter, parameter.UninitializedBuffer


def _torch_nn_runtime_error() -> RuntimeError:
    return RuntimeError(
        "PyTorch was imported, but `torch.nn` is unavailable. Launch the run from a complete "
        "PyTorch environment, for example with this project's `uv` environment and the selected "
        "torch extra."
    )



__all__ = ["Runner"]
