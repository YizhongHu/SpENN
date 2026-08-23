"""Accelerator backend resolution for a configured runtime device.

This module owns the mapping from a device (``runtime.device``) to the torch
module implementing that device's accelerator API, so no other call site names a
concrete backend such as ``torch.cuda``. Naming a backend directly is what makes
code CUDA-only: on an XPU build ``torch.cuda`` still exists and
``torch.cuda.is_available()`` returns ``False`` without raising, so a hardcoded
CUDA path silently degrades to CPU instead of failing.

``torch.get_device_module`` exposes the same surface for every backend
(``manual_seed_all``, ``get_rng_state_all``, ``synchronize``, memory statistics),
so no per-backend branching is required here.

See ADR-013: generator state is device-bound and never portable across device
types. Nothing in this module reinterprets RNG state; `canonical_device` only
makes devices comparable so callers can detect a mismatch and fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tpen.dependencies import require_torch

_MIN_TORCH_HINT = "torch>=2.5 provides torch.get_device_module"


def _torch(feature: str) -> Any:
    torch = require_torch(feature=feature)
    if not hasattr(torch, "get_device_module"):  # pragma: no cover - ancient torch
        raise RuntimeError(f"{feature} requires a torch exposing get_device_module ({_MIN_TORCH_HINT}); got {torch.__version__}")
    return torch


def _optional_device_module(torch: Any, device_type: str) -> Any | None:
    """Return the device module for ``device_type``, or ``None`` when none exists.

    Device types such as ``meta`` are valid to construct but have no registered
    accelerator module, and therefore no current device to resolve.
    """

    try:
        return torch.get_device_module(device_type)
    except (AttributeError, RuntimeError):
        return None


def current_accelerator_type(*, feature: str = "accelerator detection") -> str:
    """Return the active accelerator device type, or ``"cpu"`` when there is none.

    Parameters
    ----------
    feature : str, optional
        Feature name used in the torch-missing error message.

    Returns
    -------
    str
        Device type such as ``"cuda"``, ``"xpu"``, or ``"cpu"``.
    """

    torch = _torch(feature)
    accelerator = getattr(torch, "accelerator", None)
    current = getattr(accelerator, "current_accelerator", None)
    if callable(current):
        device = current()
        if device is not None:
            return str(torch.device(device).type)
    return "cpu"


def device_module(device: Any = None, *, feature: str = "accelerator support") -> Any:
    """Return the torch module implementing ``device``'s accelerator API.

    Parameters
    ----------
    device : torch.device, str, or None, optional
        Target device. ``None`` resolves the active accelerator, falling back to
        CPU when no accelerator is present.
    feature : str, optional
        Feature name used in the torch-missing error message.

    Returns
    -------
    module
        ``torch.cpu``, ``torch.cuda``, ``torch.xpu``, or another backend module.
    """

    torch = _torch(feature)
    device_type = current_accelerator_type(feature=feature) if device is None else str(torch.device(device).type)
    return torch.get_device_module(device_type)


def canonical_device(device: Any, *, feature: str = "device comparison") -> Any:
    """Return a fully indexed device so ``cuda`` and ``cuda:0`` compare equal.

    Tensors report an indexed accelerator device (``cuda:0``, ``xpu:0``) while
    configs and callers usually pass the index-less form, and ``torch.device``
    treats those as unequal. CPU devices are reported index-less by tensors and
    pass through unchanged, as do devices that already carry an index, device
    types with no registered accelerator module (such as ``meta``), and
    accelerators that are not currently available.

    Parameters
    ----------
    device : torch.device or str
        Device to canonicalize.
    feature : str, optional
        Feature name used in the torch-missing error message.

    Returns
    -------
    torch.device
        Device with an explicit index when one can be resolved.
    """

    torch = _torch(feature)
    resolved = torch.device(device)
    if resolved.type == "cpu" or resolved.index is not None:
        return resolved
    module = _optional_device_module(torch, resolved.type)
    if module is None:
        return resolved
    is_available = getattr(module, "is_available", None)
    if not callable(is_available) or not is_available():
        return resolved
    current_device = getattr(module, "current_device", None)
    if not callable(current_device):
        return resolved
    return torch.device(resolved.type, current_device())


def synchronize(device: Any = None, *, feature: str = "device synchronization") -> None:
    """Synchronize the accelerator so host-side timing measures device work.

    Does nothing when the accelerator is unavailable or exposes no
    ``synchronize``, which keeps CPU runs free of accelerator calls.

    Parameters
    ----------
    device : torch.device, str, or None, optional
        Device to synchronize. ``None`` uses the active accelerator.
    feature : str, optional
        Feature name used in the torch-missing error message.
    """

    module = device_module(device, feature=feature)
    is_available = getattr(module, "is_available", None)
    if callable(is_available) and not is_available():
        return
    synchronize_fn = getattr(module, "synchronize", None)
    if callable(synchronize_fn):
        synchronize_fn()


@dataclass
class DeviceEventTimer:
    """Measure elapsed device time with one accelerator event pair.

    Instances are deliberately single-use between `start` and `stop`: one
    operation scope owns one timer, so accepting nested starts would silently
    associate the wrong device interval with a metric.
    """

    _module: Any
    _event_factory: Any
    _start_event: Any | None = None

    def start(self) -> None:
        """Record the beginning of one device interval."""

        if self._start_event is not None:
            raise RuntimeError("device event timer is already running")
        start_event = self._event_factory(enable_timing=True)
        start_event.record()
        self._start_event = start_event

    def stop(self) -> float:
        """Synchronize and return elapsed device seconds for one interval."""

        start_event = self._start_event
        if start_event is None:
            raise RuntimeError("device event timer was not started")
        try:
            end_event = self._event_factory(enable_timing=True)
            end_event.record()
            event_synchronize = getattr(end_event, "synchronize", None)
            if callable(event_synchronize):
                event_synchronize()
            else:
                # Older backends expose no event-local wait. Keep this fallback
                # explicit and isolated to the opt-in device-event backend.
                self._module.synchronize()
            return float(start_event.elapsed_time(end_event)) / 1_000.0
        finally:
            self._start_event = None


def device_event_timer(
    device: Any = None, *, feature: str = "device event timing"
) -> DeviceEventTimer:
    """Return a fail-loud event timer for an available accelerator.

    Unlike `synchronize`, this never degrades to a host timer: callers opt in
    precisely because they need device elapsed time.
    """

    module = device_module(device, feature=feature)
    available = getattr(module, "is_available", None)
    if not callable(available) or not available():
        raise RuntimeError(f"{feature} requires an available accelerator")
    event_factory = getattr(module, "Event", None)
    synchronize_fn = getattr(module, "synchronize", None)
    if not callable(event_factory) or not callable(synchronize_fn):
        raise RuntimeError(f"{feature} requires timing-event support from the accelerator backend")
    return DeviceEventTimer(_module=module, _event_factory=event_factory)


def seed_all(seed: int, *, feature: str = "seeded run") -> None:
    """Seed the accelerator's global RNG for every visible device.

    This is run-level global seeding only. Per ADR-013, stochastic components own
    their own generators and must not be seeded on their behalf; this function
    exists so the one explicit run-level seed reaches the accelerator whatever
    the backend.

    Parameters
    ----------
    seed : int
        Seed value.
    feature : str, optional
        Feature name used in the torch-missing error message.
    """

    module = device_module(feature=feature)
    seed_fn = getattr(module, "manual_seed_all", None)
    if callable(seed_fn):
        seed_fn(int(seed))


__all__ = [
    "DeviceEventTimer",
    "canonical_device",
    "current_accelerator_type",
    "device_event_timer",
    "device_module",
    "seed_all",
    "synchronize",
]
