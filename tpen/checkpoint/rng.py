"""Checkpoint RNG state: what is persisted, and where it may be restored.

Both halves of the contract live in one module because they are one contract:
the writer records the device its accelerator RNG state belongs to, and the
reader refuses to resume anywhere that device cannot be reproduced.

Validation and application are separate entry points on purpose.
`require_restorable_rng_state` must run *before* any component is restored: the
sampler's own restore recreates its generator on a device mismatch (see
`tpen.sampling.metropolis.MetropolisSampler.load_mcmc_state_dict`), reseeding it
or leaving it unseeded, so a refusal raised after that point would already have
reset the dominant RNG source. A refused resume must leave the process
unmutated.

Generator state is device-bound (ADR-013), and ``get_rng_state_all`` returns a
*positional, per-device* list. torch refuses outright incompatible state
(``cpu.set_state(xpu_state)`` raises ``RuntimeError: Expected a
CPUGeneratorImplState of size 5056 but found ... 16``), so this module makes that
existing hard failure legible and moves it earlier. It also covers what torch
accepts silently: a changed visible-device set shifts the positional list, and a
changed device index continues the run on a device whose stream is not the one
that was captured.

Only CUDA accelerator RNG state is persisted today. A run on any other live
accelerator is *refused* at resume rather than resumed on a different random
stream. Widening persistence to other backends is a separate change; until then
refusing is the only option that does not silently diverge.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tpen.accelerator import canonical_device, device_module

_FEATURE = "checkpoint RNG state"

# Per-device accelerator RNG states. Named for CUDA because CUDA is the only
# backend whose state is persisted; `BACKEND_KEY` records which backend a payload
# actually came from, so "absent" and "mismatched" stay distinguishable without
# inferring either from this key's presence.
ACCELERATOR_STATE_KEY = "torch_cuda"

# Device provenance of the accelerator RNG state, recorded whether or not any
# state was persisted. `DEVICE_KEY` doubles as the marker that a payload carries
# provenance at all, and the guard derives the backend from it rather than
# reading `BACKEND_KEY`, so the two can never disagree; `BACKEND_KEY` is durable
# provenance for readers and tooling.
BACKEND_KEY = "accelerator_backend"
DEVICE_KEY = "accelerator_device"
DEVICES_KEY = "accelerator_devices"


def runtime_device(context: Any) -> Any:
    """Return the run's declared compute device, defaulting to CPU.

    This is the device the RNG contract is written against: it is recorded with
    the state on save and compared against on restore. It mirrors
    `tpen.artifacts.RunMetadata.device`, whose own default is ``"cpu"``, so a
    context carrying no metadata is treated as a CPU run rather than inheriting
    whatever accelerator the host happens to expose.

    Parameters
    ----------
    context : Any
        Run context supplying ``metadata.device``.

    Returns
    -------
    Any
        A device accepted by `tpen.accelerator.canonical_device`. Never
        ``None``, which that function does not accept.
    """

    metadata = getattr(context, "metadata", None)
    device = getattr(metadata, "device", None)
    return "cpu" if device is None else device


def rng_state_dict(device: Any) -> dict[str, Any]:
    """Return the RNG state to persist for a run on `device`.

    Parameters
    ----------
    device : torch.device or str
        The run's declared compute device. Canonicalized so an index-less
        ``"cuda"`` records the index it actually resolves to.

    Returns
    -------
    dict
        CPU, Python, and NumPy RNG state, plus the device provenance of the
        accelerator RNG state. `ACCELERATOR_STATE_KEY` is present only for a
        live CUDA device.
    """

    import torch

    resolved = canonical_device(device, feature=_FEATURE)
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        BACKEND_KEY: resolved.type,
        DEVICE_KEY: str(resolved),
        DEVICES_KEY: [],
    }
    # CUDA-only by design; see the module docstring. The module is resolved
    # through `tpen.accelerator` rather than naming `torch.cuda` so widening
    # this is a change to the predicate, not to the call.
    if draws_from_accelerator(resolved) and resolved.type == "cuda":
        accelerator_states = device_module(resolved, feature=_FEATURE).get_rng_state_all()
        state[ACCELERATOR_STATE_KEY] = accelerator_states
        # The devices the positional state list covers, in list order.
        state[DEVICES_KEY] = _device_names(resolved.type, len(accelerator_states))
    try:
        import numpy as np
    except ImportError:
        state["numpy"] = None
    else:
        state["numpy"] = np.random.get_state()
    return state


def require_restorable_rng_state(state: Mapping[str, Any], device: Any, checkpoint_dir: Path) -> None:
    """Refuse the resume unless `state`'s random streams are reproducible on `device`.

    Call this before restoring any component. It only reads `state`; nothing in
    the process is mutated, so a refusal leaves the run untouched.

    Parameters
    ----------
    state : Mapping
        Payload written by `rng_state_dict`.
    device : torch.device or str
        The resuming run's declared compute device.
    checkpoint_dir : pathlib.Path
        Checkpoint directory, used to label refusals.

    Raises
    ------
    ValueError
        If the payload records no device provenance, if its recorded device
        differs from this run's, or if this run draws from an accelerator whose
        RNG state the payload does not carry.
    """

    import torch

    resolved = canonical_device(device, feature=_FEATURE)
    recorded_device = state.get(DEVICE_KEY)

    # Provenance is required, not optional. A payload written before provenance
    # was recorded cannot prove which device's streams it captured, so it cannot
    # prove a resume is faithful -- the same reason C1 refuses a v1 manifest for
    # `train_resume` rather than guessing. Refusing is deliberate: silently
    # restoring such a payload would exempt exactly the artifacts that predate
    # this guard from the guard.
    if recorded_device is None:
        raise ValueError(
            f"{checkpoint_dir}: checkpoint records no RNG device provenance ({DEVICE_KEY!r} is "
            f"absent), so it cannot prove which device's random streams it captured; resume is "
            f"unsupported for this checkpoint. Restore with load.mode=model_only to start a fresh "
            f"random stream deliberately."
        )

    recorded = torch.device(str(recorded_device))
    if recorded.type != resolved.type:
        raise ValueError(
            f"{checkpoint_dir}: checkpoint recorded RNG state for backend {recorded.type!r}, but "
            f"this run is on backend {resolved.type!r}; resume is unsupported across backends "
            f"because generator state is device-bound and cannot be reinterpreted."
        )
    if recorded != resolved:
        raise ValueError(
            f"{checkpoint_dir}: checkpoint recorded RNG state on device {recorded}, but this run "
            f"is on device {resolved}; resume is unsupported on a different device because the "
            f"captured streams belong to the device that wrote them, so the resumed run would not "
            f"reproduce them."
        )

    if not draws_from_accelerator(resolved):
        return

    if not _has_accelerator_state(state):
        raise ValueError(
            f"{checkpoint_dir}: this run draws from accelerator {resolved}, but the checkpoint "
            f"carries no accelerator RNG state (recorded backend {recorded.type!r}); resume is "
            f"unsupported in this configuration because the accelerator random stream cannot be "
            f"restored. Only CUDA accelerator RNG state is persisted; restore with "
            f"load.mode=model_only to start a fresh random stream deliberately."
        )

    recorded_devices = list(state.get(DEVICES_KEY) or [])
    current_devices = _device_names(resolved.type, device_module(resolved, feature=_FEATURE).device_count())
    if recorded_devices != current_devices:
        raise ValueError(
            f"{checkpoint_dir}: checkpoint recorded RNG state for devices {recorded_devices}, but "
            f"this run has devices {current_devices}; resume is unsupported when the visible "
            f"device set changes because the per-device state list is assigned positionally, so "
            f"each stream would be rebound to a different device."
        )


def apply_rng_state(state: Mapping[str, Any], device: Any) -> None:
    """Restore `state` into the global RNGs.

    Mutates process state. `require_restorable_rng_state` must have been called
    for the same payload and device first; this function assumes the streams are
    restorable and does not re-check.

    Parameters
    ----------
    state : Mapping
        Payload written by `rng_state_dict`.
    device : torch.device or str
        The resuming run's declared compute device.
    """

    import torch

    resolved = canonical_device(device, feature=_FEATURE)
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if _has_accelerator_state(state) and draws_from_accelerator(resolved):
        module = device_module(resolved, feature=_FEATURE)
        module.set_rng_state_all(state[ACCELERATOR_STATE_KEY])
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        try:
            import numpy as np
        except ImportError:
            return
        np.random.set_state(state["numpy"])


def draws_from_accelerator(device: Any) -> bool:
    """Return whether a run on `device` draws from an accelerator RNG stream.

    Parameters
    ----------
    device : torch.device or str
        A device already canonicalized by `tpen.accelerator.canonical_device`.

    Returns
    -------
    bool
        ``True`` only for a device carrying an index. `canonical_device`
        resolves an index only when the backend module exists, exposes a
        callable ``is_available`` that returns ``True``, *and* exposes a
        callable ``current_device``; anything else passes through index-free.
        An index is therefore the available-accelerator signal, and reading it
        needs no ``get_device_module`` lookup -- which is what keeps a device
        type with no accelerator module (``meta``) and the torch-internal error
        such a lookup raises out of this path.

    Notes
    -----
    Known limitation, deliberately not fixed here. A backend module that is
    available but exposes no ``current_device`` (reported for ``torch.mps``)
    stays index-free and answers ``False``, so a run on it is not covered by the
    accelerator-RNG guard and keeps the pre-guard silent-divergence behaviour.
    Closing that needs `canonical_device` to resolve an index for such a backend;
    probing the module here instead would reintroduce the ``meta`` regression
    this predicate exists to avoid.
    """

    import torch

    resolved = torch.device(device)
    return resolved.type != "cpu" and resolved.index is not None


def _has_accelerator_state(state: Mapping[str, Any]) -> bool:
    """Return whether `state` carries a non-empty per-device RNG state list.

    An empty list is treated as absent: ``set_rng_state_all([])`` is a no-op, so
    accepting it would report a successful resume that restored nothing.
    """

    accelerator_states = state.get(ACCELERATOR_STATE_KEY)
    return accelerator_states is not None and len(accelerator_states) > 0


def _device_names(device_type: str, count: int) -> list[str]:
    """Return the ordered device names a positional RNG state list covers."""

    return [f"{device_type}:{index}" for index in range(count)]


__all__ = [
    "ACCELERATOR_STATE_KEY",
    "BACKEND_KEY",
    "DEVICES_KEY",
    "DEVICE_KEY",
    "apply_rng_state",
    "draws_from_accelerator",
    "require_restorable_rng_state",
    "rng_state_dict",
    "runtime_device",
]
