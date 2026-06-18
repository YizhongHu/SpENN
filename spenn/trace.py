"""Passive trace infrastructure for model instrumentation."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable

import torch

from spenn.data.equivariant_state import compare_tensor_blocks
from spenn.data.indices import permute_particle_axis
from spenn.data.permutation import Permutation


class TraceWarning(UserWarning):
    """Warning for imperfect trace naming."""


@runtime_checkable
class PermutableTraceValue(Protocol):
    """Trace value with an explicit semantic particle-permutation action."""

    def permute(self, permutation: Permutation) -> Self:
        """Return this value with the particle labels permuted."""

        ...


@dataclass(frozen=True)
class ParticleTensor:
    """Tensor trace value with an explicit particle axis."""

    value: torch.Tensor
    particle_axis: int

    def permute(self, permutation: Permutation) -> "ParticleTensor":
        """Return this tensor with its declared particle axis permuted."""

        axis = self.particle_axis
        ndim = self.value.ndim
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"particle_axis {self.particle_axis} is out of range for shape {tuple(self.value.shape)}")
        if self.value.shape[axis] != len(permutation):
            raise ValueError(
                f"particle_axis {self.particle_axis} has size {self.value.shape[axis]}, "
                f"but permutation has size {len(permutation)}"
            )
        return ParticleTensor(
            value=permute_particle_axis(self.value, permutation, axis=axis),
            particle_axis=self.particle_axis,
        )

    def compare(
        self,
        other: "ParticleTensor",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float | int | bool | str | None]]:
        """Compare particle-tensor values with the same declared axis."""

        if type(other) is not type(self) or self.particle_axis != other.particle_axis:
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks((self.value,), (other.value,), atol=atol, rtol=rtol)


@dataclass(frozen=True)
class TraceEntry:
    """One recorded semantic value."""

    index: int
    key: str
    value: Any
    slot: str
    producer_name: str | None = None
    producer_type: str | None = None
    fallback_name: bool = False
    semantic_type: str | None = None


_active_trace: ContextVar["Trace | None"] = ContextVar("spenn_active_trace", default=None)


class Trace:
    """Context-local passive recorder of semantic values during forward passes."""

    def __init__(self, model: Any = None) -> None:
        self._entries: list[TraceEntry] = []
        self._by_key: dict[str, TraceEntry] = {}
        self._module_paths: dict[int, str] = {}
        self._fallback_class_counts: dict[str, int] = {}
        self._fallback_assigned: dict[int, str] = {}
        self._warned_fallback: set[int] = set()
        if model is not None:
            for name, module in model.named_modules():
                if name:
                    self._module_paths[id(module)] = name

    @classmethod
    @contextmanager
    def capture(cls, model: Any = None) -> Iterator["Trace"]:
        """Activate a new context-local trace."""

        trace = cls(model=model)
        token = _active_trace.set(trace)
        try:
            yield trace
        finally:
            _active_trace.reset(token)

    @property
    def entries(self) -> tuple[TraceEntry, ...]:
        """Return recorded entries in insertion order."""

        return tuple(self._entries)

    def by_key(self) -> Mapping[str, TraceEntry]:
        """Return recorded entries keyed by their unique trace key."""

        return dict(self._by_key)

    def keys(self) -> tuple[str, ...]:
        """Return recorded keys in insertion order."""

        return tuple(entry.key for entry in self._entries)

    def __len__(self) -> int:
        """Return the number of recorded entries."""

        return len(self._entries)

    def __iter__(self) -> Iterator[TraceEntry]:
        """Iterate over recorded entries."""

        return iter(self._entries)

    def __contains__(self, key: str) -> bool:
        """Return whether `key` was recorded."""

        return key in self._by_key

    def __getitem__(self, key: str) -> TraceEntry:
        """Return the entry recorded under `key`."""

        return self._by_key[key]

    def get(self, key: str, default: Any = None) -> TraceEntry | Any:
        """Return the entry recorded under `key`, or `default`."""

        return self._by_key.get(key, default)

    def record(
        self,
        *,
        value: Any,
        slot: str = "output",
        key: str | None = None,
        producer: object | None = None,
        semantic_type: str | None = None,
    ) -> TraceEntry:
        """Record one value, resolving and disambiguating its key."""

        producer_type = None if producer is None else type(producer).__name__
        if key is not None:
            entry_key = key
            producer_name = self._resolve_base_name(producer, warn=False)[0]
            fallback = False
        else:
            producer_name, fallback = self._resolve_base_name(producer, warn=True)
            base = producer_name if producer_name is not None else slot
            entry_key = base if producer_name is None else f"{producer_name}/{slot}"

        entry_key = self._disambiguate(entry_key)
        entry = TraceEntry(
            index=len(self._entries),
            key=entry_key,
            value=value,
            slot=slot,
            producer_name=producer_name,
            producer_type=producer_type,
            fallback_name=fallback,
            semantic_type=semantic_type,
        )
        self._entries.append(entry)
        self._by_key[entry_key] = entry
        return entry

    def _resolve_base_name(self, producer: object | None, *, warn: bool) -> tuple[str | None, bool]:
        if producer is None:
            return None, False
        trace_name = getattr(producer, "trace_name", None)
        if trace_name:
            return str(trace_name), False
        path = self._module_paths.get(id(producer))
        if path:
            return path, False
        return self._assign_fallback(producer, warn=warn), True

    def _assign_fallback(self, producer: object, *, warn: bool) -> str:
        producer_id = id(producer)
        assigned = self._fallback_assigned.get(producer_id)
        if assigned is None:
            class_name = type(producer).__name__
            index = self._fallback_class_counts.get(class_name, 0)
            self._fallback_class_counts[class_name] = index + 1
            assigned = f"{class_name}{index}"
            self._fallback_assigned[producer_id] = assigned
        if warn and producer_id not in self._warned_fallback:
            self._warned_fallback.add(producer_id)
            warnings.warn(
                f"Trace: producer {type(producer).__name__} has no trace_name "
                f"or module path; using fallback name {assigned}.",
                TraceWarning,
                stacklevel=4,
            )
        return assigned

    def _disambiguate(self, entry_key: str) -> str:
        if entry_key not in self._by_key:
            return entry_key
        suffix = 1
        while f"{entry_key}#{suffix}" in self._by_key:
            suffix += 1
        disambiguated = f"{entry_key}#{suffix}"
        warnings.warn(
            f"Trace: duplicate trace key {entry_key!r}; recording as {disambiguated!r}.",
            TraceWarning,
            stacklevel=4,
        )
        return disambiguated


def trace_value(
    value: Any,
    *,
    slot: str = "output",
    producer: object | None = None,
    key: str | None = None,
    semantic_type: str | None = None,
) -> None:
    """Record `value` to the active trace, or do nothing if tracing is inactive."""

    trace = _active_trace.get()
    if trace is None:
        return
    trace.record(
        value=value,
        slot=slot,
        producer=producer,
        key=key,
        semantic_type=semantic_type,
    )


__all__ = [
    "ParticleTensor",
    "PermutableTraceValue",
    "Trace",
    "TraceEntry",
    "TraceWarning",
    "trace_value",
]
