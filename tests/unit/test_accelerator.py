"""Contract tests for accelerator backend resolution.

These defend the property that makes the module exist: no call site may name a
concrete backend, because on a non-CUDA build ``torch.cuda`` still exists and
reports unavailable rather than raising, so a hardcoded CUDA path degrades to
CPU silently.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tpen.accelerator import (  # noqa: E402
    canonical_device,
    current_accelerator_type,
    device_module,
    seed_all,
    synchronize,
)


def test_device_module_resolves_backend_per_device_type() -> None:
    # The whole point: one call maps any device type to its own backend module,
    # so no caller has to branch on cuda/xpu/cpu.
    assert device_module("cpu") is torch.cpu
    assert device_module("cuda") is torch.cuda
    assert device_module("xpu") is torch.xpu


def test_device_module_accepts_torch_device_and_indexed_device() -> None:
    assert device_module(torch.device("cpu")) is torch.cpu
    assert device_module("cuda:3") is torch.cuda


def test_device_module_none_follows_the_active_accelerator() -> None:
    # None must agree with the reported accelerator type, whatever this host is.
    assert device_module() is torch.get_device_module(current_accelerator_type())


def test_current_accelerator_type_is_a_real_device_type() -> None:
    device_type = current_accelerator_type()
    assert device_type in {"cpu", "cuda", "xpu", "mps", "hpu"}
    # Must be usable as a device type, not a decorated string.
    assert torch.device(device_type).type == device_type


def test_canonical_device_leaves_cpu_index_free() -> None:
    # Tensors report CPU index-free, so adding an index would break comparison.
    assert canonical_device("cpu") == torch.device("cpu")
    assert canonical_device(torch.device("cpu")) == torch.device("cpu")


def test_canonical_device_preserves_an_explicit_index() -> None:
    assert canonical_device("cuda:2") == torch.device("cuda", 2)
    assert canonical_device("xpu:1") == torch.device("xpu", 1)


def test_canonical_device_is_idempotent() -> None:
    once = canonical_device("cpu")
    assert canonical_device(once) == once


def test_canonical_device_passes_through_a_device_without_an_accelerator_module() -> None:
    # `meta` is a valid device type with no registered module, so there is no
    # current device to resolve. It must pass through rather than raising, or
    # MetropolisSampler's fail-loud device-mismatch check would surface a torch
    # internal error instead of its own message.
    assert canonical_device("meta") == torch.device("meta")
    assert canonical_device(torch.device("meta")) == torch.device("meta")


def test_canonical_device_of_unavailable_accelerator_stays_index_free() -> None:
    # Guards the CPU-only/CI path: an index must never be invented for a device
    # that is not present, because that would fabricate a comparison mismatch.
    absent = "xpu" if not torch.xpu.is_available() else "cuda"
    if torch.get_device_module(absent).is_available():
        pytest.skip("needs an accelerator that is absent on this host")
    assert canonical_device(absent) == torch.device(absent)


def test_canonical_device_indexes_an_available_accelerator() -> None:
    device_type = current_accelerator_type()
    module = torch.get_device_module(device_type)
    if device_type == "cpu" or not module.is_available():
        pytest.skip("needs a live accelerator")
    resolved = canonical_device(device_type)
    assert resolved.type == device_type
    assert resolved.index == module.current_device()


def test_canonical_device_rejects_an_unknown_device_type() -> None:
    with pytest.raises(RuntimeError):
        canonical_device("definitely_not_a_backend")


def test_synchronize_is_safe_without_an_accelerator() -> None:
    # Must not raise on a CPU-only host; timing callbacks call this per event.
    synchronize("cpu")
    synchronize()


def test_seed_all_is_reproducible_and_never_raises() -> None:
    # Run-level global accelerator seeding must work on every backend, including
    # hosts where the accelerator module exposes no seeding at all.
    seed_all(1234)
    seed_all(1234, feature="test seeding")

    device_type = current_accelerator_type()
    module = torch.get_device_module(device_type)
    if device_type == "cpu" or not module.is_available():
        pytest.skip("needs a live accelerator to observe seeded state")
    seed_all(4321)
    first = torch.randn(8, device=device_type)
    seed_all(4321)
    second = torch.randn(8, device=device_type)
    assert torch.equal(first, second)


def test_seed_all_does_not_disturb_component_generators() -> None:
    # ADR-013: run-level global seeding must not reach a component-owned
    # generator, or a resumed Markov chain would silently change.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    expected = torch.randn(4, generator=generator, dtype=torch.float64)

    generator.manual_seed(7)
    seed_all(999999)
    observed = torch.randn(4, generator=generator, dtype=torch.float64)
    assert torch.equal(expected, observed)
