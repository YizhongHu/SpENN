"""Fail-loud lazy-parameter materialization in the Train runner."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from spenn.runner import Train


class _LazyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.LazyLinear(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _ExampleSampler:
    def example_batch(self) -> torch.Tensor:
        return torch.zeros(2, 3, dtype=torch.float32)


class _NoExampleSampler:
    pass


def _train(model, sampler, construction_seed=0) -> Train:
    return Train(
        model=model,
        sampler=sampler,
        hamiltonian_terms=[],
        optimizer=None,
        trainer=None,
        construction_seed=construction_seed,
    )


def test_materialize_raises_when_sampler_has_no_example_batch() -> None:
    train = _train(_LazyModel(), _NoExampleSampler())
    with pytest.raises(RuntimeError, match="example_batch"):
        train._materialize_model()


def test_materialize_initializes_all_lazy_parameters() -> None:
    model = _LazyModel()
    assert any(isinstance(p, UninitializedParameter) for p in model.parameters())

    _train(model, _ExampleSampler())._materialize_model()

    assert not any(isinstance(p, UninitializedParameter) for p in model.parameters())


def test_materialize_does_not_leak_construction_seed_into_global_rng() -> None:
    before = torch.get_rng_state()
    _train(_LazyModel(), _ExampleSampler(), construction_seed=12345)._materialize_model()
    after = torch.get_rng_state()

    assert torch.equal(before, after)


def test_materialize_is_noop_without_lazy_parameters() -> None:
    eager = nn.Linear(3, 2)
    # No example_batch needed: there is nothing to materialize, so it must not raise.
    _train(eager, _NoExampleSampler())._materialize_model()
