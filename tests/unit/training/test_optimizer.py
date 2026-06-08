"""Unit tests for optimizer construction."""

from __future__ import annotations

import functools

import pytest
import torch
from omegaconf import OmegaConf

from spenn.training.optim import make_optimizer


def _model() -> torch.nn.Module:
    return torch.nn.Linear(2, 1)


def test_partial_config_builds_optimizer_bound_to_params() -> None:
    model = _model()
    cfg = OmegaConf.create({"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.01})

    optimizer = make_optimizer(cfg, model.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    managed = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert managed == {id(p) for p in model.parameters()}
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_yaml_needs_no_params_key() -> None:
    cfg = OmegaConf.create({"_target_": "torch.optim.SGD", "_partial_": True, "lr": 0.1})
    assert "params" not in cfg


def test_callable_factory_is_called_with_params() -> None:
    model = _model()
    factory = functools.partial(torch.optim.SGD, lr=0.1)

    optimizer = make_optimizer(factory, model.parameters())

    assert isinstance(optimizer, torch.optim.SGD)


def test_existing_optimizer_returned_unchanged() -> None:
    model = _model()
    existing = torch.optim.Adam(model.parameters(), lr=0.05)

    assert make_optimizer(existing, model.parameters()) is existing


def test_invalid_input_raises_type_error() -> None:
    model = _model()
    with pytest.raises(TypeError):
        make_optimizer(object(), model.parameters())
