"""Optimizer construction from Hydra factories or configs."""

from __future__ import annotations

from typing import Any, Iterable

from hydra.utils import instantiate
from omegaconf import DictConfig

from spenn.dependencies import require_torch

torch = require_torch(feature="optimizer construction")


def make_optimizer(factory_or_cfg: Any, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """Build an optimizer bound to ``params`` from a factory or config.

    Accepts the shapes produced by Hydra instantiation of an optimizer block:

    - a ``functools.partial`` (or any callable) from a ``_partial_: true``
      ``_target_`` block, called as ``factory(params)``;
    - an un-instantiated mapping/`DictConfig` carrying ``_target_``, which is
      instantiated (honoring ``_partial_``) and, if still callable, called with
      ``params``;
    - an already-constructed `torch.optim.Optimizer`, returned unchanged.

    The optimizer YAML never needs a ``params`` key.

    Parameters
    ----------
    factory_or_cfg : Any
        Optimizer factory, config, or instance.
    params : iterable of torch.nn.Parameter
        Parameters the optimizer should manage.

    Returns
    -------
    torch.optim.Optimizer
        Constructed optimizer.

    Raises
    ------
    TypeError
        If ``factory_or_cfg`` cannot be turned into an optimizer.
    """

    if isinstance(factory_or_cfg, torch.optim.Optimizer):
        return factory_or_cfg

    candidate = factory_or_cfg
    if isinstance(candidate, (DictConfig, dict)) and "_target_" in candidate:
        candidate = instantiate(candidate)

    if isinstance(candidate, torch.optim.Optimizer):
        return candidate
    if callable(candidate):
        optimizer = candidate(params)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer factory must return a torch.optim.Optimizer, got {type(optimizer)!r}")
        return optimizer

    raise TypeError(
        "make_optimizer expects a torch.optim.Optimizer, a callable factory, or a config with "
        f"'_target_'; got {type(factory_or_cfg)!r}"
    )


__all__ = ["make_optimizer"]
