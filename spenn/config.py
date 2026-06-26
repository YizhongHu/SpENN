"""Custom OmegaConf resolvers for SpENN configs.

These resolvers let scalar run parameters and structured choice libraries drive
model construction without hand-writing one YAML file per variant. They are
registered idempotently; importing this module (which :mod:`spenn.run` does on
the normal run path) makes them available to ``OmegaConf.resolve`` and
``hydra.utils.instantiate``.

Resolvers
---------
``spenn.basis_feature_dim``
    Given an :class:`spenn.nn.ElectronBasis` config subtree, return the
    per-particle one-body feature width. This is how a model wires its
    embedding ``in_features`` to whichever basis the architecture choice
    selected, e.g. ``in_features: ${spenn.basis_feature_dim:${model.basis}}``.
"""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

_REGISTERED = False


def basis_feature_dim(basis_config: Any) -> int:
    """Return the one-body feature width of a configured electron basis.

    Parameters
    ----------
    basis_config : Any
        A basis config node (``DictConfig`` with a ``_target_`` resolving to an
        :class:`spenn.nn.ElectronBasis`) or an already-instantiated basis.

    Returns
    -------
    int
        ``basis.out_features`` for the configured basis.
    """

    # Import lazily so configs that never use the resolver do not require torch.
    from hydra.utils import instantiate

    basis = basis_config if hasattr(basis_config, "out_features") else instantiate(basis_config)
    out_features = getattr(basis, "out_features", None)
    if out_features is None:
        raise TypeError(
            "spenn.basis_feature_dim expects an ElectronBasis config or instance with "
            f"out_features, got {type(basis_config)!r}"
        )
    return int(out_features)


def register_resolvers() -> None:
    """Register SpENN OmegaConf resolvers (idempotent)."""

    global _REGISTERED
    if _REGISTERED:
        return
    OmegaConf.register_new_resolver("spenn.basis_feature_dim", basis_feature_dim, replace=True)
    _REGISTERED = True


register_resolvers()


__all__ = ["basis_feature_dim", "register_resolvers"]
