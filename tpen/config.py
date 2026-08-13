"""Custom OmegaConf resolvers for TPEN configs.

These resolvers let scalar run parameters and structured choice libraries drive
model construction without hand-writing one YAML file per variant. They are
registered idempotently; importing this module (which :mod:`tpen.run` does on
the normal run path) makes them available to ``OmegaConf.resolve`` and
``hydra.utils.instantiate``.

Resolvers
---------
``tpen.basis_feature_dim``
    Given an :class:`tpen.nn.ElectronBasis` config subtree, return the
    per-particle one-body feature width. This is how a model wires its
    embedding ``in_features`` to whichever basis the architecture choice
    selected, e.g. ``in_features: ${tpen.basis_feature_dim:${model.basis}}``.

    Also registered under the legacy name ``spenn.basis_feature_dim``; see
    :data:`LEGACY_BASIS_FEATURE_DIM_RESOLVER`.
"""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

_REGISTERED = False

BASIS_FEATURE_DIM_RESOLVER = "tpen.basis_feature_dim"

# Retained legacy resolver name, NOT a deprecation shim on its way out.
# MIG-TPEN-000 D9 freezes ``experiments/hooke/pair_stability_v3/configs/*`` as the
# provenance record of a completed study, and its collected tests preserve exactly
# one property: those configs still OmegaConf-RESOLVE (they fail later, at
# instantiation, on their stale ``spenn.*`` ``_target_`` values). Both
# ``pair_stability.yaml`` and ``pair_validation.yaml`` interpolate
# ``${spenn.basis_feature_dim:${model.basis}}``, so dropping this registration would
# move their failure from instantiate-time to resolve-time and break the one contract
# D9 exists to protect. Same category as
# ``tpen.checkpoint.manifest.LEGACY_CHECKPOINT_KIND``: a durable external identifier
# referenced by immutable records. Do not sweep it; see task-orchestrator note
# ``resolver-name-is-not-sweepable-2026-08-11`` on item 58347e8f.
LEGACY_BASIS_FEATURE_DIM_RESOLVER = "spenn.basis_feature_dim"


def basis_feature_dim(basis_config: Any) -> int:
    """Return the one-body feature width of a configured electron basis.

    Parameters
    ----------
    basis_config : Any
        A basis config node (``DictConfig`` with a ``_target_`` resolving to an
        :class:`tpen.nn.ElectronBasis`) or an already-instantiated basis.

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
            f"{BASIS_FEATURE_DIM_RESOLVER} expects an ElectronBasis config or instance with "
            f"out_features, got {type(basis_config)!r}"
        )
    return int(out_features)


def register_resolvers() -> None:
    """Register TPEN OmegaConf resolvers (idempotent)."""

    global _REGISTERED
    if _REGISTERED:
        return
    OmegaConf.register_new_resolver(BASIS_FEATURE_DIM_RESOLVER, basis_feature_dim, replace=True)
    OmegaConf.register_new_resolver(
        LEGACY_BASIS_FEATURE_DIM_RESOLVER, basis_feature_dim, replace=True
    )
    _REGISTERED = True


register_resolvers()


__all__ = [
    "BASIS_FEATURE_DIM_RESOLVER",
    "LEGACY_BASIS_FEATURE_DIM_RESOLVER",
    "basis_feature_dim",
    "register_resolvers",
]
