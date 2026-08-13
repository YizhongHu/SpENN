"""Tests for the OmegaConf resolver names TPEN publishes.

A resolver name is a config-file-facing string, so nothing in Python fails when one
disappears -- the config simply stops resolving. That makes these registrations
exactly the kind of surface that needs an explicit pin rather than incidental
coverage through the configs that happen to use them.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from tpen.config import (
    BASIS_FEATURE_DIM_RESOLVER,
    LEGACY_BASIS_FEATURE_DIM_RESOLVER,
    register_resolvers,
)


def _hooke_basis_config() -> dict[str, object]:
    """Return a small basis config whose one-body width is known."""

    return {
        "_target_": "tpen.nn.HookeOrbitalBasis",
        "omega": 0.5,
        "spatial_dim": 3,
        "basis_semantics": "product_v2",
        "truncation": "total_shell",
        "max_total_shell": 1,
        "include_gaussian_factor": False,
        "include_spin": False,
    }


def _resolved_width(resolver_name: str) -> int:
    """Resolve ``in_features`` through ``resolver_name`` the way a config does."""

    register_resolvers()
    cfg = OmegaConf.create(
        {
            "basis": _hooke_basis_config(),
            "in_features": f"${{{resolver_name}:${{basis}}}}",
        }
    )
    return int(cfg.in_features)


def test_canonical_resolver_name_is_the_tpen_spelling() -> None:
    """New and current configs interpolate the ``tpen.`` spelling."""

    assert BASIS_FEATURE_DIM_RESOLVER == "tpen.basis_feature_dim"
    assert _resolved_width(BASIS_FEATURE_DIM_RESOLVER) > 0


def test_legacy_resolver_name_still_resolves_for_the_frozen_archive() -> None:
    """The ``spenn.`` spelling is RETAINED, not deprecated -- MIG-TPEN-000 D9.

    ``experiments/hooke/pair_stability_v3/configs/pair_stability.yaml`` and
    ``pair_validation.yaml`` interpolate ``${spenn.basis_feature_dim:${model.basis}}``.
    D9 freezes those configs as the provenance record of a completed study and
    preserves exactly one property: they still OmegaConf-RESOLVE (they fail later, at
    instantiation, on their stale ``spenn.*`` ``_target_`` values). Dropping this
    registration would move that failure to resolve time and break the one contract D9
    exists to protect, so the alias is a deliberate keep. See task-orchestrator note
    ``resolver-name-is-not-sweepable-2026-08-11``.
    """

    assert LEGACY_BASIS_FEATURE_DIM_RESOLVER == "spenn.basis_feature_dim"
    assert _resolved_width(LEGACY_BASIS_FEATURE_DIM_RESOLVER) == _resolved_width(
        BASIS_FEATURE_DIM_RESOLVER
    )
