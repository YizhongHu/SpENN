"""Tests for the typed `SamplerStats` record and its metric composition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tpen.sampling import SamplerStats

#: Geometry stand-in with one key of every shape the real summary produces.
GEOMETRY = {
    "n_walkers": 16,
    "n_electrons": 2,
    "spatial_dim": 3,
    "electron_distance_n_pairs": 16,
    "radius_mean": 1.25,
}


def _stats(**overrides) -> SamplerStats:
    fields = {
        "acceptance_rate": 0.75,
        "n_walkers": 16,
        "burn_in": 10,
        "n_steps": 5,
        "proposal_scale": 0.05,
        "geometry": GEOMETRY,
    }
    fields.update(overrides)
    return SamplerStats(**fields)


def test_as_metrics_round_trips_to_the_durable_key_set() -> None:
    metrics = _stats(seed=11).as_metrics()

    assert list(metrics) == [
        "acceptance_rate",
        "n_walkers",
        "burn_in",
        "n_steps",
        "proposal_scale",
        "seed",
        "n_electrons",
        "spatial_dim",
        "electron_distance_n_pairs",
        "radius_mean",
    ]
    assert metrics["acceptance_rate"] == pytest.approx(0.75)
    assert metrics["burn_in"] == 10
    assert metrics["seed"] == 11
    assert metrics["radius_mean"] == pytest.approx(1.25)


def test_unseeded_stats_report_no_seed_key() -> None:
    assert "seed" not in _stats().as_metrics()


def test_geometry_n_walkers_wins_over_the_named_field() -> None:
    # The pre-typed flat dict let geometry overwrite the named field, and both
    # metric views read that single key. Pin the precedence so a sampler that
    # sets n_walkers from configured capacity rather than the returned chain
    # cannot make the two views disagree.
    stats = _stats(n_walkers=4, geometry={**GEOMETRY, "n_walkers": 16})

    assert stats.reported_n_walkers == 16
    assert stats.as_metrics()["n_walkers"] == 16
    assert stats.as_check_metrics()["n_walkers"] == 16


def test_named_n_walkers_is_used_when_geometry_omits_it() -> None:
    stats = _stats(n_walkers=4, geometry={"radius_mean": 1.25})

    assert stats.reported_n_walkers == 4
    assert stats.as_metrics()["n_walkers"] == 4
    assert stats.as_check_metrics()["n_walkers"] == 4


def test_both_metric_views_agree_on_n_walkers() -> None:
    for stats in (
        _stats(),
        _stats(n_walkers=4, geometry={**GEOMETRY, "n_walkers": 16}),
        _stats(n_walkers=4, geometry={}),
    ):
        assert stats.as_metrics()["n_walkers"] == stats.as_check_metrics()["n_walkers"]


def test_as_check_metrics_is_the_fixed_check_subset() -> None:
    metrics = _stats(seed=11).as_check_metrics()

    assert metrics == {
        "acceptance_rate": pytest.approx(0.75),
        "n_walkers": 16,
        "n_steps": 5,
        "burn_in": 10,
    }


def test_integer_counts_stay_integers() -> None:
    metrics = _stats().as_metrics()

    for key in ("n_walkers", "burn_in", "n_steps", "n_electrons", "spatial_dim"):
        assert isinstance(metrics[key], int), key
    for key in ("acceptance_rate", "proposal_scale", "radius_mean"):
        assert isinstance(metrics[key], float), key


def test_geometry_is_copied_and_read_only() -> None:
    source = dict(GEOMETRY)
    stats = _stats(geometry=source)

    source["radius_mean"] = 99.0

    assert stats.geometry["radius_mean"] == pytest.approx(1.25)
    with pytest.raises(TypeError):
        stats.geometry["radius_mean"] = 99.0


def test_non_mapping_geometry_is_rejected() -> None:
    with pytest.raises(TypeError, match="geometry must be a Mapping"):
        _stats(geometry=[("radius_mean", 1.0)])


def test_record_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        _stats().acceptance_rate = 0.1


def test_record_is_hashable_despite_the_geometry_mapping() -> None:
    # geometry is stored as an unhashable mappingproxy, so it is excluded from
    # the generated __hash__; hashing must not raise.
    stats = _stats()

    assert hash(stats) == hash(_stats())
    assert len({stats, _stats()}) == 1


def test_geometry_is_excluded_from_the_hash() -> None:
    # geometry stays in __eq__ but out of __hash__, so records differing only in
    # geometry are unequal yet share a hash -- a legal collision, never the
    # reverse (equal records always hash equal).
    same_scalars = _stats(geometry={"radius_mean": 9.0})

    assert same_scalars != _stats()
    assert hash(same_scalars) == hash(_stats())
