"""Tests for deterministic non-identity permutation selection."""

from __future__ import annotations

import math

import pytest

from spenn.data.permutation import (
    Permutation,
    apply_particle_permutation,
    count_nonidentity_permutations,
    select_nonidentity_permutations,
)


def _images(permutations) -> list[tuple[int, ...]]:
    return [p.image for p in permutations]


def test_count_excludes_identity() -> None:
    assert count_nonidentity_permutations(0) == 0
    assert count_nonidentity_permutations(1) == 0
    assert count_nonidentity_permutations(2) == 1
    assert count_nonidentity_permutations(3) == 5
    assert count_nonidentity_permutations(4) == math.factorial(4) - 1


def test_fraction_and_max_count_determine_count() -> None:
    # n=4 -> 23 available; ceil(0.1 * 23) = 3, capped by max_count=8 -> 3.
    selected = select_nonidentity_permutations(n_particles=4, fraction=0.1, max_count=8, seed=0)
    assert len(selected) == 3


def test_max_count_caps_fraction_result() -> None:
    selected = select_nonidentity_permutations(n_particles=4, fraction=1.0, max_count=5, seed=0)
    assert len(selected) == 5


def test_fraction_one_high_max_returns_all_nonidentity() -> None:
    selected = select_nonidentity_permutations(n_particles=3, fraction=1.0, max_count=100, seed=0)
    images = _images(selected)
    assert len(images) == 5
    assert tuple(range(3)) not in images
    assert len(set(images)) == 5  # distinct


def test_same_seed_and_step_gives_same_subset() -> None:
    a = select_nonidentity_permutations(n_particles=5, fraction=0.2, max_count=4, seed=7, step=3)
    b = select_nonidentity_permutations(n_particles=5, fraction=0.2, max_count=4, seed=7, step=3)
    assert _images(a) == _images(b)


def test_different_step_gives_different_subset_when_possible() -> None:
    a = select_nonidentity_permutations(n_particles=5, fraction=0.2, max_count=4, seed=7, step=1)
    b = select_nonidentity_permutations(n_particles=5, fraction=0.2, max_count=4, seed=7, step=2)
    assert _images(a) != _images(b)


def test_different_seed_changes_subset() -> None:
    a = select_nonidentity_permutations(n_particles=5, fraction=0.2, max_count=4, seed=1, step=0)
    b = select_nonidentity_permutations(n_particles=5, fraction=0.2, max_count=4, seed=2, step=0)
    assert _images(a) != _images(b)


def test_selected_permutations_are_distinct_and_exclude_identity() -> None:
    selected = select_nonidentity_permutations(n_particles=4, fraction=1.0, max_count=10, seed=3, step=5)
    images = _images(selected)
    assert len(images) == len(set(images))
    assert tuple(range(4)) not in images


def test_zero_fraction_or_max_count_selects_nothing() -> None:
    assert select_nonidentity_permutations(n_particles=4, fraction=0.0, max_count=8, seed=0) == []
    assert select_nonidentity_permutations(n_particles=4, fraction=1.0, max_count=0, seed=0) == []


def test_invalid_fraction_and_max_count_raise() -> None:
    with pytest.raises(ValueError, match="fraction"):
        select_nonidentity_permutations(n_particles=4, fraction=1.5, max_count=8)
    with pytest.raises(ValueError, match="max_count"):
        select_nonidentity_permutations(n_particles=4, fraction=1.0, max_count=-1)


def test_apply_particle_permutation_dispatches_on_permute() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.seen = None

        def permute(self, permutation: Permutation) -> str:
            self.seen = permutation
            return "permuted"

    recorder = Recorder()
    assert apply_particle_permutation(recorder, Permutation((1, 0))) == "permuted"
    assert recorder.seen == Permutation((1, 0))


def test_apply_particle_permutation_rejects_non_permutable() -> None:
    with pytest.raises(TypeError, match="not particle-permutable"):
        apply_particle_permutation(object(), Permutation((1, 0)))
