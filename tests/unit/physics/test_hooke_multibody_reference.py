"""Hooke multibody independent baseline reference tests."""

from __future__ import annotations

from math import pi, sqrt

import pytest

from experiments.hooke_multibody.reference import (
    gaussian_hartree_energy_components,
    gaussian_hartree_reference,
    gaussian_pair_distance_density_rows,
    gaussian_radial_density_rows,
)


def test_gaussian_hartree_energy_components_match_closed_form_formula() -> None:
    n_electrons = 3
    omega = 0.5
    alpha = 0.2

    components = gaussian_hartree_energy_components(
        n_electrons=n_electrons,
        harmonic_omega=omega,
        alpha=alpha,
    )

    pair_count = 3.0
    expected_kinetic = 0.5 * n_electrons * 3 * alpha
    expected_harmonic = n_electrons * 3 * omega**2 / (8.0 * alpha)
    expected_coulomb = pair_count * 2.0 * sqrt(alpha / pi)
    assert components["kinetic_energy"] == expected_kinetic
    assert components["harmonic_energy"] == expected_harmonic
    assert components["coulomb_energy"] == expected_coulomb
    assert components["total_energy"] == expected_kinetic + expected_harmonic + expected_coulomb


def test_gaussian_hartree_reference_optimizes_alpha_locally() -> None:
    reference = gaussian_hartree_reference(
        n_electrons=3,
        n_up=2,
        n_down=1,
        harmonic_omega=0.5,
    )

    low_energy = gaussian_hartree_energy_components(
        n_electrons=3,
        harmonic_omega=0.5,
        alpha=0.5 * reference.alpha,
    )["total_energy"]
    high_energy = gaussian_hartree_energy_components(
        n_electrons=3,
        harmonic_omega=0.5,
        alpha=2.0 * reference.alpha,
    )["total_energy"]
    assert reference.total_energy < low_energy
    assert reference.total_energy < high_energy
    assert reference.as_row()["baseline_method"] == "gaussian_hartree_variational"
    assert reference.as_row()["baseline_high_accuracy"] is False
    assert reference.as_row()["spatial_dim"] == 3


def test_gaussian_hartree_reference_recovers_one_electron_oscillator_limit() -> None:
    omega = 0.7

    reference = gaussian_hartree_reference(
        n_electrons=1,
        n_up=1,
        n_down=0,
        harmonic_omega=omega,
    )

    assert abs(reference.alpha - omega / 2.0) < 1.0e-12
    assert abs(reference.total_energy - 1.5 * omega) < 1.0e-12
    assert reference.coulomb_energy == 0.0


def test_gaussian_density_rows_are_normalized_on_large_grid() -> None:
    alpha = 0.25
    radial_rows = gaussian_radial_density_rows(alpha=alpha, bins=512, r_max=10.0)
    pair_rows = gaussian_pair_distance_density_rows(alpha=alpha, bins=512, r_max=12.0)

    radial_integral = _density_integral(radial_rows)
    pair_integral = _density_integral(pair_rows)
    assert abs(radial_integral - 1.0) < 1.0e-5
    assert abs(pair_integral - 1.0) < 1.0e-5
    assert all(row["probability_density"] >= 0.0 for row in radial_rows)
    assert all(row["probability_density"] >= 0.0 for row in pair_rows)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_electrons": 0, "n_up": 0, "n_down": 0, "harmonic_omega": 0.5}, "n_electrons"),
        ({"n_electrons": 3, "n_up": 3, "n_down": 1, "harmonic_omega": 0.5}, r"n_up \+ n_down"),
        ({"n_electrons": 3, "n_up": 2, "n_down": 1, "harmonic_omega": -0.5}, "harmonic_omega"),
        ({"n_electrons": 3, "n_up": 2, "n_down": 1, "harmonic_omega": 0.5, "spatial_dim": 2}, "spatial_dim"),
        ({"n_electrons": 3, "n_up": 2, "n_down": 1, "harmonic_omega": 0.5, "alpha": 0.0}, "alpha"),
    ],
)
def test_gaussian_hartree_reference_rejects_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gaussian_hartree_reference(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0, "bins": 8, "r_max": 4.0}, "alpha"),
        ({"alpha": 0.25, "bins": 0, "r_max": 4.0}, "bins"),
        ({"alpha": 0.25, "bins": 8, "r_max": 0.0}, "r_max"),
    ],
)
def test_gaussian_density_rows_reject_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gaussian_radial_density_rows(**kwargs)


def _density_integral(rows: list[dict[str, float]]) -> float:
    return sum((row["bin_right"] - row["bin_left"]) * row["probability_density"] for row in rows)
