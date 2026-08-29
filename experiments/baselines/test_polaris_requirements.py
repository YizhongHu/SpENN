"""Assertions over the pinned Polaris DeepQMC requirements.

The file had no test at all: an independent verifier changed a pin
(``tqdm==4.70.0`` to ``4.69.0``) and the entire suite stayed green.

These pins are not arbitrary. Every existing DeepQMC record was produced under
them, and the whole argument for pinning rather than resolving is that a
different jax minor would make Polaris rows incomparable to Cannon rows for
reasons unrelated to physics. A silent edit to any pin defeats that.

Scope: these assert the file's CONTENT, not that the environment installs or
that the versions are correct for the platform. Installability was checked
separately by resolving against ``x86_64-manylinux_2_28``; that cannot run in
this suite because the CUDA wheels are Linux-only.
"""

from __future__ import annotations

from pathlib import Path

REQUIREMENTS = Path(__file__).parent / "polaris_deepqmc_requirements.txt"

#: The versions the comparator rows were produced under. Changing any of these
#: changes what a Polaris row is comparable to, so they are spelled out here
#: rather than read from the file the test is checking.
LOAD_BEARING_PINS = {
    "jax": "0.8.3",
    "jaxlib": "0.8.3",
    "jax-cuda12-pjrt": "0.8.3",
    "jax-cuda12-plugin": "0.8.3",
    "hydra-core": "1.3.5",
    "omegaconf": "2.3.1",
    "kfac-jax": "0.0.8",
    "folx": "0.2.30",
    "uncertainties": "3.2.3",
    "h5py": "3.16.0",
    "pyscf": "2.14.0",
    "numpy": "2.5.2",
    "tqdm": "4.70.0",
}

#: Absent from the Cannon reference and therefore NOT required, despite
#: appearing among DeepQMC's optional extras. Their absence is a finding.
MUST_BE_ABSENT = ("e3nn-jax", "tensorboard")


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def test_the_load_bearing_pins_are_exactly_the_reference_versions() -> None:
    pins = _pins()
    for name, version in LOAD_BEARING_PINS.items():
        assert name in pins, f"{name} is missing from the pin file"
        assert pins[name] == version, f"{name} is pinned to {pins[name]}, expected {version}"


def test_jax_is_below_the_version_deepqmc_excludes() -> None:
    """DeepQMC declares jax<0.9.0. jax 0.9.2 is what made a new venv necessary."""
    major, minor, _ = _pins()["jax"].split(".")
    assert (int(major), int(minor)) < (0, 9)


def test_the_jax_family_versions_agree() -> None:
    """A mismatched jaxlib or CUDA plugin is a silent runtime hazard."""
    pins = _pins()
    family = {pins[n] for n in ("jax", "jaxlib", "jax-cuda12-pjrt", "jax-cuda12-plugin")}
    assert len(family) == 1, f"jax family versions disagree: {family}"


def test_optional_extras_absent_from_the_reference_stay_absent() -> None:
    pins = _pins()
    for name in MUST_BE_ABSENT:
        assert name not in pins, f"{name} is absent from the Cannon reference and not required"


def test_every_line_is_an_exact_pin_and_no_duplicates() -> None:
    """A range or an unpinned name would reintroduce resolver freedom."""
    lines = [
        line.strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in lines:
        assert "==" in line, f"not an exact pin: {line}"
        assert not any(op in line for op in (">=", "<=", ">", "<", "~=", ",")), line
    assert len(lines) == len(_pins()) == 56, (len(lines), len(_pins()))
