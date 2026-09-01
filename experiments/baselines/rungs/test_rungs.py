"""Tests for the cluster rung helpers.

These pin two defects that reached PRODUCTION, so they are regression tests
rather than coverage.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """Load a rung script as a module without executing its __main__ body."""
    src = (HERE / name).read_text(encoding="utf-8")
    # These are scripts, not modules: they read sys.argv at import time. Take
    # only the definitions ABOVE the first argv use, so the tables and helpers
    # are importable without invoking the script body.
    lines = src.splitlines(keepends=True)
    cut_line = next((i for i, l in enumerate(lines) if "sys.argv" in l), len(lines))
    cut = sum(len(l) for l in lines[:cut_line])
    module = type(sys)(name)
    module.__dict__["__file__"] = str(HERE / name)
    exec(compile(src[:cut], str(HERE / name), "exec"), module.__dict__)
    return module


def test_h2_pins_its_geometry_explicitly() -> None:
    """H2 must never accept diatomic.py's default bond length.

    `diatomic.py` defaults to 0.737164 angstrom; every reference row was produced
    at 1.4 bohr. Accepting the default is SILENT and worth ~2.1e-4 Ha, about six
    times the seed standard deviation, so it cannot be caught by any structural
    check -- only by the energy.
    """

    mod = _load("rung_makeplan.py")
    flags = mod.SYSTEMS["h2"]["config_flags"]("/fn")
    assert "--config.system.bond_length" in flags
    assert flags[flags.index("--config.system.bond_length") + 1] == "1.4"
    assert flags[flags.index("--config.system.units") + 1] == "bohr"
    assert "diatomic.py" in flags[1]


@pytest.mark.parametrize("system,atom", [("he", "He"), ("li", "Li"), ("be", "Be"), ("b", "B"), ("n", "N")])
def test_atoms_take_no_geometry_override(system: str, atom: str) -> None:
    """Atoms use atom.py and must not carry a molecular geometry flag."""

    mod = _load("rung_makeplan.py")
    flags = mod.SYSTEMS[system]["config_flags"]("/fn")
    assert "atom.py" in flags[1]
    assert flags[flags.index("--config.system.atom") + 1] == atom
    assert "--config.system.bond_length" not in flags
    assert "--config.system.molecule_name" not in flags


def test_every_system_is_distinguishable_by_its_flags() -> None:
    """No two systems may emit the same command flags.

    Production job 7579539 ran H2 while labelled He. Two near-identical builders
    existed and the wrong one was selected; a single parameterised table removes
    the choice, and this asserts the table itself cannot collapse.
    """

    mod = _load("rung_makeplan.py")
    rendered = {s: mod.SYSTEMS[s]["config_flags"]("/fn") for s in mod.SYSTEMS}
    assert len(set(rendered.values())) == len(rendered), rendered


def test_energy_bands_do_not_overlap() -> None:
    """The gate's wrong-system check is only meaningful if bands are disjoint."""

    mod = _load("rung_gate.py")
    bands = mod.ENERGY_BAND
    items = sorted(bands.items(), key=lambda kv: kv[1][0])
    for (a, (alo, ahi)), (b, (blo, bhi)) in zip(items, items[1:]):
        assert ahi < blo, f"bands for {a} and {b} overlap: {(alo, ahi)} vs {(blo, bhi)}"


def test_energy_bands_cover_each_system_reference() -> None:
    """Each band must contain the value that system actually produces.

    Values observed in real runs; a band that excluded them would fail every
    correct run, which is the opposite failure and just as bad.
    """

    observed = {"he": -2.9037, "h2": -1.1744, "li": -7.4780, "be": -14.6673, "b": -24.6539, "n": -54.5893}
    mod = _load("rung_gate.py")
    for system, value in observed.items():
        lo, hi = mod.ENERGY_BAND[system]
        assert lo <= value <= hi, f"{system}: {value} outside {(lo, hi)}"
