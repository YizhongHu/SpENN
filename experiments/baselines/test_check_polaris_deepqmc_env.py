"""Tests for the DeepQMC-on-Polaris seed check.

The fixture in ``testdata/deepqmc_hydra_config.yaml`` is a byte copy of a real
``training/.hydra/config.yaml`` from Cannon run
``dqmc-he-default-200k-seedspread-42610526_1``, taken 2026-08-28. It is used
verbatim rather than reduced to a three-line stub because the property under
test is precisely the one a stub destroys: in the real file ``task:`` is line 1
and ``seed:`` is line 50, forty-nine lines apart, across a 283-line config with
four top-level blocks. A stub would pass whatever the reader wrote it to pass.

The ``env`` subcommand is not covered here. It requires jax, a GPU and a
DeepQMC checkout, none of which exist in this suite; its evidence is the PBS job
log instead.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from experiments.baselines.check_polaris_deepqmc_env import (
    HE_EXACT_HARTREE,
    HYDRA_CONFIG_RELPATH,
    EnvCheckError,
    check_seed,
    compare_energy,
    read_seed,
)

FIXTURE = Path(__file__).parent / "testdata" / "deepqmc_hydra_config.yaml"


def _run_dir(tmp_path: Path, config_text: str | None = None) -> Path:
    """Materialise a run directory holding a Hydra config at the real relpath."""
    run_dir = tmp_path / "run"
    target = run_dir / HYDRA_CONFIG_RELPATH
    target.parent.mkdir(parents=True)
    if config_text is None:
        shutil.copyfile(FIXTURE, target)
    else:
        target.write_text(config_text)
    return run_dir


def test_fixture_still_has_the_shape_that_makes_this_check_necessary() -> None:
    """Guard the fixture itself: if it is ever trimmed, the other tests weaken."""
    lines = FIXTURE.read_text().splitlines()
    assert lines[0] == "task:"
    seed_lines = [i for i, line in enumerate(lines, 1) if line.strip().startswith("seed:")]
    assert seed_lines == [50], seed_lines
    # Far enough below `task:` that a two-line-context grep misses it entirely,
    # which is the exact way this check gets skipped in practice.
    assert seed_lines[0] - 1 > 2
    top_level = [line for line in lines if line and not line[0].isspace()]
    assert top_level == ["task:", "logging:", "ansatz:", "hamil:"], top_level


def test_read_seed_reports_value_line_number_and_raw_line(tmp_path: Path) -> None:
    found = read_seed(_run_dir(tmp_path))
    assert found["seed"] == 1
    # The line number and raw line are what let a receipt quote the file.
    assert found["line_number"] == 50
    assert found["raw_line"] == "  seed: 1"


def test_check_seed_passes_when_the_override_took(tmp_path: Path) -> None:
    result = check_seed(_run_dir(tmp_path), expect_seed=1)
    assert result["ok"] is True
    assert result["expected"] == 1


def test_check_seed_raises_when_the_override_did_not_take(tmp_path: Path) -> None:
    """The whole point: a requested seed that never reached the config must fail.

    Without this the run trains happily on the default seed and the row looks
    legitimate.
    """
    with pytest.raises(EnvCheckError) as excinfo:
        check_seed(_run_dir(tmp_path), expect_seed=7)
    message = str(excinfo.value)
    assert "did not take" in message
    # Both numbers must appear, or the failure cannot be diagnosed from the log.
    assert "1" in message and "7" in message


def test_missing_config_is_an_error_not_a_silent_pass(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EnvCheckError, match="no Hydra config"):
        read_seed(empty)


def test_config_without_a_task_seed_key_is_an_error(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, "task:\n  steps: 200000\nansatz:\n  name: default\n")
    with pytest.raises(EnvCheckError, match="no task.seed key"):
        read_seed(run_dir)


def test_seed_outside_the_task_block_is_not_mistaken_for_task_seed(tmp_path: Path) -> None:
    """A sibling block's own `seed:` must not be reported as `task.seed`.

    An unbounded search for the first `seed:` line would quote the ansatz block
    here and still return a plausible-looking line number.
    """
    text = "ansatz:\n  seed: 99\ntask:\n  steps: 10\n  seed: 4\n"
    found = read_seed(_run_dir(tmp_path, text))
    assert found["seed"] == 4
    assert found["line_number"] == 5
    assert found["raw_line"] == "  seed: 4"


# --- A5 pre-registered comparison -------------------------------------------
# The comparator is Cannon run dqmc-he-39358341/default: 20000 steps, batch 4096,
# seed 0, code_commit edf373e7, on an A100-SXM4-80GB.
CANNON_HE_DEFAULT_20K = -2.9036863634347916
CANNON_HE_DEFAULT_20K_STDERR = 1.6026224325770225e-05


def test_a_matching_energy_is_sane() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 2e-5,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "sane"


def test_the_investigate_band_is_not_silently_absorbed_into_either_verdict() -> None:
    """A difference between 1e-4 and 1e-3 is a finding, not a pass and not a fail."""
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 5e-4,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "investigate"


def test_a_third_decimal_difference_is_broken() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 2e-3,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "broken"


def test_below_exact_is_broken_even_when_the_delta_is_tiny() -> None:
    """The disqualifier must outrank a small delta, not be averaged with it.

    An energy can sit very close to the Cannon row and still be variationally
    impossible. This program has produced below-exact energies four times from
    too-short tails, so this is the check that must not be conditional on delta.
    """
    below = HE_EXACT_HARTREE - 1e-3
    result = compare_energy(
        polaris_energy=below,
        polaris_stderr=1e-6,
        reference_energy=below,  # delta is exactly zero
    )
    assert result["abs_delta_hartree"] == 0.0
    assert result["verdict"] == "broken"
    assert any("variationally impossible" in r for r in result["reasons"])


def test_a_short_row_is_broken_regardless_of_energy() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        steps_observed=11000,
        steps_expected=20000,
    )
    assert result["verdict"] == "broken"
    assert any("short" in r for r in result["reasons"])


def test_non_finite_energy_is_broken() -> None:
    result = compare_energy(
        polaris_energy=float("nan"),
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "broken"


def test_verdict_is_derived_from_the_numbers_it_reports() -> None:
    """Guard against a label drifting from the value printed beside it."""
    for offset in (0.0, 5e-5, 2e-4, 5e-3):
        result = compare_energy(
            polaris_energy=CANNON_HE_DEFAULT_20K + offset,
            polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
            reference_energy=CANNON_HE_DEFAULT_20K,
        )
        magnitude = result["abs_delta_hartree"]
        expected = (
            "sane" if magnitude < 1e-4 else "investigate" if magnitude < 1e-3 else "broken"
        )
        assert result["verdict"] == expected, (offset, result)
