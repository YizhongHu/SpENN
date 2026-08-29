"""Behavioural tests for the Polaris verification job script.

Nothing tested this file before, and an independent verifier demonstrated the
cost: mutating the restart guard from ``[ -e "$GUARD" ]`` to ``[ ! -e "$GUARD" ]``
-- inverting the check entirely -- left the whole suite green.

These tests EXTRACT the guard block from the shipped ``.pbs`` file between its
delimiter comments and execute that text under ``bash``. They therefore run the
code that ships rather than a copy of it, so a mutation to the script changes
what the test runs. A test that re-implemented the guard would have passed under
the same mutation and proved nothing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PBS = Path(__file__).parent / "polaris_deepqmc_verify.pbs"
BEGIN = "# --- BEGIN RESTART GUARD"
END = "# --- END RESTART GUARD"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _guard_source() -> str:
    """Slice the shipped guard out of the real script."""
    text = PBS.read_text()
    start = text.index(BEGIN)
    end = text.index(END)
    block = text[start:end]
    assert 'if [ -e "$GUARD" ]' in block, "guard block not found; delimiters moved?"
    return block


def _run_guard(tmp_path: Path, *, preexisting: bool) -> subprocess.CompletedProcess[str]:
    out_root = tmp_path / "runs"
    out_root.mkdir()
    guard = out_root / ".deepqmc-verify-guard-9999"
    if preexisting:
        guard.write_text("2026-08-28T00:00:00+00:00\n")
    script = f"""
set -euo pipefail
OUT_ROOT={out_root}
JOB=9999
D="$OUT_ROOT/deepqmc-polaris-verify-probe-$JOB"
GUARD="$OUT_ROOT/.deepqmc-verify-guard-$JOB"
{_guard_source()}
echo GUARD_PASSED
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_the_pbs_file_still_has_its_delimiters() -> None:
    """If the delimiters are removed, every test below silently tests nothing."""
    text = PBS.read_text()
    assert text.count(BEGIN) == 1
    assert text.count(END) == 1
    assert text.index(BEGIN) < text.index(END)


def test_first_run_passes_the_guard_and_writes_the_marker(tmp_path: Path) -> None:
    result = _run_guard(tmp_path, preexisting=False)
    assert result.returncode == 0, result.stderr
    assert "GUARD_PASSED" in result.stdout
    assert (tmp_path / "runs" / ".deepqmc-verify-guard-9999").is_file()
    assert (tmp_path / "runs" / "deepqmc-polaris-verify-probe-9999" / "STARTED").is_file()


def test_a_second_run_of_the_same_job_id_is_refused(tmp_path: Path) -> None:
    """The requeue case. This program discards restarted stochastic runs, so a
    silent second start is a correctness bug, not an inconvenience."""
    result = _run_guard(tmp_path, preexisting=True)
    assert result.returncode == 9, (result.returncode, result.stdout, result.stderr)
    assert "REFUSING" in result.stdout
    assert "GUARD_PASSED" not in result.stdout


def test_the_guard_key_does_not_depend_on_mode_or_out_root() -> None:
    """The guard must key on JOB IDENTITY alone.

    It previously keyed on $D, which is derived from TPEN_DQMC_MODE and
    TPEN_DQMC_OUT -- both settable per submission -- so the same job id under a
    different MODE wrote elsewhere and found no marker. A guard whose scope is
    controlled by whoever is retrying is not a guard.
    """
    text = PBS.read_text()
    guard_line = re.search(r'^GUARD=.*$', text, re.M)
    assert guard_line is not None
    assert "$MODE" not in guard_line.group(0), guard_line.group(0)
    assert "$D" not in guard_line.group(0), guard_line.group(0)
    assert "$JOB" in guard_line.group(0)


# --- Whole-script properties the extracted-block test cannot see -------------
# An independent verifier's three mutants ALL survived here: flipping the row's
# `set +e` to `set -e`, changing a requirements pin, and flipping
# XLA_PYTHON_CLIENT_PREALLOCATE from false to true. Each is load-bearing and none
# was executed by any test, because the block test runs ONE block in isolation
# and nothing else reads the script at all.
# These are static assertions over the shipped file. They are weaker than
# execution -- they check that the text says the right thing, not that the
# program behaves -- and that limit is the reason they are stated as such rather
# than presented as coverage of the script.


def _line_of(pattern: str) -> int:
    text = PBS.read_text()
    match = re.search(pattern, text, re.M)
    assert match is not None, f"pattern not found in the shipped script: {pattern}"
    return text[: match.start()].count("\n") + 1


def test_traps_are_installed_before_the_guard_and_the_marker() -> None:
    """Installation ORDER is a whole-program property.

    The traps were previously installed ~70 lines below the guard and the marker
    write. A TERM or INT in that window took the shell default and produced no
    KILLED record -- the ambiguous absence the trap exists to prevent, in the
    window where a scheduler is most likely to kill a job it has just started.
    """
    trap_line = _line_of(r"^trap 'on_signal TERM 15' TERM")
    guard_line = _line_of(r'^if \[ -e "\$GUARD" \]')
    marker_line = _line_of(r'^date -Iseconds > "\$GUARD"')
    assert trap_line < guard_line, (trap_line, guard_line)
    assert trap_line < marker_line, (trap_line, marker_line)


def test_preallocation_is_disabled_before_any_python_runs() -> None:
    """XLA_PYTHON_CLIENT_PREALLOCATE=false is load-bearing for E6.

    With preallocation on, JAX takes ~75% of the card and every memory reading is
    identical regardless of workload -- so flipping this to true would not fail
    anything, it would silently make the measurement meaningless. A verifier's
    mutant flipping it survived the entire suite.
    """
    text = PBS.read_text()
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in text
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=true" not in text
    export_line = _line_of(r"^export XLA_PYTHON_CLIENT_PREALLOCATE=false")
    first_python = _line_of(r'"\$VENV/bin/python"')
    assert export_line < first_python, (export_line, first_python)


def test_the_row_status_is_captured_without_errexit() -> None:
    """The row's own exit code must survive a non-zero return.

    Under `set -e` a failing row would abort the script before RC is captured,
    so the guards keyed on RC could never run and the row's status would never
    be reported. A verifier's mutant flipping this survived.
    """
    text = PBS.read_text()
    row = text.index('"$VENV/bin/deepqmc" hydra.run.dir="$RD"')
    before = text[:row]
    assert before.rstrip().endswith("set +e"), (
        "the training row must be preceded by `set +e` so RC can be captured"
    )
    after = text[row:]
    assert after.index("RC=$?") < after.index("set -e")
