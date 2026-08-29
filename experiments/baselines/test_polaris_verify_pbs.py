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
