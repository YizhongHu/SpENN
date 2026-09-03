"""Tests for the PBS/PALS -> SLURM translation used for DeepQMC multi-host DDP.

The wrapper's entire job is six environment assignments, so those assignments are
what must be tested. It ends in ``exec "$@"``, which makes this straightforward:
invoke it with ``env`` as the command and read back what it set.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parent / "multihost_launch.sh"


def _run(tmp_path: Path, nodes: list[str], **env: str) -> dict[str, str]:
    """Invoke the wrapper with a synthetic PBS/PALS environment; return its env."""
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("\n".join(nodes) + "\n", encoding="utf-8")
    base = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PBS_NODEFILE": str(nodefile),
        "PBS_JOBID": "7587171.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov",
    }
    base.update(env)
    proc = subprocess.run(
        ["bash", str(WRAPPER), "env"], capture_output=True, text=True, env=base
    )
    assert proc.returncode == 0, proc.stderr
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_job_id_is_the_integer_prefix(tmp_path: Path) -> None:
    """jax derives its port as SLURM_JOB_ID % 4096, so a bare integer is required.

    A PBS job id carries a hostname suffix; passing it through unchanged would make
    the int() conversion inside jax raise.
    """

    env = _run(tmp_path, ["hostA"])
    assert env["SLURM_JOB_ID"] == "7587171"


def test_nodelist_is_sorted_deduplicated_and_comma_joined(tmp_path: Path) -> None:
    """Every rank must derive the SAME coordinator, which jax takes as the FIRST host.

    PBS_NODEFILE repeats a hostname per slot and its order is not guaranteed, so
    identical sorting on every rank is what makes the choice agree.
    """

    env = _run(tmp_path, ["hostB", "hostA", "hostB", "hostA"])
    assert env["SLURM_STEP_NODELIST"] == "hostA,hostB"


def test_multi_rank_values_come_from_pals(tmp_path: Path) -> None:
    env = _run(tmp_path, ["hostA", "hostB"], PMI_SIZE="2", PMI_RANK="1",
               PALS_LOCAL_RANKID="0")
    assert env["SLURM_NTASKS"] == "2"
    assert env["SLURM_PROCID"] == "1"
    assert env["SLURM_LOCALID"] == "0"


def test_single_rank_must_not_claim_multiple_tasks(tmp_path: Path) -> None:
    """The dangerous direction: a false NTASKS>1 on a single-rank run.

    DeepQMC calls jax.distributed.initialize() only when SLURM_NTASKS > 1. If a
    lone process reported 2, it would attempt a coordinator handshake against a peer
    that will never arrive and hang until the initialization timeout rather than
    running.
    """

    env = _run(tmp_path, ["hostA"])
    assert env["SLURM_NTASKS"] == "1"
    assert env["SLURM_PROCID"] == "0"


def test_visible_devices_default_and_override(tmp_path: Path) -> None:
    """initialize() asserts CUDA_VISIBLE_DEVICES is set and parses it as device ids."""

    assert _run(tmp_path, ["hostA"])["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert _run(tmp_path, ["hostA"], MH_VISIBLE="0,1")["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_refuses_to_run_outside_a_pbs_allocation(tmp_path: Path) -> None:
    """Outside PBS the mapping would be silently wrong, so it must fail loudly."""

    proc = subprocess.run(
        ["bash", str(WRAPPER), "env"], capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    assert proc.returncode != 0
    assert "PBS_NODEFILE" in proc.stderr
