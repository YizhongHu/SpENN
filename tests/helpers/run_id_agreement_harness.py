"""Bounded CPU/Gloo subprocess launcher for run-identity agreement tests.

WHY THIS IS A SECOND HARNESS, not a reuse of
:func:`tests.helpers.ddp_subprocess_harness.run_gloo_subprocess_group`. That
function hardcodes its own worker module
(``tests.helpers.ddp_worker_entrypoint``) and that worker's synthetic
fault-injection step sequence; it exposes no hook for running a different body,
so no amount of argument passing lets it exercise TPEN's run setup. Its safety
DISCIPLINE is what carries over, and is reproduced here deliberately: genuine OS
subprocesses in their own session, a fresh per-invocation rendezvous file that is
never reused, one shared deadline rather than N stacked per-process timeouts, and
process-GROUP kills on the way out. The two nested bounds here
(``process_group_timeout`` < ``watchdog_timeout``) sit inside the caller's Slurm
wall clock as the third.

Distinct from that module, and the reason a shared harness would be the wrong
shape anyway: these tests need a per-RANK configured value (one rank null while
its peers carry explicit ids), and they must collect a receipt from a rank that
REFUSED, since "every rank raised" is the assertion.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tests.helpers.run_id_agreement_worker import NULL_RUN_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAP_WAIT_SECONDS = 5.0

#: Inner bound: caps any single collective inside a child.
PROCESS_GROUP_TIMEOUT_SECONDS = 20.0
#: Outer bound: caps the whole invocation. Must exceed the inner bound so a
#: collective-level fault resolves itself before the group is force-killed.
WATCHDOG_TIMEOUT_SECONDS = 90.0


@dataclass(frozen=True)
class AgreementResult:
    """Outcome of one harness invocation.

    Parameters
    ----------
    receipts : tuple of dict or None
        One slot per rank, in rank order. ``None`` means that rank never wrote a
        receipt -- it died or hung before reaching the write. Kept distinct from
        a receipt carrying an ``error`` key, which means the rank refused and
        reported the refusal, because the negative tests must be able to tell a
        deliberate refusal from a crash.
    watchdog_fired : bool
        Whether the outer bound had to force-kill any child.
    exit_codes : tuple of int or None
        Per-rank exit status, ``None`` if never collected.
    invocation_dir : str
        Durable location of the per-rank logs and receipts.
    """

    receipts: tuple[dict | None, ...]
    watchdog_fired: bool
    exit_codes: tuple[int | None, ...]
    invocation_dir: str

    def run_ids(self) -> tuple[str | None, ...]:
        """Return each rank's reported run id, ``None`` where absent."""

        return tuple(
            None if receipt is None else receipt.get("run_id") for receipt in self.receipts
        )

    def errors(self) -> tuple[str | None, ...]:
        """Return each rank's reported error type, ``None`` where absent."""

        return tuple(
            None if receipt is None else receipt.get("error_type") for receipt in self.receipts
        )


def run_agreement_group(
    world_size: int,
    configured_run_ids: list[str | None],
    tmp_path: Path,
    *,
    mode: str = "resolve",
) -> AgreementResult:
    """Launch ``world_size`` fresh workers and collect one receipt per rank.

    Parameters
    ----------
    world_size : int
        Number of ranks to launch.
    configured_run_ids : list of str or None
        Per-rank configured ``run.run_id``; ``None`` means the config leaves it
        null. Length must equal ``world_size`` -- per-rank rather than one
        shared value precisely so a mixed launch is expressible.
    tmp_path : pathlib.Path
        Test-owned directory; a fresh uniquely named subdirectory is created
        inside it for this invocation.
    mode : {"resolve", "context"}
        ``resolve`` calls ``tpen.artifacts.resolve_run_id`` and touches no
        filesystem; ``context`` runs the whole of ``prepare_run_context`` so run
        directory convergence is observable.
    """

    if len(configured_run_ids) != world_size:
        raise ValueError(
            f"configured_run_ids has {len(configured_run_ids)} entries for world_size {world_size}"
        )

    invocation_dir = Path(tempfile.mkdtemp(dir=tmp_path, prefix="agreement-"))
    run_root = invocation_dir / "outputs"

    # torch's FileStore requires a path that does not yet exist, and a name that
    # was never used by an earlier invocation: a stale rendezvous file silently
    # joins two unrelated groups.
    rendezvous_fd, rendezvous_path = tempfile.mkstemp(dir=invocation_dir, prefix="rdzv-")
    os.close(rendezvous_fd)
    os.unlink(rendezvous_path)

    procs: list[subprocess.Popen] = []
    receipt_paths: list[Path] = []
    for rank in range(world_size):
        receipt_path = invocation_dir / f"receipt_{rank}.json"
        receipt_paths.append(receipt_path)
        configured = configured_run_ids[rank]
        argv = [
            sys.executable,
            "-m",
            "tests.helpers.run_id_agreement_worker",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--rendezvous-file",
            rendezvous_path,
            "--receipt-path",
            str(receipt_path),
            "--run-root",
            str(run_root),
            "--pg-timeout",
            str(PROCESS_GROUP_TIMEOUT_SECONDS),
            "--mode",
            mode,
            "--configured-run-id",
            NULL_RUN_ID if configured is None else configured,
        ]
        # Captured per rank so a PASSING negative test still leaves an
        # attributable diagnostic on disk; pytest's fd-level capture would
        # otherwise discard an inherited child's traceback entirely.
        log_path = invocation_dir / f"rank_{rank}.log"
        with open(log_path, "wb") as log_file:
            procs.append(
                subprocess.Popen(
                    argv,
                    cwd=str(_REPO_ROOT),
                    start_new_session=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            )

    # One shared deadline. Waiting on N children with independent per-process
    # timeouts would let the total wait stack to N * watchdog_timeout, blowing
    # the outer bound the watchdog exists to enforce.
    exit_codes: list[int | None] = [None] * world_size
    deadline = time.monotonic() + WATCHDOG_TIMEOUT_SECONDS
    while time.monotonic() < deadline and any(code is None for code in exit_codes):
        for index, proc in enumerate(procs):
            if exit_codes[index] is None:
                exit_codes[index] = proc.poll()
        if any(code is None for code in exit_codes):
            time.sleep(0.05)
    watchdog_fired = any(code is None for code in exit_codes)

    # Unconditional, not gated on watchdog_fired: a child that resolved its own
    # process-group timeout still exited inside the window, and killing the
    # group is how a survivor is prevented from outliving the test either way.
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for proc in procs:
        try:
            proc.wait(timeout=_REAP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass

    receipts: list[dict | None] = []
    for receipt_path in receipt_paths:
        if not receipt_path.exists():
            receipts.append(None)
            continue
        try:
            receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            receipts.append({"malformed": repr(exc)})

    return AgreementResult(
        receipts=tuple(receipts),
        watchdog_fired=watchdog_fired,
        exit_codes=tuple(exit_codes),
        invocation_dir=str(invocation_dir),
    )


def run_root_for(invocation_dir: str) -> Path:
    """Return the artifact root the workers of one invocation shared."""

    return Path(invocation_dir) / "outputs"


__all__ = [
    "AgreementResult",
    "PROCESS_GROUP_TIMEOUT_SECONDS",
    "WATCHDOG_TIMEOUT_SECONDS",
    "run_agreement_group",
    "run_root_for",
]
