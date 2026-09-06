"""Serial and no-process-group behaviour of run-identity resolution.

The multi-rank half of the contract lives in
``tests/unit/test_run_id_rank_agreement.py``, which needs real subprocesses and
a Gloo build. Everything here runs in one torch-free-capable process: the serial
path that local and smoke runs take, and the refusal that closes the multi-rank
launch which never initialized a process group at all.
"""

from __future__ import annotations

import re
import sys

import pytest
from omegaconf import OmegaConf

import tpen.artifacts as artifacts
from tpen.artifacts import RunIdentityError, generate_run_id, resolve_run_id
from tpen.distributed import ExecutionTopology
from tpen.run import prepare_run_context

#: The historical id shape, pinned so the fix cannot quietly change it: the
#: random suffix is KEPT, because the defect was per-rank divergence rather than
#: suffix randomness.
_RUN_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_[A-Za-z0-9_.-]+_[0-9a-f]{6}$")


def _multi_rank_topology(global_size: int = 2) -> ExecutionTopology:
    """Return a launcher-supplied topology that declares several ranks."""

    return ExecutionTopology(
        global_rank=0,
        global_size=global_size,
        local_rank=0,
        local_size=global_size,
        node_rank=0,
        node_size=1,
        host="node-a",
        pid=42,
        device="cpu",
    )


def _cfg(tmp_path, run_id):
    return OmegaConf.create(
        {
            "experiment": {"name": "identity", "sector": "unit", "run_name": "serial-run"},
            "run": {"root": str(tmp_path), "run_id": run_id, "dir": None},
            "runtime": {"device": "cpu", "dtype": "float64"},
            "callbacks": [],
            "loggers": [],
        }
    )


def test_serial_null_id_keeps_the_historical_derived_shape() -> None:
    """C1: one process, null id -- unchanged behaviour, suffix still random."""

    first = resolve_run_id(None, "serial run")
    second = resolve_run_id(None, "serial run")

    assert _RUN_ID_PATTERN.match(first), first
    # Two resolutions in one process must still differ. Collision resistance
    # across runs is the property the random suffix buys, and the fix keeps it.
    assert first != second


def test_serial_explicit_id_is_returned_unchanged(monkeypatch) -> None:
    """C3: an explicit id is honored as text and nothing is derived."""

    def forbidden(*args, **kwargs):  # pragma: no cover - asserted not to run
        raise AssertionError("an explicit run id must not be regenerated")

    monkeypatch.setattr(artifacts, "generate_run_id", forbidden)

    assert resolve_run_id("2026-01-01_000000_fixed_abcdef", "serial run") == (
        "2026-01-01_000000_fixed_abcdef"
    )


def test_serial_resolution_survives_an_absent_torch_distributed(monkeypatch) -> None:
    """A torch-free environment is a serial run, not an error.

    `tpen.artifacts` is importable without torch and run setup must stay usable,
    so the agreement probe treats an unimportable ``torch.distributed`` as "no
    channel" rather than propagating.
    """

    # A ``None`` entry in sys.modules makes the import statement itself raise
    # ImportError, which is exactly what a torch-free environment does.
    monkeypatch.setitem(sys.modules, "torch.distributed", None)

    assert _RUN_ID_PATTERN.match(resolve_run_id(None, "serial run"))


def test_single_rank_topology_still_derives_locally() -> None:
    """A one-rank launcher topology is the serial path, not a refusal."""

    topology = ExecutionTopology.single_process(device="cpu")

    assert _RUN_ID_PATTERN.match(resolve_run_id(None, "serial run", topology=topology))


def test_multi_rank_topology_without_a_process_group_refuses_a_derived_id() -> None:
    """C5: the reachable scatter path -- several ranks, nothing to agree over.

    ``ExecutionTopology`` is launcher-supplied and torch-free, so a launcher can
    declare three ranks with no process group initialized. There is no channel
    to broadcast over, and deriving locally is precisely the defect, so the only
    safe answer is a loud refusal.
    """

    with pytest.raises(RunIdentityError) as excinfo:
        resolve_run_id(None, "serial run", topology=_multi_rank_topology(3))

    message = str(excinfo.value)
    assert "3 ranks" in message
    assert "no distributed process group is initialized" in message


def test_multi_rank_topology_accepts_an_explicit_id_without_a_process_group() -> None:
    """The refusal must not close a legitimate configuration.

    An explicit id needs no channel: it is already common to every rank that
    reads the same config. This is the over-restriction direction of the C6
    mutation pair -- widening the refusal to cover explicit ids turns this red.
    """

    agreed = resolve_run_id("explicit-id", "serial run", topology=_multi_rank_topology(4))

    assert agreed == "explicit-id"


def test_prepare_run_context_still_fills_a_null_id_for_a_serial_run(tmp_path) -> None:
    """C1 through the real construction path, not just the helper."""

    context = prepare_run_context(_cfg(tmp_path, None), config_path="test.yaml", command="pytest")

    assert _RUN_ID_PATTERN.match(context.metadata.run_id)
    assert str(OmegaConf.select(context.cfg, "run.run_id")) == context.metadata.run_id
    assert context.run_dir.is_dir()


def test_prepare_run_context_refuses_multi_rank_null_id_before_creating_anything(tmp_path) -> None:
    """C5 through the real construction path: refuse while the tree is absent.

    The ordering is the load-bearing part. A refusal raised after ``make_dirs``
    would leave behind exactly the per-rank directories the agreement exists to
    prevent, so the assertion is on the empty filesystem, not only on the raise.
    """

    with pytest.raises(RunIdentityError):
        prepare_run_context(
            _cfg(tmp_path, None),
            config_path="test.yaml",
            command="pytest",
            topology=_multi_rank_topology(2),
        )

    assert list(tmp_path.iterdir()) == []


def test_generate_run_id_remains_process_local() -> None:
    """The raw generator is deliberately unchanged and still non-agreeing.

    Pinned because `tests/unit/test_hi_schema.py` reasons about this property,
    and because the fix must not smuggle a collective into a function whose
    contract is a fresh draw per call.
    """

    assert generate_run_id("hi") != generate_run_id("hi")
