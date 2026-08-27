from __future__ import annotations

import pytest

import run_train_row


def test_allocation_detection_accepts_slurm() -> None:
    assert run_train_row.require_allocation({"SLURM_JOB_ID": "123"}) == "123"


def test_allocation_detection_accepts_pbs() -> None:
    assert run_train_row.require_allocation({"PBS_JOBID": "456.server"}) == "456.server"


def test_allocation_detection_refuses_login_node() -> None:
    with pytest.raises(RuntimeError, match="allocation required"):
        run_train_row.require_allocation({})

