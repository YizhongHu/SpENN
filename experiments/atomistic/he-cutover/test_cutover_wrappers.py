from __future__ import annotations

import pytest

# Siblings are loaded study-scoped, not by bare import: experiments/ has
# several same-named modules and the first study loaded would otherwise own
# the bare name for every study after it. See experiments/toolkit/study_imports.py.
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

_TPEN_REPO_ROOT = _TpenPath(__file__).resolve().parents[3]
if str(_TPEN_REPO_ROOT) not in _tpen_sys.path:
    _tpen_sys.path.insert(0, str(_TPEN_REPO_ROOT))

from experiments.toolkit.study_imports import sibling  # noqa: E402

run_train_row = sibling(__file__, 'run_train_row')


def test_allocation_detection_accepts_slurm() -> None:
    assert run_train_row.require_allocation({"SLURM_JOB_ID": "123"}) == "123"


def test_allocation_detection_accepts_pbs() -> None:
    assert run_train_row.require_allocation({"PBS_JOBID": "456.server"}) == "456.server"


def test_allocation_detection_refuses_login_node() -> None:
    with pytest.raises(RuntimeError, match="allocation required"):
        run_train_row.require_allocation({})

