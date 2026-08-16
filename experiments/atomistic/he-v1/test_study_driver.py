"""Driver behavior inside the allocation: what was asked for versus what arrived.

The constraint is a request. This layer is where the request is checked against
reality, and the checks are failures rather than notes:

- a row that finds no GPU, or the wrong one, FAILS -- including the MIG slice
  case, where ``NVIDIA A100-SXM4-40GB MIG 3g.20gb`` must not satisfy an
  ``a100`` (80 GB) row;
- the receipt is written on the failing path too, so a mismatched row still
  says what card it held; and
- a row refuses to run outside a Slurm allocation, because the login-node
  boundary is policy, not preference.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

STUDY_DIR = Path(__file__).resolve().parent


def _load_study_module(name: str) -> ModuleType:
    """Load one study module by path (the study directory is not a package)."""

    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_study_module("driver")
eval_driver = _load_study_module("eval")

# Taken from the subject rather than loaded again: a second importlib load would
# create a SECOND module object with its own exception classes, and
# ``pytest.raises`` would then miss the mismatch this file exists to prove.
strata = driver.strata

ROW: dict[str, Any] = {
    "row_id": "eval-seed0000-step000000010-chain00",
    "index": 1,
    "kind": "eval",
    "stage": "03_eval",
    "seed": 0,
    "checkpoint_step": 10,
    "chain": 0,
    "chain_seed": 900000,
    "config": "experiments/atomistic/he-v1/configs/eval.yaml",
    "overrides": ["runtime.seed=900000", "evaluation.seed=900000"],
    "retained_checkpoint_steps": [],
    "depends_on": ["train-seed0000"],
    "resources": {
        "partition": "seas_gpu",
        "stratum": "h200",
        "timeout_min": 120,
        "cpus": 8,
        "mem_gb": 64,
        "gpus": 1,
        "constraint": "h200",
    },
    "checkpoint_dir_name": "step_000010",
}


def _row(**resource_overrides: Any) -> dict[str, Any]:
    row = json.loads(json.dumps(ROW))
    row["resources"].update(resource_overrides)
    return row


def test_row_refuses_to_run_outside_an_allocation() -> None:
    """A login-node run would produce a receipt citing no node."""

    with pytest.raises(driver.DriverError, match="SLURM_JOB_ID"):
        driver.require_scheduler({})
    with pytest.raises(driver.DriverError, match="SLURM_JOB_ID"):
        driver.require_scheduler({"SLURM_JOB_ID": "   "})
    assert driver.require_scheduler({"SLURM_JOB_ID": "12345"}) == "12345"


def test_matching_device_passes_and_records_the_receipt(tmp_path: Path) -> None:
    """The delivered card is recorded next to the constraint that asked for it."""

    receipt = driver.verify_delivered_device(
        _row(),
        receipt_dir=tmp_path,
        job_id="12345",
        device_reader=lambda: "NVIDIA H200",
        environ={"SLURM_JOB_PARTITION": "seas_gpu"},
    )
    assert receipt["delivered_matches_requested"] is True
    assert receipt["requested_constraint"] == "h200"
    written = json.loads((tmp_path / driver.ALLOCATION_RECEIPT).read_text(encoding="utf-8"))
    assert written["delivered_device"] == "NVIDIA H200"
    assert written["job_id"] == "12345"


def test_mismatched_device_fails_loudly_and_still_writes_the_receipt(tmp_path: Path) -> None:
    """seas_gpu can hand an A100 to an H200 row; that row must not proceed."""

    with pytest.raises(strata.DeliveredDeviceMismatch, match="does not match requested stratum"):
        driver.verify_delivered_device(
            _row(),
            receipt_dir=tmp_path,
            job_id="12345",
            device_reader=lambda: "NVIDIA A100-SXM4-80GB",
            environ={},
        )
    written = json.loads((tmp_path / driver.ALLOCATION_RECEIPT).read_text(encoding="utf-8"))
    assert written["delivered_matches_requested"] is False
    assert written["delivered_device"] == "NVIDIA A100-SXM4-80GB"
    assert "does not match" in written["mismatch_reason"]


def test_absent_device_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """An unverifiable card is not a matching card."""

    with pytest.raises(strata.DeliveredDeviceMismatch, match="no GPU device name"):
        driver.verify_delivered_device(
            _row(),
            receipt_dir=tmp_path,
            job_id="12345",
            device_reader=lambda: None,
            environ={},
        )
    written = json.loads((tmp_path / driver.ALLOCATION_RECEIPT).read_text(encoding="utf-8"))
    assert written["delivered_device_status"] == "absent"
    assert written["delivered_matches_requested"] is False


def test_mig_slice_does_not_satisfy_a_full_card_stratum(tmp_path: Path) -> None:
    """gpu_test serves MIG slices; an a100 row asked for the 80 GB card."""

    with pytest.raises(strata.DeliveredDeviceMismatch):
        driver.verify_delivered_device(
            _row(stratum="a100", partition="kozinsky_gpu", constraint="a100"),
            receipt_dir=tmp_path,
            job_id="1",
            device_reader=lambda: "NVIDIA A100-SXM4-40GB MIG 3g.20gb",
            environ={},
        )


def test_overrides_pin_the_run_directory_to_the_row(tmp_path: Path) -> None:
    """A run directory is a function of the row, not of a generated timestamp."""

    overrides = driver.row_overrides(_row(), run_root=tmp_path, extra=["load.path=/ckpt"])
    assert f"run.run_id={ROW['row_id']}" in overrides
    assert "run.layout=flat" in overrides
    assert f"run.root={tmp_path}" in overrides
    assert "load.path=/ckpt" in overrides
    assert overrides[:2] == ROW["overrides"]


def test_override_paths_absent_from_the_config_are_rejected(tmp_path: Path) -> None:
    """OmegaConf silently CREATES unknown dotlist keys, so a typo would no-op."""

    config = tmp_path / "config.yaml"
    config.write_text("runtime:\n  seed: 0\n", encoding="utf-8")
    merged = driver.build_config(config, ["runtime.seed=7"])
    assert merged.runtime.seed == 7
    with pytest.raises(driver.DriverError, match="absent from"):
        driver.build_config(config, ["runtime.sedd=7"])


def test_run_row_fails_before_running_when_the_card_is_wrong(tmp_path: Path) -> None:
    """No physics starts on a mismatched allocation."""

    started: list[Any] = []
    with pytest.raises(strata.DeliveredDeviceMismatch):
        driver.run_row(
            _row(),
            results_root=tmp_path,
            plan_attempt_id="20260815T120000",
            launch_attempt_id="20260815T130000",
            device_reader=lambda: "NVIDIA A100-SXM4-80GB",
            runner=lambda cfg, **kwargs: started.append(cfg) or 0,
            environ={"SLURM_JOB_ID": "1"},
        )
    assert started == []


def test_planned_overrides_exist_in_the_checked_in_configs() -> None:
    """Every scientific override a row applies must target a declared key.

    The configs belong to the atom lane (A7) and to H-F1; this asserts the
    driver's overrides land on them rather than creating a shadow key that
    changes nothing.
    """

    plan = _load_study_module("plan")
    repo_root = STUDY_DIR.parents[2]
    config = plan.validate_grid_config(
        {
            "study": "tpen_he_v1",
            "train_config": "experiments/atomistic/he-v1/configs/train.yaml",
            "eval_config": "experiments/atomistic/he-v1/configs/eval.yaml",
            "seeds": [0],
            "checkpoint_steps": [10],
            "eval_chains": 1,
            "eval_chain_seed_base": 900000,
            "train_resources": {
                "partition": "kozinsky_gpu",
                "stratum": "a100",
                "timeout_min": 600,
                "cpus": 16,
                "mem_gb": 128,
                "gpus": 1,
            },
            "eval_resources": {
                "partition": "seas_gpu",
                "stratum": "h200",
                "timeout_min": 120,
                "cpus": 8,
                "mem_gb": 64,
                "gpus": 1,
            },
            "gate_spec": {},
    "seed_stages": [[0]],
    "convergence_assessment": {
        "method": "windowed_means_sign_test",
        "n_windows": 20,
        "n_trailing_windows": 8,
        "window_width_min_tau_multiple": 5.0,
        "on_inadequate": "report_only",
        "may_reselect": False,
    },
        }
    )
    for row in plan.expand_rows(config):
        extra = ["load.path=/ckpt"] if row["kind"] == "eval" else []
        driver.build_config(
            repo_root / row["config"],
            [*row["overrides"], *extra],
            checked=[*row["overrides"], *extra],
        )


def test_run_row_returns_the_configured_runs_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed run stays failed; nothing here converts it into a success."""

    config = tmp_path / "eval.yaml"
    config.write_text(
        "runtime:\n  seed: 0\nevaluation:\n  seed: 0\nrun:\n  root: out\n  run_id: null\n",
        encoding="utf-8",
    )
    row = _row()
    row["config"] = str(config)
    exit_code = driver.run_row(
        row,
        results_root=tmp_path,
        plan_attempt_id="20260815T120000",
        launch_attempt_id="20260815T130000",
        device_reader=lambda: "NVIDIA H200",
        runner=lambda cfg, **kwargs: 1,
        environ={"SLURM_JOB_ID": "77"},
    )
    assert exit_code == 1
    result_dir = tmp_path / "03_eval" / row["row_id"] / "20260815T120000"
    assert (result_dir / driver.ALLOCATION_RECEIPT).is_file()
    written = json.loads((result_dir / "row.json").read_text(encoding="utf-8"))
    assert written["launch_attempt_id"] == "20260815T130000"
    assert written["job_id"] == "77"


def test_incomplete_checkpoint_is_refused(tmp_path: Path) -> None:
    """A partially written checkpoint restores a model nobody planned."""

    checkpoint = tmp_path / "step_000010"
    with pytest.raises(eval_driver.driver.DriverError, match="does not exist"):
        eval_driver.require_complete_checkpoint(checkpoint)
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(eval_driver.driver.DriverError, match="incomplete"):
        eval_driver.require_complete_checkpoint(checkpoint)
    (checkpoint / eval_driver.COMPLETE_MARKER).write_text("", encoding="utf-8")
    assert eval_driver.require_complete_checkpoint(checkpoint) == checkpoint


def test_slurm_time_formats_days_and_hours() -> None:
    """A wrong ``--time`` is a truncated run, and rows may not resume."""

    assert strata.slurm_time(120) == "02:00:00"
    assert strata.slurm_time(2880) == "2-00:00:00"
    assert strata.slurm_time(10080) == "7-00:00:00"


def test_experiments_import_boundary_holds() -> None:
    """``experiments/`` imports ``tpen`` in exactly two enumerated places.

    The exceptions are listed one by one rather than relaxed to a rule, so a
    third one cannot appear without a deliberate edit here.

    `driver.py` launches runs, so it needs the runner. `assess_convergence.py`
    needs the SANCTIONED tau estimator: `tpen.statistics` is the single producer
    of ``tau_int`` in this repository and re-implementing Geyer's initial
    positive sequence inside the study would create a second estimator that
    could silently drift from the one the study reports. Both imports are
    function-local, so both modules still load on a login node without torch --
    which is the property this boundary exists to protect.
    """

    offenders: dict[str, list[str]] = {}
    for path in sorted(STUDY_DIR.glob("*.py")):
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith(("import tpen", "from tpen"))
        ]
        if lines:
            offenders[path.name] = lines
    assert offenders == {
        "driver.py": [
            "from tpen.run import run_from_config  # noqa: PLC0415 - sanctioned launcher exception"
        ],
        "assess_convergence.py": [
            "from tpen.statistics.autocorrelation import (  # noqa: PLC0415 - sanctioned exception"
        ],
    }


def test_the_study_planning_modules_import_without_torch() -> None:
    """The boundary's PURPOSE, asserted rather than inferred from import lines.

    Enumerating import statements proves what the files say; this proves what
    they do. Every study module must be importable in an interpreter with no
    torch, because planning, launching and assessment all run on a login node
    where torch is absent. A future top-level ``import torch`` in any of them
    passes the enumeration above only if someone also edits it, but fails here
    unconditionally.

    A SUBPROCESS is required, not a convenience. In the composed suite torch is
    already in ``sys.modules`` from earlier tests, so an in-process check of
    ``"torch" not in sys.modules`` could only ever fail -- an instrument that
    cannot see the thing it is pointed at. A clean interpreter is the only
    context in which this claim is decidable.
    """

    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    probe = """
import importlib.util, sys
names = ["plan", "launch", "collect", "assess_convergence", "layout", "strata"]
import pathlib
study = pathlib.Path(sys.argv[1])
for name in names:
    path = study / (name + ".py")
    if not path.exists():
        continue
    spec = importlib.util.spec_from_file_location("probe_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise SystemExit("%s pulled torch in at import time" % name)
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(STUDY_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ok" in completed.stdout
