"""Cluster parity test for pair-stability v2 and v3 lineages."""

from __future__ import annotations

import os

import pytest

import parity


def test_pair_stability_v3_parity_runbook_uses_test_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the e2e parity runbook submits only to the requested test partitions."""

    monkeypatch.setattr(parity, "prepare_v2_config", lambda *, attempt_id: parity.V2_DIR / "results" / "grid.yaml")
    commands = parity.submission_runbook(attempt_id="T1")
    command_texts = [
        " ".join(command)
        for command in commands
        if not isinstance(command, str)
    ]

    assert any("train.py" in command and "--slurm-cpu-partition test" in command for command in command_texts)
    assert any("train.py" in command and "--device cpu" in command for command in command_texts)
    assert not any("train.py" in command and "--device cpu,cuda" in command for command in command_texts)
    assert any("validate.py" in command and "--slurm-partition gpu_test" in command for command in command_texts)
    assert any("validate.py" in command and "--device cuda" in command for command in command_texts)
    assert any("final_train.py" in command and "--slurm-cpu-partition test" in command for command in command_texts)
    assert any("final_train.py" in command and "--device cpu" in command for command in command_texts)
    assert any("final_eval.py" in command and "--slurm-partition gpu_test" in command for command in command_texts)
    assert any("--slurm-mem-gb 60" in command for command in command_texts)
    assert all("--chunk-size 8" in command for command in command_texts if "--extra submitit" in command)
    assert any("collect.py" in command for command in command_texts)
    assert any("final_report.py" in command for command in command_texts)


@pytest.mark.integration
def test_pair_stability_v3_matches_v2_completed_submission_lineage() -> None:
    """Compare completed v2/v3 parity artifacts, including submissions."""

    if os.environ.get("SPENN_PAIR_STABILITY_PARITY") != "1":
        pytest.skip(
            "set SPENN_PAIR_STABILITY_PARITY=1 after running "
            "`python experiments/hooke/pair_stability_v3/parity.py print-runbook` commands"
        )
    attempt_id = os.environ.get("SPENN_PAIR_STABILITY_PARITY_ATTEMPT", parity.DEFAULT_ATTEMPT_ID)
    differences = parity.compare_lineages(attempt_id=attempt_id)
    assert differences == []
