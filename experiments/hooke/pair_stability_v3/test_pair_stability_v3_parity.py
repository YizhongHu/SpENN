"""Cluster parity test for pair-stability v2 and v3 lineages."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import parity


def _write_fixture_lineages(root: Path, *, v2_attempt: str, v3_attempt: str) -> tuple[Path, Path]:
    """Write minimal, comparison-complete v2/v3 result trees."""

    v2_dir = root / "pair_stability_v2"
    v3_dir = root / "pair_stability_v3"
    for study_dir, attempt, study in ((v2_dir, v2_attempt, "pair_stability_v2"), (v3_dir, v3_attempt, "pair_stability_v3")):
        for parts in parity._comparison_artifacts():
            stage, *rest = parts
            path = study_dir / "results" / stage / attempt / Path(*rest)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Embed the study name and the concrete attempt id so the test
            # proves both are normalized before comparison.
            if path.suffix == ".csv":
                path.write_text(f"study,attempt_id\n{study},{attempt}\n")
            elif path.suffix == ".json":
                path.write_text(json.dumps({"study": study, "attempt_id": attempt}) + "\n")
            else:
                path.write_text(f"{study} report for {attempt}\n")
    for stage in ("01_train", "02_validation"):
        plan_dir = v3_dir / "results" / stage / "stage_plans" / v3_attempt
        plan_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("stage_manifest.json", "tasks.jsonl", "execution_records.jsonl"):
            (plan_dir / filename).write_text("{}\n" if filename.endswith(".json") else "")
    return v2_dir, v3_dir


def test_pair_stability_v3_parity_compares_fixed_v2_reference_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh v3 attempt compares clean against a fixed, older v2 attempt."""

    v2_attempt = "parity-v2v3"
    v3_attempt = "parity-v2v3-20260705-210000"
    v2_dir, v3_dir = _write_fixture_lineages(tmp_path, v2_attempt=v2_attempt, v3_attempt=v3_attempt)
    monkeypatch.setattr(parity, "V2_DIR", v2_dir)
    monkeypatch.setattr(parity, "V3_DIR", v3_dir)

    differences = parity.compare_lineages(v2_attempt_id=v2_attempt, v3_attempt_id=v3_attempt)
    assert differences == []

    # A real content difference must still be caught after normalization.
    summary = v3_dir / "results" / "03_collect" / v3_attempt / "summary.csv"
    summary.write_text(f"study,attempt_id\npair_stability_v3,{v3_attempt}\nextra,row\n")
    differences = parity.compare_lineages(v2_attempt_id=v2_attempt, v3_attempt_id=v3_attempt)
    assert differences == ["normalized artifact differs: 03_collect/summary.csv"]


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


def test_pair_stability_v3_parity_normalizes_volatile_selection_fallback() -> None:
    """Volatile fallback winners should not fail v2/v3 artifact comparison."""

    left = {
        "overall_metric": "train/runtime/wall_time_sec_seed_median",
        "overall_metric_value": "1.0",
        "overall_champion": "b-B01_m-A01_lr-3e-4_ch-4",
        "secondary_metric": "train/runtime/wall_time_sec_seed_median",
        "secondary_metric_value": "1.0",
        "secondary_champion": "b-B01_m-A01_lr-3e-4_ch-4",
        "champions": [{"config_id": "stable-winner"}],
    }
    right = {
        "overall_metric": "train/runtime/wall_time_sec_seed_median",
        "overall_metric_value": "2.0",
        "overall_champion": "b-B01_m-A01_lr-1e-3_ch-4",
        "secondary_metric": "train/runtime/wall_time_sec_seed_median",
        "secondary_metric_value": "2.0",
        "secondary_champion": "b-B01_m-A01_lr-1e-3_ch-4",
        "champions": [{"config_id": "stable-winner"}],
    }

    assert parity._normalize(left) == parity._normalize(right)


@pytest.mark.integration
def test_pair_stability_v3_matches_v2_completed_submission_lineage() -> None:
    """Compare completed v2/v3 parity artifacts, including submissions."""

    if os.environ.get("SPENN_PAIR_STABILITY_PARITY") != "1":
        pytest.skip(
            "set SPENN_PAIR_STABILITY_PARITY=1 after running "
            "`python experiments/hooke/pair_stability_v3/parity.py print-runbook` commands"
        )
    attempt_id = os.environ.get("SPENN_PAIR_STABILITY_PARITY_ATTEMPT", parity.DEFAULT_ATTEMPT_ID)
    differences = parity.compare_lineages(
        attempt_id=attempt_id,
        v2_attempt_id=os.environ.get("SPENN_PAIR_STABILITY_PARITY_V2_ATTEMPT"),
        v3_attempt_id=os.environ.get("SPENN_PAIR_STABILITY_PARITY_V3_ATTEMPT"),
    )
    assert differences == []
