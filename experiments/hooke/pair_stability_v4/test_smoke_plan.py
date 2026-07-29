"""Closed-contract tests for the approved V4-0 smoke plan."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import audit  # noqa: E402


@pytest.fixture(scope="module")
def approved_plan(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """Generate the real pinned 64-job V4 plan once for mutation tests."""

    root = tmp_path_factory.mktemp("approved-smoke") / "results"
    attempt = "approved-smoke-test"
    _run_plan(root, attempt=attempt, study="pair_stability_v4")
    return root, attempt


def test_approved_fixture_is_self_verified_and_matches_real_plan(
    approved_plan: tuple[Path, str],
) -> None:
    """The tracked oracle binds its inputs and the real ordered plan."""

    root, attempt = approved_plan
    contract = json.loads(audit.SMOKE_PLAN_CONTRACT_PATH.read_text())

    assert audit.verify_smoke_plan_contract(contract) == ()
    assert (
        audit.smoke_plan_digest(
            root,
            attempt=attempt,
            expected_study="pair_stability_v4",
        )
        == contract["sha256"]
    )


def test_v3_and_v4_plans_share_one_identity_normalized_oracle(
    approved_plan: tuple[Path, str],
    tmp_path: Path,
) -> None:
    """Only the enumerated migration identity transform differs."""

    v4_root, v4_attempt = approved_plan
    v3_root = tmp_path / "v3-results"
    v3_attempt = "approved-v3-test"
    _run_plan(v3_root, attempt=v3_attempt, study="pair_stability_v3")

    assert audit.smoke_plan_projection(
        v3_root,
        attempt=v3_attempt,
        expected_study="pair_stability_v3",
    ) == audit.smoke_plan_projection(
        v4_root,
        attempt=v4_attempt,
        expected_study="pair_stability_v4",
    )


def test_projection_never_runs_the_planner(
    approved_plan: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auditing is a read-only comparison to the tracked oracle."""

    root, attempt = approved_plan

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("planner executed during audit")

    monkeypatch.setattr(subprocess, "run", fail)
    assert audit.smoke_plan_digest(
        root,
        attempt=attempt,
        expected_study="pair_stability_v4",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("major_choices", {"basis": "wrong"}),
        ("scan_seed", 999),
        ("static_overrides", {"training.max_steps": 999}),
        ("tags", ["unexpected"]),
    ],
)
def test_projection_rejects_scientific_job_mutations(
    approved_plan: tuple[Path, str],
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    """Choices, seeds, overrides, and tags remain verbatim science."""

    root, attempt = _copy_plan(approved_plan, tmp_path)
    manifest_path = root / "00_grid" / attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["jobs"][0][field] = replacement
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError):
        audit.smoke_plan_projection(
            root,
            attempt=attempt,
            expected_study="pair_stability_v4",
        )


@pytest.mark.parametrize("mutation", ["reorder", "minimal", "extra-field"])
def test_projection_rejects_manifest_population_and_schema_drift(
    approved_plan: tuple[Path, str],
    tmp_path: Path,
    mutation: str,
) -> None:
    """Order, all 64 rows, and the closed manifest schema are mandatory."""

    root, attempt = _copy_plan(approved_plan, tmp_path)
    manifest_path = root / "00_grid" / attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "reorder":
        manifest["jobs"].reverse()
    elif mutation == "minimal":
        manifest["jobs"] = manifest["jobs"][:1]
        manifest["n_jobs"] = 1
    else:
        manifest["not_approved"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError):
        audit.smoke_plan_projection(
            root,
            attempt=attempt,
            expected_study="pair_stability_v4",
        )


@pytest.mark.parametrize("mutation", ["empty", "not-inverse"])
def test_projection_rejects_invalid_unblind_bijections(
    approved_plan: tuple[Path, str],
    tmp_path: Path,
    mutation: str,
) -> None:
    """Every blinded slot must map bijectively to the complete semantic axis."""

    root, attempt = _copy_plan(approved_plan, tmp_path)
    unblind_path = root / "00_grid" / attempt / "unblind.json"
    unblind = json.loads(unblind_path.read_text())
    first_axis = next(iter(unblind["axes"]))
    if mutation == "empty":
        unblind["axes"][first_axis] = {
            "slot_to_value": {},
            "value_to_slot": {},
        }
    else:
        first_slot = next(
            iter(unblind["axes"][first_axis]["slot_to_value"])
        )
        unblind["axes"][first_axis]["slot_to_value"][first_slot] = "wrong"
    _write_json(unblind_path, unblind)

    with pytest.raises(ValueError):
        audit.smoke_plan_projection(
            root,
            attempt=attempt,
            expected_study="pair_stability_v4",
        )


def test_projection_rejects_command_override_order_and_job_file_drift(
    approved_plan: tuple[Path, str],
    tmp_path: Path,
) -> None:
    """Command ordering and per-run job copies are part of the exact plan."""

    root, attempt = _copy_plan(approved_plan, tmp_path)
    grid_dir = root / "00_grid" / attempt
    manifest_path = grid_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    job = manifest["jobs"][0]
    job["overrides"][:2] = reversed(job["overrides"][:2])
    _write_json(manifest_path, manifest)
    _write_json(grid_dir / "jobs" / f"{job['run_id']}.json", job)

    contract = json.loads(audit.SMOKE_PLAN_CONTRACT_PATH.read_text())
    assert (
        audit.smoke_plan_digest(
            root,
            attempt=attempt,
            expected_study="pair_stability_v4",
        )
        != contract["sha256"]
    )

    root, attempt = _copy_plan(approved_plan, tmp_path / "job-file")
    grid_dir = root / "00_grid" / attempt
    manifest = json.loads((grid_dir / "manifest.json").read_text())
    job_path = grid_dir / "jobs" / f"{manifest['jobs'][0]['run_id']}.json"
    job = json.loads(job_path.read_text())
    job["tags"] = ["job-file-only-drift"]
    _write_json(job_path, job)
    with pytest.raises(ValueError, match="job file differs"):
        audit.smoke_plan_projection(
            root,
            attempt=attempt,
            expected_study="pair_stability_v4",
        )


def test_fixture_tamper_and_generator_provenance_drift_are_rejected() -> None:
    """Neither fixture bytes nor their pinned generator inputs may float."""

    original = json.loads(audit.SMOKE_PLAN_CONTRACT_PATH.read_text())
    tampered = json.loads(json.dumps(original))
    tampered["projection"]["n_jobs"] = 1
    assert "projection self-digest mismatch" in audit.verify_smoke_plan_contract(
        tampered
    )

    stale = json.loads(json.dumps(original))
    stale["generator"]["blind_seed"] = 812
    assert (
        "generator source/config/route provenance mismatch"
        in audit.verify_smoke_plan_contract(stale)
    )


def _run_plan(root: Path, *, attempt: str, study: str) -> None:
    study_dir = STUDY_DIR.parent / study
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(STUDY_DIR.parent / "pair_stability_v3" / "plan.py"),
            "--grid",
            str(study_dir / "configs" / "smoke.yaml"),
            "--config",
            str(study_dir / "configs" / "pair_stability.yaml"),
            "--results-root",
            str(root),
            "--attempt-id",
            attempt,
            "--timezone",
            "America/New_York",
            "--blind",
            "--blind-seed",
            "811",
            "--python",
            "python",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _copy_plan(
    source: tuple[Path, str],
    destination: Path,
) -> tuple[Path, str]:
    _, attempt = source
    fresh = destination / "results"
    _run_plan(fresh, attempt=attempt, study="pair_stability_v4")
    return fresh, attempt


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
