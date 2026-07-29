"""Consumer-backed tests for V4-1A sidecar projection and verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import contract_sidecars  # noqa: E402
import audit  # noqa: E402
from experiments.toolkit.contracts import publish_bundle, read_bundle  # noqa: E402
from test_control_audit import _write_valid_control_evidence  # noqa: E402
from test_reference import _completed_lineage  # noqa: E402


def test_finalizer_projects_verifies_and_publishes_closed_sidecars(
    tmp_path: Path,
) -> None:
    """A completed synthetic route gains one verified, create-only sidecar set."""

    root, attempts = _completed_lineage(tmp_path)
    lineage = attempts["grid"]
    _write_valid_control_evidence(root, attempts, write_terminal_result=False)

    receipt = contract_sidecars.finalize_contract_sidecars(root, lineage_id=lineage)
    bundle_dir = contract_sidecars.contract_bundle_directory(root, lineage_id=lineage)
    bundle = read_bundle(bundle_dir, source_root=root)

    assert receipt == contract_sidecars.contract_verification_receipt_path(
        root, lineage_id=lineage
    )
    assert len(bundle.trials) == 64
    assert len(bundle.runs) == 72
    assert len(bundle.producers) == len(bundle.producer_attempts) == 72
    assert {stage.logical_role for stage in bundle.stage_results} == {
        "screen_plan",
        "screen_train",
        "screen_eval",
        "screen_collect",
        "select",
        "confirm_plan",
        "confirm_train",
        "confirm_eval",
        "confirm_collect",
        "report",
    }
    assert all("latest" not in source.root_relative_path for source in bundle.sources)
    assert all(run.source_champion_key is None for run in bundle.runs if run.lane == "scan")
    assert all(run.source_champion_key is not None for run in bundle.runs if run.lane == "confirm")
    assert contract_sidecars.verify_contract_sidecars(root, lineage_id=lineage)[
        "bundle_scope_id"
    ] == bundle.bundle_scope_id
    assert not (root / "_v4" / "stack" / lineage / "controller-result.json").exists()
    receipt_payload = json.loads(receipt.read_text())
    receipt_payload["manifest_sha256"] = "0" * 64
    receipt.write_text(json.dumps(receipt_payload, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="receipt manifest_sha256 differs"):
        contract_sidecars.verify_contract_sidecars(root, lineage_id=lineage)
    with pytest.raises(FileExistsError):
        contract_sidecars.finalize_contract_sidecars(root, lineage_id=lineage)


def test_projection_ignores_live_checkpoint_pointer_and_rejects_competing_attempt(
    tmp_path: Path,
) -> None:
    """Accepted-A ignores live checkpoint pointers and competing attempts fail."""

    root, attempts = _completed_lineage(tmp_path)
    lineage = attempts["grid"]
    _write_valid_control_evidence(root, attempts, write_terminal_result=False)
    initial = contract_sidecars.project_contract_bundle(root, lineage_id=lineage)
    run_key = next(run.run_key for run in initial.runs if run.lane == "scan")
    checkpoint_root = root / "01_train" / run_key / lineage / "checkpoints"
    _publish_second_complete_checkpoint(checkpoint_root)
    pointer = checkpoint_root / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "checkpoint_dir": "step_000001",
                "step": 1,
                "created_at_unix": 1.0,
            },
            sort_keys=True,
        )
        + "\n"
    )

    # The live pointer now names a different valid, complete concrete
    # checkpoint, so this remains a valid legacy completed lineage.
    assert audit.audit_completed_lineage(root, attempts=attempts) == ()
    after_pointer = contract_sidecars.project_contract_bundle(root, lineage_id=lineage)

    assert _serialized_bundle(after_pointer) == _serialized_bundle(initial)

    competing = root / "01_train" / run_key / "other-recognized-attempt"
    competing.mkdir()
    with pytest.raises(ValueError, match="unsupported attempt evidence"):
        contract_sidecars.project_contract_bundle(root, lineage_id=lineage)


def test_projection_rejects_typed_resume(tmp_path: Path) -> None:
    """Typed resume evidence is a separate rejected semantic attempt edge."""

    root, attempts = _completed_lineage(tmp_path)
    lineage = attempts["grid"]
    _write_valid_control_evidence(root, attempts, write_terminal_result=False)

    tasks_path = root / "01_train" / "stage_plans" / lineage / "tasks.jsonl"
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    rows[0]["resume"] = {"recognized": "future-retry"}
    tasks_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    with pytest.raises(ValueError, match="TaskSpec.resume"):
        contract_sidecars.project_contract_bundle(root, lineage_id=lineage)


def test_verifier_rejects_structural_bundle_without_receipt(tmp_path: Path) -> None:
    """A manifest-readable sidecar is not accepted V4-1A evidence by itself."""

    root, attempts = _completed_lineage(tmp_path)
    lineage = attempts["grid"]
    _write_valid_control_evidence(root, attempts, write_terminal_result=False)
    bundle = contract_sidecars.project_contract_bundle(root, lineage_id=lineage)
    publish_bundle(
        contract_sidecars.contract_bundle_directory(root, lineage_id=lineage),
        bundle,
    )

    with pytest.raises(ValueError, match="required structured artifact is unavailable"):
        contract_sidecars.verify_contract_sidecars(root, lineage_id=lineage)


def _publish_second_complete_checkpoint(checkpoint_root: Path) -> None:
    """Make a second concrete checkpoint that satisfies legacy completeness."""

    original = checkpoint_root / "step_000000"
    second = checkpoint_root / "step_000001"
    manifest = json.loads((original / "manifest.json").read_text())
    manifest["step"] = 1
    second.mkdir()
    (second / "model.pt").write_bytes(b"second-model")
    (second / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (second / "COMPLETE").write_text("complete\n")


def _serialized_bundle(bundle: object) -> dict[str, object]:
    """Return every source and row serialization for pointer-invariance proof."""

    return {
        "sources": [source.to_dict() for source in bundle.sources],
        "trials": [row.to_dict() for row in bundle.trials],
        "seed_assignments": [row.to_dict() for row in bundle.seed_assignments],
        "runs": [row.to_dict() for row in bundle.runs],
        "producers": [row.to_dict() for row in bundle.producers],
        "producer_attempts": [row.to_dict() for row in bundle.producer_attempts],
        "execution_profiles": [row.to_dict() for row in bundle.execution_profiles],
        "metric_keys": [row.to_dict() for row in bundle.metric_keys],
        "stage_results": [row.to_dict() for row in bundle.stage_results],
    }
