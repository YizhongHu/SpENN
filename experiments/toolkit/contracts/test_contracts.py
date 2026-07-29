"""Focused tests for strict V4 contract records and bundle publication."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit.contracts import (
    ContractBundleV1,
    ContractError,
    ExecutionProfileV1,
    MetricKeyV1,
    ProducerAttemptV1,
    ProducerV1,
    RunV1,
    SeedAssignmentV1,
    SourceDescriptorV1,
    StageResultV1,
    TrialV1,
    publish_bundle,
    read_bundle,
)


def test_records_are_strict_immutable_and_canonical() -> None:
    """IDs ignore source ordering and nested semantic values cannot mutate."""

    choices = {"b": {"z": 2}, "a": 1}
    first = TrialV1(
        bundle_scope_id="scope-a",
        trial_key="trial-a",
        blinded_choices=choices,
        source_keys=("source-a",),
    )
    second = TrialV1(
        bundle_scope_id="scope-a",
        trial_key="trial-a",
        blinded_choices={"a": 1, "b": {"z": 2}},
        source_keys=("source-a",),
    )
    choices["b"]["z"] = 99

    assert first.id == second.id
    assert first.blinded_choices["b"]["z"] == 2
    with pytest.raises(TypeError):
        first.blinded_choices["a"] = 2  # type: ignore[index]

    malformed = first.to_dict()
    malformed["unknown"] = True
    with pytest.raises(ContractError, match="fields mismatch"):
        TrialV1.from_dict(malformed)


def test_bundle_publish_reopen_tamper_and_source_checks(tmp_path: Path) -> None:
    """Rows/manifest/source digests and create-only targets fail closed."""

    source = tmp_path / "input.json"
    source.write_text('{"input": 1}\n')
    bundle = _bundle(source)
    destination = tmp_path / "bundle"

    assert publish_bundle(destination, bundle) == destination
    reopened = read_bundle(destination, source_root=tmp_path)
    assert reopened.bundle_scope_id == bundle.bundle_scope_id
    with pytest.raises(FileExistsError):
        publish_bundle(destination, bundle)

    (destination / "unexpected.txt").write_text("extra\n")
    with pytest.raises(ContractError, match="file population mismatch"):
        read_bundle(destination)
    (destination / "unexpected.txt").unlink()

    source.write_text('{"input": 2}\n')
    with pytest.raises(ContractError, match="source digest mismatch"):
        read_bundle(destination, source_root=tmp_path)
    source.write_text('{"input": 1}\n')

    rows = destination / "trials.jsonl"
    rows.write_text(rows.read_text().replace("trial-a", "trial-b"))
    with pytest.raises(ContractError, match="row file digest mismatch"):
        read_bundle(destination)


def test_bundle_rejects_duplicate_singleton_attempt(tmp_path: Path) -> None:
    """The accepted-A graph cannot serialize two semantic attempts."""

    bundle = _bundle(tmp_path / "input.json")
    duplicate = bundle.producer_attempts[0]
    with pytest.raises(ContractError, match="duplicate ids"):
        ContractBundleV1(
            study=bundle.study,
            bundle_scope_id=bundle.bundle_scope_id,
            sources=bundle.sources,
            trials=bundle.trials,
            seed_assignments=bundle.seed_assignments,
            runs=bundle.runs,
            producers=bundle.producers,
            producer_attempts=(duplicate, duplicate),
            execution_profiles=bundle.execution_profiles,
            metric_keys=bundle.metric_keys,
            stage_results=bundle.stage_results,
        )


def test_bundle_rejects_foreign_run_parent_identity(tmp_path: Path) -> None:
    """Cross-record parent IDs cannot point outside the closed bundle graph."""

    bundle = _bundle(tmp_path / "input.json")
    foreign_run = RunV1(
        bundle_scope_id=bundle.bundle_scope_id,
        trial_id="trial-v1:foreign",
        seed_assignment_id=bundle.seed_assignments[0].id,
        lane="scan",
        run_key="foreign-run",
        source_champion_key=None,
        source_keys=("source-a",),
    )
    with pytest.raises(ContractError, match="run parent identity is absent"):
        ContractBundleV1(
            study=bundle.study,
            bundle_scope_id=bundle.bundle_scope_id,
            sources=bundle.sources,
            trials=bundle.trials,
            seed_assignments=bundle.seed_assignments,
            runs=(foreign_run,),
            producers=bundle.producers,
            producer_attempts=bundle.producer_attempts,
            execution_profiles=bundle.execution_profiles,
            metric_keys=bundle.metric_keys,
            stage_results=bundle.stage_results,
        )


def test_bundle_requires_confirm_champion_source_cited_by_run(tmp_path: Path) -> None:
    """A known champion source still must be explicit evidence for its run."""

    bundle = _bundle(tmp_path / "input.json")
    champion_source = SourceDescriptorV1(
        source_key="source-b",
        root_relative_path="champion.json",
        logical_role="confirm_plan",
        artifact_kind="fixture-champion",
        schema="fixture/v1",
        sha256="b" * 64,
    )
    confirm_seed = SeedAssignmentV1(
        bundle_scope_id=bundle.bundle_scope_id,
        assignment_kind="confirm",
        values={"model_seed": 2},
        source_keys=("source-a",),
    )
    confirm_run = RunV1(
        bundle_scope_id=bundle.bundle_scope_id,
        trial_id=bundle.trials[0].id,
        seed_assignment_id=confirm_seed.id,
        lane="confirm",
        run_key="confirm-run",
        source_champion_key=champion_source.source_key,
        source_keys=("source-a",),
    )
    confirm_producer = ProducerV1(
        bundle_scope_id=bundle.bundle_scope_id,
        run_id=confirm_run.id,
        role="confirm_train",
        source_keys=("source-a",),
    )
    confirm_attempt = ProducerAttemptV1(
        bundle_scope_id=bundle.bundle_scope_id,
        producer_id=confirm_producer.id,
        source_task_id="06_final_train:confirm-run:lineage-a",
        source_execution_task_id="06_final_train:confirm-run:lineage-a",
        source_keys=("source-a",),
    )

    with pytest.raises(ContractError, match="not cited by run source_keys"):
        ContractBundleV1(
            study=bundle.study,
            bundle_scope_id=bundle.bundle_scope_id,
            sources=(bundle.sources[0], champion_source),
            trials=bundle.trials,
            seed_assignments=tuple(
                sorted((*bundle.seed_assignments, confirm_seed), key=lambda row: row.id)
            ),
            runs=tuple(sorted((*bundle.runs, confirm_run), key=lambda row: row.id)),
            producers=tuple(
                sorted((*bundle.producers, confirm_producer), key=lambda row: row.id)
            ),
            producer_attempts=tuple(
                sorted(
                    (*bundle.producer_attempts, confirm_attempt),
                    key=lambda row: row.id,
                )
            ),
            execution_profiles=bundle.execution_profiles,
            metric_keys=bundle.metric_keys,
            stage_results=bundle.stage_results,
        )


def test_partial_bundle_without_manifest_is_rejected_and_not_reused(
    tmp_path: Path,
) -> None:
    """Interrupted row-only publication is invalid and permanently create-only."""

    bundle = _bundle(tmp_path / "input.json")
    partial = tmp_path / "partial-bundle"
    partial.mkdir()
    for table_name, rows in bundle.tables().items():
        (partial / f"{table_name}.jsonl").write_text(
            "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows)
        )

    with pytest.raises(ContractError, match="file population mismatch"):
        read_bundle(partial)
    with pytest.raises(FileExistsError, match="destination already exists"):
        publish_bundle(partial, bundle)


def test_toolkit_contracts_import_without_study_or_spenn_modules() -> None:
    """Generic contracts remain importable in a clean process without a study."""

    command = (
        "import sys; import experiments.toolkit.contracts; "
        "bad=sorted(name for name in sys.modules if "
        "name == 'spenn' or name.startswith('spenn.') or 'pair_stability' in name); "
        "raise SystemExit('unexpected imports: '+repr(bad) if bad else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", command],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _bundle(source_path: Path) -> ContractBundleV1:
    digest = hashlib.sha256(b'{"input": 1}\n').hexdigest()
    source = SourceDescriptorV1(
        source_key="source-a",
        root_relative_path="input.json",
        logical_role="screen_train",
        artifact_kind="fixture",
        schema="fixture/v1",
        sha256=digest,
    )
    trial = TrialV1(
        bundle_scope_id="scope-a",
        trial_key="trial-a",
        blinded_choices={"basis": "x"},
        source_keys=(source.source_key,),
    )
    seed = SeedAssignmentV1(
        bundle_scope_id="scope-a",
        assignment_kind="scan",
        values={"sampler_seed": 1},
        source_keys=(source.source_key,),
    )
    run = RunV1(
        bundle_scope_id="scope-a",
        trial_id=trial.id,
        seed_assignment_id=seed.id,
        lane="scan",
        run_key="run-a",
        source_champion_key=None,
        source_keys=(source.source_key,),
    )
    producer = ProducerV1(
        bundle_scope_id="scope-a",
        run_id=run.id,
        role="screen_train",
        source_keys=(source.source_key,),
    )
    attempt = ProducerAttemptV1(
        bundle_scope_id="scope-a",
        producer_id=producer.id,
        source_task_id="01_train:run-a:lineage-a",
        source_execution_task_id="01_train:run-a:lineage-a",
        source_keys=(source.source_key,),
    )
    profile = ExecutionProfileV1(
        bundle_scope_id="scope-a",
        profile_kind="fanout",
        requested={"partition": "gpu_test"},
        source_keys=(source.source_key,),
    )
    stage = StageResultV1(
        bundle_scope_id="scope-a",
        logical_role="screen_train",
        physical_stage="01_train",
        execution_profile_id=profile.id,
        terminal_population_sha256="0" * 64,
        source_keys=(source.source_key,),
    )
    metric = MetricKeyV1(
        bundle_scope_id="scope-a",
        stage_result_id=stage.id,
        namespace="eval",
        key="energy",
        scalar_representation="float",
        source_keys=(source.source_key,),
    )
    return ContractBundleV1(
        study="fixture-study",
        bundle_scope_id="scope-a",
        sources=(source,),
        trials=(trial,),
        seed_assignments=(seed,),
        runs=(run,),
        producers=(producer,),
        producer_attempts=(attempt,),
        execution_profiles=(profile,),
        metric_keys=(metric,),
        stage_results=(stage,),
    )
