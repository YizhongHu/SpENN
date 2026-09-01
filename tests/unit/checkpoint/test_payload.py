"""Independent tests for checkpoint payload profiles and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import (
    MODEL_ONLY_PAYLOAD,
    PAYLOAD_MANIFEST_SCHEMA,
    TRAIN_RESUME_PAYLOAD,
    CheckpointPayload,
    ModelOnly,
    TrainResume,
    save_checkpoint,
)
from tpen.checkpoint.manifest import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
)
from tpen.checkpoint.schema import read_manifest


def test_model_only_profile_has_a_stable_evaluation_manifest() -> None:
    payload = ModelOnly()

    assert payload == MODEL_ONLY_PAYLOAD
    assert payload.profile == "model_only"
    assert payload.required_files == ("model",)
    assert payload.required_state == ()
    assert payload.allowed_restore_modes == ("model_only",)
    assert payload.to_manifest() == {
        "schema": PAYLOAD_MANIFEST_SCHEMA,
        "profile": "model_only",
        "required_files": ["model"],
        "required_state": [],
        "restore_intents": ["model_only"],
    }


def test_train_resume_profile_names_every_recovery_component() -> None:
    payload = TrainResume()

    assert payload == TRAIN_RESUME_PAYLOAD
    assert payload.profile == "train_resume"
    assert payload.required_files == (
        "model",
        "optimizer",
        "trainer",
        "sampler",
        "rng",
    )
    assert payload.required_state == ("next_iteration", "completed_updates")
    assert payload.allowed_restore_modes == ("model_only", "train_resume")


@pytest.mark.parametrize("payload", [ModelOnly(), TrainResume()])
def test_profile_manifest_round_trips_to_the_canonical_profile(
    payload: CheckpointPayload,
) -> None:
    manifest = payload.to_manifest()

    assert CheckpointPayload.from_manifest(manifest) == payload
    assert json.dumps(manifest, sort_keys=True, separators=(",", ":")) == json.dumps(
        payload.to_manifest(), sort_keys=True, separators=(",", ":")
    )


def test_model_only_is_admitted_for_evaluation_and_rejected_for_training_resume() -> None:
    payload = ModelOnly()

    payload.validate_restore_intent("model_only")
    with pytest.raises(ValueError, match="cannot satisfy restore mode 'train_resume'"):
        payload.validate_restore_intent("train_resume")


def test_train_resume_admits_both_restore_intents() -> None:
    payload = TrainResume()

    payload.validate_restore_intent("model_only")
    payload.validate_restore_intent("train_resume")


def test_train_resume_rejects_each_missing_required_file() -> None:
    payload = TrainResume()
    complete = {name: f"{name}.payload" for name in payload.required_files}

    for missing in payload.required_files:
        files = {name: path for name, path in complete.items() if name != missing}
        with pytest.raises(ValueError, match=missing):
            payload.validate_files(files)


def test_train_resume_rejects_missing_progress_state() -> None:
    payload = TrainResume()

    with pytest.raises(ValueError, match="completed_updates"):
        payload.validate_state({"next_iteration": 3})


def test_profile_value_validation_rejects_empty_or_duplicate_contract_entries() -> None:
    with pytest.raises(ValueError, match="required_files must not be empty"):
        CheckpointPayload(
            profile="model_only",
            required_files=(),
            required_state=(),
            restore_intents=("model_only",),
        )

    with pytest.raises(ValueError, match="required_files entries must be unique"):
        CheckpointPayload(
            profile="model_only",
            required_files=("model", "model"),
            required_state=("state",),
            restore_intents=("model_only",),
        )


def test_manifest_restore_gate_reads_payload_intent_without_callback_state(
    tmp_path,
) -> None:
    path = tmp_path / "manifest.json"
    CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        kind=CHECKPOINT_KIND,
        next_iteration=3,
        completed_updates=3,
        created_at_unix=0.0,
        files={"model": "model.pt"},
        hashes={},
        runtime={},
        provenance={},
        payload=ModelOnly().to_manifest(),
    ).write(path)

    assert read_manifest(path, mode="model_only").payload == ModelOnly().to_manifest()
    with pytest.raises(ValueError, match="cannot satisfy restore mode 'train_resume'"):
        read_manifest(path, mode="train_resume")


def test_manifest_rejects_a_noncanonical_payload_contract(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    payload = TrainResume().to_manifest()
    payload["required_files"] = list(reversed(payload["required_files"]))
    CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        kind=CHECKPOINT_KIND,
        next_iteration=3,
        completed_updates=3,
        created_at_unix=0.0,
        files={name: f"{name}.payload" for name in TrainResume().required_files},
        hashes={},
        runtime={},
        provenance={},
        payload=payload,
    ).write(path)

    with pytest.raises(ValueError, match="not canonical"):
        read_manifest(path, mode="train_resume")


def test_explicit_model_only_payload_owns_save_defaults_and_manifest(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=2,
        completed_updates=2,
        model=model,
        context=_context(),
        payload=ModelOnly(),
    )

    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    assert manifest["payload"] == ModelOnly().to_manifest()
    assert manifest["files"] == {
        "model": "model.pt",
        "resolved_config": "resolved_config.yaml",
    }
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "COMPLETE",
        "manifest.json",
        "model.pt",
        "resolved_config.yaml",
    ]


def test_explicit_model_only_payload_rejects_train_state_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires save_optimizer=False"):
        save_checkpoint(
            output_dir=tmp_path / "checkpoints",
            next_iteration=2,
            completed_updates=2,
            model=torch.nn.Linear(2, 1).double(),
            context=_context(),
            payload=ModelOnly(),
            save_optimizer=True,
        )
    assert not (tmp_path / "checkpoints").exists()


def _context() -> SimpleNamespace:
    """Minimal direct-save context; no callback or scheduling object is used."""

    return SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "model": {"name": "linear"},
                "optimizer": {"name": "adam"},
                "trainer": {"name": "trainer"},
                "sampler": {"name": "sampler"},
                "hamiltonian_terms": {"name": "constant"},
                "run": {"run_id": "payload-test", "dir": "/tmp/payload-test"},
                "study": {"name": "unit", "config_id": "payload"},
            }
        ),
        metadata=SimpleNamespace(
            run_id="payload-test",
            device="cpu",
            dtype="float64",
            git_commit="deadbeef",
            git_branch="codex/checkpoint",
            dirty_worktree=False,
            command="pytest",
            extra={"slurm": {}},
        ),
        run_dir=Path("/tmp/payload-test"),
    )
