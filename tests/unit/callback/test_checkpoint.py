"""Tests for package-owned checkpoint restore helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import tpen.checkpoint.restore as restore_module
from tpen.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    checkpoint_hashes,
    resolve_checkpoint_dir,
    restore_checkpoint,
    restore_checkpoint_with_events,
    save_checkpoint,
    stable_config_hash,
)
from tpen.checkpoint.artifact import prune_old_checkpoints, write_latest
from tpen.checkpoint.manifest import LEGACY_CHECKPOINT_KIND, LEGACY_CHECKPOINT_SCHEMA_VERSION
from tpen.nn import HookeOrbitalBasis


def _cfg(*, model_out: int = 2):
    return OmegaConf.create(
        {
            "model": {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": model_out},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"_target_": "tests.Trainer", "max_steps": 2},
            "sampler": {"_target_": "tests.Sampler", "n_steps": 5},
            "hamiltonian_terms": {"constant": {"_target_": "tests.ConstantHamiltonian"}},
            "run": {"run_id": "run", "dir": "/tmp/run"},
            "study": {"name": "unit", "config_id": "cfg"},
        }
    )


def _context(cfg=None):
    return SimpleNamespace(
        cfg=_cfg() if cfg is None else cfg,
        metadata=SimpleNamespace(
            run_id="run",
            device="cpu",
            dtype="float64",
            git_commit="deadbeef",
            git_branch="codex/checkpoint",
            dirty_worktree=False,
            command="pytest",
            extra={"slurm": {}},
        ),
        run_dir=Path("/tmp/run"),
    )


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self) -> dict[str, int]:
        return {"next_iteration": 3, "completed_updates": 3}

    def load_state_dict(self, state) -> None:
        self.loaded = dict(state)


class _Sampler:
    def __init__(self) -> None:
        self.loaded = None

    def mcmc_state_dict(self) -> dict[str, object]:
        return {"has_burned_in": True, "position": torch.ones(1)}

    def load_mcmc_state_dict(self, state) -> None:
        self.loaded = dict(state)


def _write_checkpoint(tmp_path: Path, model: torch.nn.Module | None = None, **kwargs) -> Path:
    model = torch.nn.Linear(3, 2).double() if model is None else model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=3,
        completed_updates=3,
        model=model,
        optimizer=optimizer,
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
        **kwargs,
    )


def _rewrite_manifest_as_v1(checkpoint_dir: Path) -> None:
    """Rewrite a written manifest into the retired v1 shape.

    v1 recorded one ambiguous ``step`` -- which was in fact the resume cursor
    -- under the pre-rename kind, and never recorded ``completed_updates``.
    Nothing writes that shape any more, so tests synthesize it to pin read-side
    acceptance for archived artifacts.
    """

    manifest_path = checkpoint_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["schema_version"] = LEGACY_CHECKPOINT_SCHEMA_VERSION
    data["kind"] = LEGACY_CHECKPOINT_KIND
    data["step"] = data.pop("next_iteration")
    data.pop("completed_updates")
    data["provenance"]["spenn_version"] = data["provenance"].pop("tpen_version")
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


def test_model_only_restore_loads_weights_into_configured_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trained = torch.nn.Linear(3, 2).double()
    root = _write_checkpoint(tmp_path, model=trained).parent

    torch.manual_seed(1)
    fresh = torch.nn.Linear(3, 2).double()
    assert not torch.equal(fresh.weight, trained.weight)

    report = restore_checkpoint(
        load={"path": str(root), "mode": "model_only", "strict": True},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.loaded_model is True
    assert report.loaded_optimizer is False
    assert report.loaded_sampler is False


def test_restore_runtime_device_check_normalizes_unindexed_cuda() -> None:
    assert restore_module._canonical_runtime_device("cuda") == torch.device("cuda:0")
    assert restore_module._canonical_runtime_device(torch.device("cuda:0")) == torch.device("cuda:0")
    assert restore_module._canonical_runtime_device("cpu") == torch.device("cpu")


def test_model_only_restore_emits_load_lifecycle_events(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    root = _write_checkpoint(tmp_path, model=trained).parent
    fresh = torch.nn.Linear(3, 2).double()
    events = []

    def emit(name, context, *, payload=None):
        events.append((name, payload))

    report = restore_checkpoint_with_events(
        load={"path": str(root), "mode": "model_only", "strict": True},
        model=fresh,
        context=_context(),
        emit=emit,
    )

    assert report.loaded_model is True
    assert [name for name, _ in events] == ["load_start", "load_success"]
    assert events[0][1] == {"path": str(root), "mode": "model_only", "strict": True}
    assert events[1][1] == {
        "path": str(root),
        "resolved_checkpoint_dir": str(root / "step_000003"),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "next_iteration": 3,
        "completed_updates": 3,
        "loaded_model": True,
        "loaded_optimizer": False,
        "loaded_trainer": False,
        "loaded_sampler": False,
        "loaded_rng": False,
    }


def test_restore_rejects_checkpoint_without_complete_marker(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)
    (checkpoint_dir / "COMPLETE").unlink()

    with pytest.raises(ValueError, match="COMPLETE"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only"},
            model=torch.nn.Linear(3, 2).double(),
            context=_context(),
        )


def test_restore_rejects_model_config_hash_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="model_config"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only"},
            model=torch.nn.Linear(3, 4).double(),
            context=_context(_cfg(model_out=4)),
        )


# axiswise_v1 must be selected explicitly since the D11 default flip; the
# deprecation warning it emits is expected and irrelevant to this contract.
@pytest.mark.filterwarnings("ignore:HookeOrbitalBasis basis_semantics:FutureWarning")
def test_restore_rejects_equal_width_legacy_and_product_basis_semantics(tmp_path: Path) -> None:
    """Equal parameter shapes cannot permit an axiswise/product checkpoint swap."""

    legacy_basis_kwargs = {
        "omega": 0.5,
        "spatial_dim": 2,
        "basis_semantics": "axiswise_v1",
        "max_shell": 2,
        "include_gaussian_factor": True,
        "include_spin": False,
    }
    product_basis_kwargs = {
        "omega": 0.5,
        "spatial_dim": 2,
        "basis_semantics": "product_v2",
        "truncation": "total_shell",
        "max_total_shell": 2,
        "include_gaussian_factor": True,
        "include_spin": False,
    }
    legacy_basis_config = {"_target_": "tpen.nn.HookeOrbitalBasis", **legacy_basis_kwargs}
    product_basis_config = {"_target_": "tpen.nn.HookeOrbitalBasis", **product_basis_kwargs}
    legacy_basis = HookeOrbitalBasis(**legacy_basis_kwargs)
    product_basis = HookeOrbitalBasis(**product_basis_kwargs)

    # d * (S + 1) == binomial(S + d, d) == 6 here, but channel meanings differ.
    assert legacy_basis.out_features == product_basis.out_features == 6
    legacy_config = OmegaConf.merge(
        _cfg(),
        {"model": {"in_features": legacy_basis.out_features, "basis": legacy_basis_config}},
    )
    product_config = OmegaConf.merge(
        _cfg(),
        {"model": {"in_features": product_basis.out_features, "basis": product_basis_config}},
    )
    assert "max_shell" not in product_config.model.basis
    trained = torch.nn.Linear(legacy_basis.out_features, 2).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=trained,
        context=_context(legacy_config),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )

    same_semantics = torch.nn.Linear(legacy_basis.out_features, 2).double()
    restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
        model=same_semantics,
        context=_context(legacy_config),
    )
    assert torch.equal(same_semantics.weight, trained.weight)

    different_semantics = torch.nn.Linear(product_basis.out_features, 2).double()
    before_restore = {name: value.detach().clone() for name, value in different_semantics.state_dict().items()}
    assert {name: value.shape for name, value in trained.state_dict().items()} == {
        name: value.shape for name, value in different_semantics.state_dict().items()
    }

    with pytest.raises(ValueError, match="model_config"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
            model=different_semantics,
            context=_context(product_config),
        )

    for name, value in different_semantics.state_dict().items():
        assert torch.equal(value, before_restore[name])


def test_model_only_does_not_require_train_resume_files(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=trained,
        context=_context(),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )

    fresh = torch.nn.Linear(3, 2).double()
    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only"},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.loaded_model is True
    assert report.loaded_optimizer is False


def test_train_resume_restores_all_train_state(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = _write_checkpoint(tmp_path, model=trained)
    fresh = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.Adam(fresh.parameters(), lr=0.01)
    trainer = _Trainer()
    sampler = _Sampler()

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "train_resume"},
        model=fresh,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert trainer.loaded == {"next_iteration": 3, "completed_updates": 3}
    assert sampler.loaded["has_burned_in"] is True
    assert report.loaded_optimizer is True
    assert report.loaded_trainer is True
    assert report.loaded_sampler is True
    assert report.loaded_rng is True


def test_train_resume_fails_when_required_file_is_missing(tmp_path: Path) -> None:
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=torch.nn.Linear(3, 2).double(),
        optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
        save_optimizer=False,
    )

    with pytest.raises(FileNotFoundError, match="optimizer"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "train_resume"},
            model=torch.nn.Linear(3, 2).double(),
            optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
            trainer=_Trainer(),
            sampler=_Sampler(),
            context=_context(),
        )


def test_restore_strict_load_fails_on_unexpected_keys(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)
    state = torch.load(checkpoint_dir / "model.pt", weights_only=False)
    state["ghost"] = torch.zeros(1)
    torch.save(state, checkpoint_dir / "model.pt")

    with pytest.raises(RuntimeError, match="ghost"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
            model=torch.nn.Linear(3, 2).double(),
            context=_context(),
        )


def test_v1_manifest_restores_model_only_without_a_completed_update_count(tmp_path: Path) -> None:
    """An archived v1 artifact still restores weights; it just cannot resume."""

    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = _write_checkpoint(tmp_path, model=trained)
    _rewrite_manifest_as_v1(checkpoint_dir)
    fresh = torch.nn.Linear(3, 2).double()

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only"},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.schema_version == LEGACY_CHECKPOINT_SCHEMA_VERSION
    # v1's lone `step` was the resume cursor, so it maps onto `next_iteration`.
    assert report.next_iteration == 3
    # v1 never recorded the update count, and there is no upgrade path.
    assert report.completed_updates is None
    assert report.to_dict()["completed_updates"] is None


def test_v1_manifest_is_refused_for_train_resume_at_the_schema_gate(tmp_path: Path) -> None:
    """Rejection names version and mode, and precedes any trainer load.

    Note what this pins and what it does not. The refusal is *not* because the
    restore path needs `manifest.completed_updates` -- ``train_resume`` reads
    trainer state from ``trainer.json``, never from the manifest. It is because
    `schema_version 1` spans two incompatible ``trainer.json`` key sets (B1
    renamed the trainer keys without bumping the manifest schema), so the
    version cannot prove the artifact is resumable. Hence this checkpoint --
    synthesized from a v2 write and therefore carrying *post*-B1 trainer keys
    that would in fact load -- is still refused.
    """

    checkpoint_dir = _write_checkpoint(tmp_path)
    _rewrite_manifest_as_v1(checkpoint_dir)
    trainer = _Trainer()

    with pytest.raises(
        ValueError,
        match=r"schema_version 1 is not supported for restore mode 'train_resume'",
    ):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "train_resume"},
            model=torch.nn.Linear(3, 2).double(),
            optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
            trainer=trainer,
            sampler=_Sampler(),
            context=_context(),
        )

    # The gate refuses before any component is touched. `load_state_dict`'s own
    # KeyError remains a backstop for a genuinely pre-B1 artifact, but it is
    # never the first failure for a v1 manifest.
    assert trainer.loaded is None


def test_restore_report_counters_for_mode_none() -> None:
    report = restore_checkpoint(
        load={"mode": "none"},
        model=torch.nn.Linear(3, 2).double(),
        context=_context(),
    )

    assert (report.next_iteration, report.completed_updates) == (None, None)
    assert report.to_dict()["next_iteration"] is None
    assert report.to_dict()["completed_updates"] is None


def test_restore_report_counters_for_model_only(tmp_path: Path) -> None:
    """Both counters come from the manifest and are reported separately."""

    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        # Diverged on purpose: one iteration skipped its optimizer update, so a
        # report that conflated the two counters could not pass this.
        next_iteration=4,
        completed_updates=3,
        model=torch.nn.Linear(3, 2).double(),
        context=_context(),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only"},
        model=torch.nn.Linear(3, 2).double(),
        context=_context(),
    )

    assert report.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert (report.next_iteration, report.completed_updates) == (4, 3)


def test_restore_report_counters_for_train_resume(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=4,
        completed_updates=3,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
    )

    fresh = torch.nn.Linear(3, 2).double()
    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "train_resume"},
        model=fresh,
        optimizer=torch.optim.Adam(fresh.parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
    )

    # `train_resume` admits only v2, so neither counter can be `None` here.
    assert (report.next_iteration, report.completed_updates) == (4, 3)
    assert report.to_dict()["next_iteration"] == 4
    assert report.to_dict()["completed_updates"] == 3


def test_prune_never_deletes_the_latest_pointer_target(tmp_path: Path) -> None:
    """Pruning spares `latest.json`'s target even outside the keep window."""

    root = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        model = torch.nn.Linear(3, 2).double()
        save_checkpoint(
            output_dir=root,
            next_iteration=step,
            completed_updates=step,
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            trainer=_Trainer(),
            sampler=_Sampler(),
            context=_context(),
        )
    # Point `latest.json` back at the oldest checkpoint, the state a run reaches
    # when its newest directories were written after the pointer it resumes from.
    write_latest(root, root / "step_000001", step=1, created_at_unix=0.0)

    prune_old_checkpoints(root, keep_last=1)

    # The pointer target survives *in addition* to the newest `keep_last`.
    assert sorted(path.name for path in root.glob("step_*")) == [
        "step_000001",
        "step_000003",
    ]
    assert resolve_checkpoint_dir(root) == root / "step_000001"


def test_prune_without_a_latest_pointer_still_trims(tmp_path: Path) -> None:
    """A missing pointer spares nothing extra and must not raise."""

    root = tmp_path / "checkpoints"
    for step in (1, 2):
        model = torch.nn.Linear(3, 2).double()
        save_checkpoint(
            output_dir=root,
            next_iteration=step,
            completed_updates=step,
            model=model,
            context=_context(),
            save_optimizer=False,
            save_trainer=False,
            save_sampler=False,
            save_rng=False,
        )
    (root / "latest.json").unlink()

    prune_old_checkpoints(root, keep_last=1)

    assert sorted(path.name for path in root.glob("step_*")) == ["step_000002"]


def test_stable_config_hash_is_canonical_and_strict() -> None:
    config = {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 2}
    reordered = {"out_features": 2, "_target_": "torch.nn.Linear", "in_features": 3}

    assert stable_config_hash(config) == stable_config_hash(reordered)
    assert stable_config_hash(config) == stable_config_hash(OmegaConf.create(config))
    assert stable_config_hash(config) != stable_config_hash({**config, "out_features": 4})

    with pytest.raises(TypeError, match="JSON/YAML-safe"):
        stable_config_hash({"path": Path("not-json-safe")})


def test_checkpoint_hashes_resolve_interpolations_and_track_components() -> None:
    cfg = OmegaConf.create(
        {
            "width": 4,
            "steps": 5,
            "omega": 0.5,
            "run": {"run_id": "a", "dir": "/tmp/a"},
            "model": {"channels": "${width}"},
            "sampler": {"n_steps": "${steps}"},
            "hamiltonian_terms": {"trap": {"omega": "${omega}"}},
        }
    )
    same = OmegaConf.create(
        {
            "width": 4,
            "steps": 5,
            "omega": 0.5,
            "run": {"run_id": "b", "dir": "/tmp/b"},
            "model": {"channels": 4},
            "sampler": {"n_steps": 5},
            "hamiltonian_terms": {"trap": {"omega": 0.5}},
        }
    )
    changed_sampler = OmegaConf.merge(same, {"sampler": {"n_steps": 6}})
    changed_hamiltonian = OmegaConf.merge(same, {"hamiltonian_terms": {"trap": {"omega": 0.7}}})

    hashes = checkpoint_hashes(cfg)

    assert hashes["model_config"] == checkpoint_hashes(same)["model_config"]
    assert hashes["resolved_config"] == checkpoint_hashes(same)["resolved_config"]
    assert hashes["sampler_config"] != checkpoint_hashes(changed_sampler)["sampler_config"]
    assert hashes["hamiltonian_config"] != checkpoint_hashes(changed_hamiltonian)["hamiltonian_config"]
