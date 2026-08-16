"""Plan-stage behavior: the grid is explicit, ordered, and reproducible.

Each test guards a way the study could be submitted while describing a
different experiment than the one asked for:

determinism
    Re-planning the same grid must produce the same row ids, the same order,
    and the same plan hash. If it does not, "the plan did not change" is
    unverifiable and two attempts cannot be compared.

no implicit defaults
    A missing or unknown grid key is an error. A silently defaulted seed count
    or wall time produces a green run answering a different question.

independence
    Evaluation chain seeds may not collide with training seeds, or a "fresh"
    chain is a replay of the sampler that trained the model.

no resume
    `production-grid-v0` forbids relying on restart/resume, so a row that asks
    for one, or that is sized past its partition's measured wall ceiling, fails
    at plan time rather than at the wall.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
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


plan = _load_study_module("plan")
strata = _load_study_module("strata")


GRID: dict[str, Any] = {
    "study": "tpen_he_v1",
    "train_config": "experiments/atomistic/he-v1/configs/train.yaml",
    "eval_config": "experiments/atomistic/he-v1/configs/eval.yaml",
    "seeds": [0, 1, 2],
    "checkpoint_steps": [10000, 30000, 100000],
    "eval_chains": 4,
    "eval_chain_seed_base": 900000,
    "train_resources": {
        "partition": "kozinsky_gpu",
        "stratum": "a100",
        "timeout_min": 8640,
        "cpus": 16,
        "mem_gb": 128,
        "gpus": 1,
    },
    "eval_resources": {
        "partition": "seas_gpu",
        "stratum": "h200",
        "timeout_min": 240,
        "cpus": 8,
        "mem_gb": 64,
        "gpus": 1,
    },
    "gate_spec": {},
    "seed_stages": [[0], [1, 2]],
    "convergence_assessment": {
        "method": "windowed_means_sign_test",
        "n_windows": 20,
        "n_trailing_windows": 8,
        "window_width_min_tau_multiple": 5.0,
        "on_inadequate": "report_only",
        "may_reselect": False,
    },
}


def _grid(**overrides: Any) -> dict[str, Any]:
    payload = copy.deepcopy(GRID)
    payload.update(overrides)
    return payload


def test_production_grid_shape_is_three_seeds_times_checkpoints_times_chains() -> None:
    """The manifest expresses `production-grid-v0` exactly: 3 x 3 x 4 plus 3 trainings."""

    rows = plan.expand_rows(plan.validate_grid_config(_grid()))
    train_rows = [row for row in rows if row["kind"] == "train"]
    eval_rows = [row for row in rows if row["kind"] == "eval"]
    assert len(train_rows) == 3
    assert len(eval_rows) == 3 * 3 * 4
    assert [row["index"] for row in rows] == list(range(len(rows)))


def test_row_ids_are_stable_and_unique() -> None:
    """Row ids are a pure function of coordinates, so reruns land beside reruns."""

    rows = plan.expand_rows(plan.validate_grid_config(_grid()))
    row_ids = [row["row_id"] for row in rows]
    assert len(set(row_ids)) == len(row_ids)
    assert plan.train_row_id(1) == "train-seed0001"
    assert (
        plan.eval_row_id(seed=1, checkpoint_step=30000, chain=2)
        == "eval-seed0001-step000030000-chain02"
    )
    assert row_ids[0] == "train-seed0000"


def test_plan_hash_is_deterministic_and_content_addressed() -> None:
    """The same grid hashes identically; a changed grid does not."""

    config = plan.validate_grid_config(_grid())
    first = plan.plan_hash(plan.expand_rows(config))
    second = plan.plan_hash(plan.expand_rows(plan.validate_grid_config(_grid())))
    assert first == second
    changed = plan.plan_hash(plan.expand_rows(plan.validate_grid_config(_grid(seeds=[0, 1, 7], seed_stages=[[0], [1, 7]]))))
    assert changed != first


def test_manifest_hash_excludes_timestamps() -> None:
    """Two attempts of one grid differ in attempt id but not in plan hash."""

    config = plan.validate_grid_config(_grid())
    rows = plan.expand_rows(config)
    first = plan.build_manifest(
        config=config,
        rows=rows,
        attempt_id="20260815T120000",
        results_root="/tmp/results",
        grid_config_path=None,
        grid_config_sha256=None,
        created_at="2026-08-15T12:00:00-04:00",
    )
    second = plan.build_manifest(
        config=config,
        rows=rows,
        attempt_id="20260815T130000",
        results_root="/tmp/results",
        grid_config_path=None,
        grid_config_sha256=None,
        created_at="2026-08-15T13:00:00-04:00",
    )
    assert first["plan_hash"] == second["plan_hash"]
    assert first["attempt_id"] != second["attempt_id"]


@pytest.mark.parametrize("missing", sorted(plan.GRID_KEYS))
def test_every_grid_key_is_required(missing: str) -> None:
    """No grid key is optional: an omitted one is an error, not a default."""

    payload = _grid()
    payload.pop(missing)
    with pytest.raises(plan.PlanError, match="missing required keys"):
        plan.validate_grid_config(payload)


def test_unknown_grid_key_is_rejected() -> None:
    """A mistyped key would otherwise be ignored, disabling what it meant to set."""

    with pytest.raises(plan.PlanError, match="unknown keys"):
        plan.validate_grid_config(_grid(eval_chain=4))


def test_empty_gate_spec_is_accepted_but_recorded_as_undeclared() -> None:
    """An empty spec is a legitimate pre-H-F1 state, and the manifest says so."""

    config = plan.validate_grid_config(_grid())
    manifest = plan.build_manifest(
        config=config,
        rows=plan.expand_rows(config),
        attempt_id="a",
        results_root="/tmp/results",
        grid_config_path=None,
        grid_config_sha256=None,
        created_at="2026-08-15T12:00:00-04:00",
    )
    assert manifest["gate_spec"] == {}
    assert manifest["gate_spec_declared"] is False


def test_gate_spec_must_be_a_mapping() -> None:
    """``gate_spec: null`` is not the same statement as ``gate_spec: {}``."""

    with pytest.raises(plan.PlanError, match="must be a mapping"):
        plan.validate_grid_config(_grid(gate_spec=None))


def test_chain_seeds_may_not_collide_with_training_seeds() -> None:
    """A colliding chain seed would replay the sampler that trained the model."""

    with pytest.raises(plan.PlanError, match="collide with training seeds"):
        plan.validate_grid_config(_grid(seeds=[0, 1, 2], eval_chain_seed_base=0))


def test_chain_seeds_are_distinct_across_seed_checkpoint_and_chain() -> None:
    """Independent chains need independent streams."""

    rows = plan.expand_rows(plan.validate_grid_config(_grid()))
    chain_seeds = [row["chain_seed"] for row in rows if row["kind"] == "eval"]
    assert len(set(chain_seeds)) == len(chain_seeds)


def test_wall_time_past_the_partition_ceiling_is_rejected() -> None:
    """seas_gpu ends at 2 days and rows may not resume, so 3 days is a planning defect."""

    resources = dict(GRID["eval_resources"], timeout_min=3 * 24 * 60)
    with pytest.raises(plan.PlanError, match="exceeds the measured seas_gpu ceiling"):
        plan.expand_rows(plan.validate_grid_config(_grid(eval_resources=resources)))


def test_wall_time_at_the_partition_ceiling_is_accepted() -> None:
    """The ceiling itself is submittable; only past it is not."""

    resources = dict(GRID["eval_resources"], timeout_min=2 * 24 * 60)
    rows = plan.expand_rows(plan.validate_grid_config(_grid(eval_resources=resources)))
    assert rows[1]["resources"]["timeout_min"] == 2880


def test_stratum_absent_from_the_partition_is_rejected() -> None:
    """kozinsky_gpu carries no H200; pinning one there would never schedule."""

    resources = dict(GRID["eval_resources"], partition="kozinsky_gpu", stratum="h200")
    with pytest.raises(plan.PlanError, match="not available on partition"):
        plan.expand_rows(plan.validate_grid_config(_grid(eval_resources=resources)))


def test_smoke_partition_is_rejected_for_grid_rows() -> None:
    """`production-grid-v0` forbids production rows on gpu_test."""

    resources = dict(GRID["eval_resources"], partition="gpu_test", stratum="a100")
    with pytest.raises(plan.PlanError, match="smoke/pilot target"):
        plan.expand_rows(plan.validate_grid_config(_grid(eval_resources=resources)))


def test_every_row_carries_a_resolved_constraint() -> None:
    """A GPU row without a pin is the failure this study refuses to ship."""

    rows = plan.expand_rows(plan.validate_grid_config(_grid()))
    for row in rows:
        stratum = row["resources"]["stratum"]
        assert row["resources"]["constraint"] == strata.constraint_for(stratum)


@pytest.mark.parametrize(
    "override",
    ["trainer.resume=true", "trainer.restart_from=step_10", "runner.requeue=true"],
)
def test_resume_flavored_override_is_rejected(override: str) -> None:
    """No row may ask to resume while resume semantics are undemonstrated."""

    rows = plan.expand_rows(plan.validate_grid_config(_grid()))
    poisoned = dict(rows[0])
    poisoned["overrides"] = [*poisoned["overrides"], override]
    with pytest.raises(plan.PlanError, match="restart/resume"):
        plan.reject_resume_overrides(poisoned)


def test_planned_rows_carry_no_resume_override() -> None:
    """The rows this stage actually emits pass their own check."""

    for row in plan.expand_rows(plan.validate_grid_config(_grid())):
        plan.reject_resume_overrides(row)


def test_eval_rows_depend_on_their_training_row_and_name_its_checkpoint() -> None:
    """An eval row states which model it evaluates, by row and by directory name."""

    rows = plan.expand_rows(plan.validate_grid_config(_grid()))
    eval_row = next(row for row in rows if row["kind"] == "eval")
    assert eval_row["depends_on"] == [plan.train_row_id(eval_row["seed"])]
    assert eval_row["checkpoint_dir_name"] == plan.checkpoint_dir_name(eval_row["checkpoint_step"])


def test_seed_staging_is_declared_but_not_executed() -> None:
    """`seed_stages` is a predeclared PROCEDURE; assert only what is true of it.

    The grid declares ``[[0], [1, 2]]``, but nothing sequences the seeds: every
    training row carries an empty ``depends_on`` whatever stage its seed sits
    in, and `launch.py` never reads the key, so ``--submit`` fires all three at
    once. This test pins that reality so the docstrings in `plan.py` and the
    comment in `production_grid.yaml` cannot drift into implying execution --
    and so that whoever later WIRES the staging is forced to update the prose
    in the same change, because this test will go red when they do.
    """

    config = plan.validate_grid_config(_grid(seeds=[0, 1, 2], seed_stages=[[0], [1, 2]]))
    rows = plan.expand_rows(config)
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert len(train_rows) == 3
    # Not "the stage-2 rows depend on stage 1" -- no row depends on any other.
    assert all(row["depends_on"] == [] for row in train_rows)
    # The declaration still survives into the manifest, which is what makes it
    # a predeclaration rather than a comment.
    assert config["seed_stages"] == [[0], [1, 2]]
    launch_source = (STUDY_DIR / "launch.py").read_text(encoding="utf-8")
    assert "seed_stages" not in launch_source


def test_checkpoint_dir_name_matches_the_writer_format() -> None:
    """Pins ``step_%06d``; drift would point every eval row at nothing."""

    assert plan.checkpoint_dir_name(5) == "step_000005"
    assert plan.checkpoint_dir_name(100000) == "step_100000"


def test_training_rows_run_to_the_largest_retained_checkpoint() -> None:
    """The training row is sized by the predeclared checkpoints, not by a default."""

    rows = plan.expand_rows(plan.validate_grid_config(_grid()))
    train_row = rows[0]
    assert "trainer.max_steps=100000" in train_row["overrides"]
    assert train_row["retained_checkpoint_steps"] == [10000, 30000, 100000]


def test_written_plan_round_trips_and_records_latest(tmp_path: Path) -> None:
    """A written plan reads back identically and is discoverable by pointer."""

    config = plan.validate_grid_config(_grid())
    rows = plan.expand_rows(config)
    manifest = plan.build_manifest(
        config=config,
        rows=rows,
        attempt_id="20260815T120000",
        results_root=str(tmp_path),
        grid_config_path=None,
        grid_config_sha256=None,
        created_at="2026-08-15T12:00:00-04:00",
    )
    plan.write_plan(manifest, results_root=tmp_path)
    reread = plan.read_manifest(tmp_path, "20260815T120000")
    assert reread["plan_hash"] == manifest["plan_hash"]
    assert len(reread["rows"]) == len(rows)
    assert plan.row_by_id(reread, "train-seed0000")["kind"] == "train"


def test_rows_csv_renders_inapplicable_fields_as_absent(tmp_path: Path) -> None:
    """A training row has no chain; the cell says ``absent``, never blank or 0."""

    config = plan.validate_grid_config(_grid())
    manifest = plan.build_manifest(
        config=config,
        rows=plan.expand_rows(config),
        attempt_id="20260815T120000",
        results_root=str(tmp_path),
        grid_config_path=None,
        grid_config_sha256=None,
        created_at="2026-08-15T12:00:00-04:00",
    )
    directory = plan.write_plan(manifest, results_root=tmp_path)
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    train_row = next(row for row in table if row["kind"] == "train")
    assert train_row["chain"] == "absent"
    assert train_row["checkpoint_step"] == "absent"
    assert train_row["depends_on"] == "none"
    eval_row = next(row for row in table if row["kind"] == "eval")
    assert eval_row["constraint"] == "h200"


def test_unknown_plan_attempt_is_not_guessed(tmp_path: Path) -> None:
    """With no pointer and no explicit id, the stage refuses rather than picking one."""

    layout = _load_study_module("layout")
    with pytest.raises(FileNotFoundError):
        layout.resolve_attempt_id(tmp_path, layout.STAGE_PLAN, None)
