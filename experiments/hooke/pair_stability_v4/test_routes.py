"""Tests for V4-0 root ownership and the pinned legacy route boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]

while str(STUDY_DIR) in sys.path:
    sys.path.remove(str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR))

for module_name in ("roots", "routes"):
    sys.modules.pop(module_name, None)

import roots  # noqa: E402
import routes  # noqa: E402

EXPECTED_INPUTS = {
    "screen_plan": (),
    "screen_train": ("grid",),
    "screen_eval": ("grid", "train"),
    "screen_collect": ("grid",),
    "select": ("collection",),
    "confirm_plan": ("grid", "selection"),
    "confirm_train": ("final_grid",),
    "confirm_eval": ("final_grid", "final_train"),
    "confirm_collect": ("final_grid", "final_eval"),
    "report": ("final_collect",),
}
EXPECTED_CONFIGS = {
    "screen_plan": ("smoke", "train"),
    "screen_train": (),
    "screen_eval": ("validation",),
    "screen_collect": (),
    "select": (),
    "confirm_plan": ("train", "validation"),
    "confirm_train": ("train",),
    "confirm_eval": ("validation",),
    "confirm_collect": (),
    "report": (),
}
EXPECTED_SCRIPTS = {
    "screen_plan": "plan.py",
    "screen_train": "train.py",
    "screen_eval": "validate.py",
    "screen_collect": "collect.py",
    "select": "select_champions.py",
    "confirm_plan": "final_plan.py",
    "confirm_train": "final_train.py",
    "confirm_eval": "final_eval.py",
    "confirm_collect": "final_collect.py",
    "report": "final_report.py",
}


def test_v4_configs_are_semantic_copies_and_legacy_source_is_pinned() -> None:
    """Scientific config drift and changes to the v3 closure fail closed."""

    assert routes.verify_v4_config_copies(REPO_ROOT) == ()
    assert routes.verify_legacy_source_manifest(REPO_ROOT) == ()

    source_receipt = routes.legacy_source_receipt(REPO_ROOT)
    assert len(source_receipt["files"]) == 25
    assert len(source_receipt["closure_sha256"]) == 64
    assert {
        row["path"] for row in source_receipt["files"]
    } >= {
        "experiments/hooke/pair_stability_v3/plan.py",
        "experiments/hooke/pair_stability_v3/utils/io.py",
        "experiments/hooke/pair_stability_v3/configs/smoke.yaml",
    }
    for name in ("smoke.yaml", "pair_stability.yaml", "pair_validation.yaml"):
        v3_text = (routes.V3_STUDY_DIR / "configs" / name).read_text()
        expected = (
            v3_text.replace("pair_stability_v3", "pair_stability_v4")
            .replace("V3", "V4")
        )
        assert (STUDY_DIR / "configs" / name).read_text() == expected


def test_config_parity_rejects_mapping_reorder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordered transformed text, not unordered YAML equality, defines parity."""

    v3_study = tmp_path / "v3"
    v4_study = tmp_path / "v4"
    (v3_study / "configs").mkdir(parents=True)
    (v4_study / "configs").mkdir(parents=True)
    for name in ("smoke.yaml", "pair_stability.yaml", "pair_validation.yaml"):
        source = routes.V3_STUDY_DIR / "configs" / name
        shutil.copyfile(source, v3_study / "configs" / name)
        transformed = (
            source.read_text()
            .replace("pair_stability_v3", "pair_stability_v4")
            .replace("V3", "V4")
        )
        (v4_study / "configs" / name).write_text(transformed)
    smoke_path = v4_study / "configs" / "smoke.yaml"
    smoke = smoke_path.read_text()
    smoke = smoke.replace(
        "study: pair_stability_v4\nconfig:",
        "config:",
        1,
    ).replace(
        "validation_config:",
        "study: pair_stability_v4\nvalidation_config:",
        1,
    )
    smoke_path.write_text(smoke)
    monkeypatch.setattr(routes, "V3_STUDY_DIR", v3_study)
    monkeypatch.setattr(routes, "STUDY_DIR", v4_study)

    assert any(
        "text/order" in error
        for error in routes.verify_v4_config_copies(tmp_path)
    )


def test_legacy_source_digest_mismatch_is_reported(tmp_path: Path) -> None:
    """The dispatcher reports source drift before executing a legacy stage."""

    manifest = json.loads(routes.SOURCE_MANIFEST_PATH.read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    changed = tmp_path / "legacy-source.json"
    changed.write_text(json.dumps(manifest))

    errors = routes.verify_legacy_source_manifest(
        REPO_ROOT,
        manifest_path=changed,
    )

    assert len(errors) == 1
    assert errors[0].startswith("legacy source digest mismatch:")


def test_legacy_source_manifest_requires_exact_explicit_closure(
    tmp_path: Path,
) -> None:
    """Removing a required file or adding an unrelated file is rejected."""

    manifest = json.loads(routes.SOURCE_MANIFEST_PATH.read_text())
    manifest["files"].pop()
    manifest["files"].append(
        {
            "path": "experiments/hooke/pair_stability_v3/README.md",
            "sha256": "0" * 64,
        }
    )
    changed = tmp_path / "legacy-source.json"
    changed.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="legacy source closure mismatch"):
        routes.load_legacy_source_manifest(changed)


def test_route_manifest_is_closed_and_covers_each_numbered_stage() -> None:
    """Exactly ten declared roles map one-to-one onto stages 00 through 09."""

    loaded = routes.load_routes()

    assert tuple(loaded) == tuple(routes.ROLE_TO_STAGE)
    assert {
        role: route.required_input_attempts for role, route in loaded.items()
    } == EXPECTED_INPUTS
    assert {
        role: route.required_configs for role, route in loaded.items()
    } == EXPECTED_CONFIGS
    assert {
        role: Path(route.legacy_script).name for role, route in loaded.items()
    } == EXPECTED_SCRIPTS
    assert {
        route.physical_stage for route in loaded.values()
    } == {
        f"{index:02d}_{suffix}"
        for index, suffix in enumerate(
            (
                "grid",
                "train",
                "validation",
                "collect",
                "select",
                "final_grid",
                "final_train",
                "final_eval",
                "final_collect",
                "final_report",
            )
        )
    }
    assert {
        role for role, route in loaded.items() if route.kind == "fanout"
    } == {
        "screen_train",
        "screen_eval",
        "confirm_train",
        "confirm_eval",
    }
    assert all("--wait-job" not in route.arguments for route in loaded.values())
    assert all("--smoke" not in route.arguments for route in loaded.values())


def test_routes_render_complete_argv_without_passthrough(tmp_path: Path) -> None:
    """Every route renders a complete command from its exact typed inputs."""

    results_root = tmp_path / "v4-results"
    attempts = {
        "grid": "grid-a",
        "train": "grid-a",
        "collection": "collect-a",
        "selection": "select-a",
        "final_grid": "final-grid-a",
        "final_train": "final-train-a",
        "final_eval": "final-eval-a",
        "final_collect": "final-collect-a",
    }
    configs = {
        "smoke": STUDY_DIR / "configs" / "smoke.yaml",
        "train": STUDY_DIR / "configs" / "pair_stability.yaml",
        "validation": STUDY_DIR / "configs" / "pair_validation.yaml",
    }
    loaded = routes.load_routes()

    for role, route in loaded.items():
        role_inputs = {name: attempts[name] for name in EXPECTED_INPUTS[role]}
        role_configs = {name: configs[name] for name in EXPECTED_CONFIGS[role]}
        output_attempt = (
            attempts["grid"]
            if route.output_attempt_rule == "grid_input"
            else f"{role}-out"
        )
        argv = routes.render_legacy_argv(
            route,
            results_root=results_root,
            output_attempt=output_attempt,
            input_attempts=role_inputs,
            config_paths=role_configs,
            repo_root=REPO_ROOT,
        )

        assert argv == _golden_argv(
            role,
            results_root=results_root,
            output_attempt=output_attempt,
            attempts=attempts,
            configs=configs,
        )


def test_route_rejects_unknown_fields_partial_tokens_and_duplicate_keys() -> None:
    """The route schema cannot become an arbitrary legacy argv surface."""

    base = {
        "logical_role": "screen_plan",
        "physical_stage": "00_grid",
        "legacy_script": "experiments/hooke/pair_stability_v3/plan.py",
        "kind": "local",
        "required_input_attempts": [],
        "required_configs": [],
        "output_attempt_rule": "explicit",
        "arguments": ["--root", "{results_root}", "{output_attempt}"],
    }

    with pytest.raises(ValueError, match="unknown=.*passthrough"):
        routes.LegacyStageRoute.from_dict({**base, "passthrough": True})

    partial = {**base, "arguments": ["--root={results_root}", "{output_attempt}"]}
    with pytest.raises(ValueError, match="partial or unknown"):
        routes.LegacyStageRoute.from_dict(partial)

    duplicate = {
        **base,
        "required_configs": ["bad-key", "bad-key"],
        "arguments": [
            "{results_root}",
            "{output_attempt}",
            "{config:bad-key}",
            "{config:bad-key}",
        ],
    }
    with pytest.raises(ValueError, match="duplicates"):
        routes.LegacyStageRoute.from_dict(duplicate)

    malformed = {
        **base,
        "required_configs": ["bad:key"],
        "arguments": [
            "{results_root}",
            "{output_attempt}",
            "{config:bad:key}",
        ],
    }
    with pytest.raises(ValueError, match="invalid route key"):
        routes.LegacyStageRoute.from_dict(malformed)


def test_render_rejects_missing_inputs_and_screen_train_attempt_split(
    tmp_path: Path,
) -> None:
    """Declared upstream lineages and the legacy train coupling stay explicit."""

    loaded = routes.load_routes()
    with pytest.raises(ValueError, match="input attempts mismatch"):
        routes.render_legacy_argv(
            loaded["screen_eval"],
            results_root=tmp_path / "results",
            output_attempt="eval",
            input_attempts={"grid": "grid"},
            config_paths={
                "validation": STUDY_DIR / "configs" / "pair_validation.yaml"
            },
            repo_root=REPO_ROOT,
        )

    with pytest.raises(ValueError, match="must equal its grid"):
        routes.render_legacy_argv(
            loaded["screen_train"],
            results_root=tmp_path / "results",
            output_attempt="different-train",
            input_attempts={"grid": "grid"},
            config_paths={},
            repo_root=REPO_ROOT,
        )


def test_root_sentinel_guards_lineage_purpose_and_artifact_containment(
    tmp_path: Path,
) -> None:
    """One absolute root belongs to one lineage and one declared purpose."""

    requested = (tmp_path / "candidate").absolute()
    root = roots.initialize_root(requested, lineage_id="lineage-a")

    assert roots.require_v4_root(root, lineage_id="lineage-a") == root
    assert roots.root_metadata(root)["purpose"] == roots.PURPOSE_EXPERIMENT
    assert roots.require_beneath_root("00_grid/attempt", root) == (
        root / "00_grid" / "attempt"
    )
    with pytest.raises(ValueError, match="belongs to lineage"):
        roots.require_v4_root(root, lineage_id="lineage-b")
    with pytest.raises(ValueError, match="escapes"):
        roots.require_beneath_root(root, root)
    with pytest.raises(ValueError, match="traversal"):
        roots.require_beneath_root("../v3", root)
    with pytest.raises(ValueError, match="reuse initialized"):
        roots.initialize_root(requested, lineage_id="lineage-a")


def test_root_rejects_nonempty_unsentinelled_and_symlink_escape(
    tmp_path: Path,
) -> None:
    """Existing data and links cannot silently cross the V4 ownership boundary."""

    occupied = (tmp_path / "occupied").absolute()
    occupied.mkdir()
    (occupied / "data.json").write_text("{}")
    with pytest.raises(ValueError, match="nonempty directory"):
        roots.initialize_root(occupied, lineage_id="lineage")

    root = roots.initialize_root(
        (tmp_path / "guarded").absolute(),
        lineage_id="lineage",
        purpose=roots.PURPOSE_OWNERSHIP_AUDIT,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    assert roots.validate_root_links(root) == ("escape",)
    with pytest.raises(ValueError, match="escapes"):
        roots.require_beneath_root(root / "escape" / "artifact.json", root)


def test_root_rejects_wrong_in_repo_namespace_and_symlink_sentinel(
    tmp_path: Path,
) -> None:
    """Repository roots are v4-results-only and sentinels must be regular."""

    with pytest.raises(ValueError, match="in-repository"):
        roots._validate_requested_root(REPO_ROOT / "unowned-results")
    assert roots._validate_requested_root(
        STUDY_DIR / "results" / "lineage"
    ) == STUDY_DIR / "results" / "lineage"

    root = (tmp_path / "sentinel-link").absolute()
    root.mkdir()
    target = tmp_path / "sentinel.json"
    target.write_text("{}")
    (root / roots.ROOT_SENTINEL).symlink_to(target)
    with pytest.raises(ValueError, match="sentinel"):
        roots.require_v4_root(root)


def test_runtime_receipt_fails_closed_on_git_error_or_empty_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance cannot silently degrade to empty source or commit fields."""

    def failed_git(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=2, stdout="", stderr="git failed")

    monkeypatch.setattr(routes.subprocess, "run", failed_git)
    with pytest.raises(RuntimeError, match="git ls-files failed"):
        routes.runtime_source_receipt(REPO_ROOT)

    def empty_git(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(routes.subprocess, "run", empty_git)
    with pytest.raises(RuntimeError, match="closure is empty"):
        routes.runtime_source_receipt(REPO_ROOT)


def test_fresh_routes_import_does_not_import_v3_modules() -> None:
    """Importing the bridge contract has no direct legacy-module dependency."""

    script = (
        "import json, sys; "
        f"sys.path.insert(0, {str(STUDY_DIR)!r}); "
        "import routes; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name in {'plan','train','validate','utils','launch'})))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def _golden_argv(
    role: str,
    *,
    results_root: Path,
    output_attempt: str,
    attempts: dict[str, str],
    configs: dict[str, Path],
) -> tuple[str, ...]:
    root = str(results_root)
    repo = str(REPO_ROOT)
    smoke = str(configs["smoke"].resolve())
    train = str(configs["train"].resolve())
    validation = str(configs["validation"].resolve())
    suffixes = {
        "screen_plan": (
            "--grid", smoke, "--config", train, "--results-root", root,
            "--attempt-id", output_attempt, "--timezone", "America/New_York",
            "--blind", "--blind-seed", "811", "--python", "python",
        ),
        "screen_train": (
            "--results-root", root, "--grid-attempt-id", attempts["grid"],
            "--repo-root", repo, "--backend", "submitit", "--device", "cuda",
            "--chunk-size", "32", "--slurm-cpus", "4", "--slurm-partition",
            "gpu_test", "--slurm-mem-per-cpu-gb", "8", "--slurm-timeout-min", "60",
        ),
        "screen_eval": (
            "--results-root", root, "--grid-attempt-id", attempts["grid"],
            "--train-attempt-id", attempts["train"], "--attempt-id",
            output_attempt, "--config", validation, "--repo-root", repo,
            "--backend", "submitit", "--device", "cuda", "--chunk-size", "32",
            "--slurm-cpus", "4", "--slurm-partition", "gpu_test",
            "--slurm-mem-per-cpu-gb", "8", "--slurm-timeout-min", "120",
        ),
        "screen_collect": (
            "--results-root", root, "--grid-attempt-id", attempts["grid"],
            "--attempt-id", output_attempt,
        ),
        "select": (
            "--results-root", root, "--collection-attempt-id",
            attempts["collection"], "--attempt-id", output_attempt,
        ),
        "confirm_plan": (
            "--results-root", root, "--selection-attempt-id",
            attempts["selection"], "--attempt-id", output_attempt,
            "--train-config", train, "--eval-config", validation,
            "--replicates", "1",
        ),
        "confirm_train": (
            "--results-root", root, "--final-grid-attempt-id",
            attempts["final_grid"], "--attempt-id", output_attempt, "--config",
            train, "--repo-root", repo, "--backend", "submitit", "--device",
            "cuda", "--chunk-size", "8", "--slurm-cpus", "4",
            "--slurm-cuda-partition", "gpu_test", "--slurm-mem-per-cpu-gb",
            "8", "--slurm-cuda-timeout-min", "60",
        ),
        "confirm_eval": (
            "--results-root", root, "--final-grid-attempt-id",
            attempts["final_grid"], "--final-train-attempt-id",
            attempts["final_train"], "--attempt-id", output_attempt, "--config",
            validation, "--repo-root", repo, "--backend", "submitit", "--device",
            "cuda", "--chunk-size", "8", "--slurm-cpus", "4",
            "--slurm-partition", "gpu_test", "--slurm-mem-per-cpu-gb", "8",
            "--slurm-timeout-min", "120",
        ),
        "confirm_collect": (
            "--results-root", root, "--final-grid-attempt-id",
            attempts["final_grid"], "--final-eval-attempt-id",
            attempts["final_eval"], "--attempt-id", output_attempt,
        ),
        "report": (
            "--results-root", root, "--final-collect-attempt-id",
            attempts["final_collect"], "--attempt-id", output_attempt,
        ),
    }
    script = (
        REPO_ROOT
        / "experiments"
        / "hooke"
        / "pair_stability_v3"
        / EXPECTED_SCRIPTS[role]
    )
    return (sys.executable, "-B", str(script), *suffixes[role])
