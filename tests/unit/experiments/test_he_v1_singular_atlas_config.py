"""Static contracts for the append-only He-v1 P3 atlas configuration."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "experiments" / "atomistic" / "he-v1" / "configs" / "eval.yaml"

ATLAS_TASKS = [
    "he_en_numerical_atlas",
    "he_ee_ideal_vs_executed_numerical_atlas",
    "he_one_electron_tail_atlas",
    "he_center_of_mass_tail_atlas",
    "he_angular_shell_atlas",
]


def _load() -> dict:
    with EVAL.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def test_p3_atlas_tasks_are_one_append_only_evaluator_block() -> None:
    config = _load()
    references = config["evaluator"]["tasks"]

    assert references[-5:] == [f"${{evaluation_tasks.{name}}}" for name in ATLAS_TASKS]
    assert list(config["evaluation_tasks"])[-5:] == ATLAS_TASKS
    text = EVAL.read_text(encoding="utf-8")
    assert text.count("P3 singular/tail atlas block") == 2


def test_p3_atlas_generators_cover_directions_spectator_com_and_shells() -> None:
    tasks = _load()["evaluation_tasks"]
    en = tasks["he_en_numerical_atlas"]["generator"]
    ee = tasks["he_ee_ideal_vs_executed_numerical_atlas"]["generator"]
    one = tasks["he_one_electron_tail_atlas"]["generator"]
    com = tasks["he_center_of_mass_tail_atlas"]["generator"]
    shell = tasks["he_angular_shell_atlas"]["generator"]

    assert en["_target_"] == "tpen.evaluation.generators.HeliumElectronNucleusApproachGenerator"
    assert ee["_target_"] == "tpen.evaluation.generators.HeliumElectronElectronApproachGenerator"
    assert one["_target_"] == "tpen.evaluation.generators.HeliumOneElectronEscapeGenerator"
    assert com["_target_"] == "tpen.evaluation.generators.HeliumCenterOfMassEscapeGenerator"
    assert shell["_target_"] == "tpen.evaluation.generators.HeliumAngularShellGenerator"
    for generator in (en, ee, one, com, shell):
        assert generator["atoms"] == "${atoms}"
        assert generator["directions"] == [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    assert en["spectator_position"] == [0.0, 0.0, -1.0]
    assert one["spectator_position"] == [0.0, 0.0, -1.0]
    assert shell["spectator_position"] == [0.0, 0.0, -1.0]
    assert ee["center_of_mass"] == [0.0, 0.0, 0.0]
    assert com["relative_positions"] == [[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]


def test_p3_atlas_calculators_bind_restored_factors_and_registry_terms() -> None:
    tasks = _load()["evaluation_tasks"]

    for task_name in ATLAS_TASKS:
        calculators = tasks[task_name]["calculators"]
        assert len(calculators) == 1
        calculator = calculators[0]
        assert calculator["_target_"] == "tpen.evaluation.calculators.HeliumAtlasCalculator"
        assert calculator["hamiltonian_terms"] == "${hamiltonian_terms}"
        assert calculator["factor_indices"] == {
            "executed_smoothed_ee_factor": 0,
            "executed_electron_nucleus_factor": 1,
        }
        assert calculator["chunk_size"] == 8


def test_p3_curvature_windows_are_predeclared_nested_and_not_kato_targets() -> None:
    tasks = _load()["evaluation_tasks"]

    curvature = [
        summary
        for task_name in ATLAS_TASKS[:2]
        for summary in tasks[task_name]["summaries"]
        if summary["_target_"] == "tpen.evaluation.summaries.HeliumCurvatureSummary"
    ]
    assert len(curvature) == 4
    for summary in curvature:
        bounds = [float(value) for value in summary["windows"].values()]
        assert len(bounds) >= 2
        assert all(right > left > 0.0 for left, right in zip(bounds, bounds[1:]))
        assert "kato" not in str(summary).lower()


def test_p3_tail_tasks_emit_all_five_named_quantities_through_explicit_prefixes() -> None:
    tasks = _load()["evaluation_tasks"]
    expected_prefixes = {
        "he_one_electron_tail_atlas": "executed_full_logabs_one_electron_tail",
        "he_center_of_mass_tail_atlas": "executed_full_logabs_center_of_mass_tail",
        "he_angular_shell_atlas": "executed_full_logabs_angular_shell_tail",
    }

    for task_name, prefix in expected_prefixes.items():
        summary = next(
            item
            for item in tasks[task_name]["summaries"]
            if item["_target_"] == "tpen.evaluation.summaries.HeliumTailSummary"
        )
        assert summary["series_name"] == "executed_full_logabs"
        assert summary["metric_prefix"] == prefix
        expected_metric_names = {
            f"{prefix}_slope",
            f"{prefix}_extrema_min",
            f"{prefix}_extrema_max",
            f"{prefix}_sign_fraction",
            f"{prefix}_outer_radius",
            f"{prefix}_directional_spread",
        }
        assert len(expected_metric_names) == 6  # extrema has explicit min/max endpoints.


def test_p3_ee_names_distinguish_ideal_unfloored_from_executed_smoothed() -> None:
    task = _load()["evaluation_tasks"]["he_ee_ideal_vs_executed_numerical_atlas"]
    rendered = str(task).lower()

    assert "ideal_unfloored" in rendered
    assert "executed_smoothed" in rendered
    assert "executed_smoothed_ee_factor" in rendered
    assert "kato" not in rendered
