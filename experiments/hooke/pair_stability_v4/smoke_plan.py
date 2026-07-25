"""Exact approved-smoke plan projection and fixture verification."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from audit_receipts import sha256_file
from roots import validate_lineage_id
from routes import (
    REPO_ROOT,
    ROUTES_PATH,
    V3_STUDY_DIR,
    config_source_receipt,
    legacy_source_receipt,
)

EXPECTED_SCAN_COUNT = 64
EXPECTED_BLIND_SEED = 811
SMOKE_PLAN_CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "reference"
    / "approved_smoke_plan_v1.json"
)
SMOKE_PLAN_SCHEMA_VERSION = "pair-stability-v4/approved-smoke-plan/v1"
SMOKE_PLAN_NORMALIZATION_VERSION = (
    "pair-stability-v4/smoke-plan-projection/v1"
)
GRID_MANIFEST_FIELDS = frozenset(
    {
        "study",
        "stage",
        "attempt_id",
        "created_at",
        "config",
        "grid",
        "results_root",
        "n_jobs",
        "jobs",
        "grid_schema",
        "major_axes",
        "minor_axes",
        "scan_seed_axis",
        "major_grid",
        "minor_grid",
        "scan_seeds",
        "scan_seed_rows",
        "axis_id_labels",
        "axis_overrides",
        "config_snapshots",
        "choice_validation",
        "seed_overrides",
        "final_seed_sequences",
        "static_overrides",
        "champions",
        "champion_kinds",
        "champion_reference_metrics",
        "final_replicates",
        "validation_config",
        "blinding",
    }
)
GRID_JOB_FIELDS = frozenset(
    {
        "run_id",
        "major_id",
        "minor_id",
        "config_id",
        "major_choices",
        "minor_choices",
        "scan_seed",
        "seed_values",
        "seed_overrides",
        "static_overrides",
        "train_dir",
        "validation_dir",
        "train_attempt_dir",
        "overrides",
        "command",
        "choices",
        "tags",
        "submitted",
        "launcher",
        "launcher_job_id",
    }
)


def audit_smoke_plan(
    root: Path,
    *,
    attempt: str,
    manifest: Mapping[str, Any],
    expected_study: str,
) -> tuple[str, ...]:
    """Return every mismatch against the approved exact smoke-plan fixture."""

    errors: list[str] = []
    smoke_path = (
        V3_STUDY_DIR / "configs" / "smoke.yaml"
        if expected_study == "pair_stability_v3"
        else V3_STUDY_DIR.parent / "pair_stability_v4" / "configs" / "smoke.yaml"
    )
    try:
        smoke = yaml.safe_load(smoke_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"cannot load approved smoke config: {exc}")
        return tuple(errors)
    if not isinstance(smoke, dict):
        errors.append("approved smoke config is not an object")
        return tuple(errors)
    expected_major_axes = list(smoke["major_grid"])
    expected_minor_axes = list(smoke["minor_grid"])
    if manifest.get("major_axes") != expected_major_axes:
        errors.append("grid major-axis order differs from smoke contract")
    if manifest.get("minor_axes") != expected_minor_axes:
        errors.append("grid minor-axis order differs from smoke contract")
    major_grid = manifest.get("major_grid")
    if not isinstance(major_grid, dict) or {
        key: len(value) if isinstance(value, list) else -1
        for key, value in major_grid.items()
    } != {
        key: len(value)
        for key, value in smoke["major_grid"].items()
    }:
        errors.append("grid blinded major-axis cardinalities differ")
    exact_fields = (
        "minor_grid",
        "scan_seed_axis",
        "scan_seed_rows",
        "axis_id_labels",
        "axis_overrides",
        "config_snapshots",
        "choice_validation",
        "seed_overrides",
        "final_seed_sequences",
        "static_overrides",
        "champions",
        "champion_reference_metrics",
        "final_replicates",
    )
    for key in exact_fields:
        if manifest.get(key) != smoke.get(key):
            errors.append(f"grid {key} differs from approved smoke contract")
    if manifest.get("grid_schema") != "major_minor_scan":
        errors.append("grid schema is not major_minor_scan")
    unblind = _read_json_for_audit(
        root / "00_grid" / attempt / "unblind.json",
        errors,
    )
    if unblind.get("blind_seed") != EXPECTED_BLIND_SEED:
        errors.append("unblind artifact blind seed is not 811")
    axes = unblind.get("axes")
    if not isinstance(axes, dict) or set(axes) != set(expected_major_axes):
        errors.append("unblind artifact axis population differs")
    try:
        contract = json.loads(SMOKE_PLAN_CONTRACT_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read approved smoke-plan contract: {exc}")
        return tuple(errors)
    fixture_errors = verify_smoke_plan_contract(contract)
    if fixture_errors:
        errors.extend(f"approved smoke-plan fixture: {error}" for error in fixture_errors)
        return tuple(errors)
    try:
        projection = smoke_plan_projection(
            root,
            attempt=attempt,
            expected_study=expected_study,
        )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        errors.append(f"cannot project exact smoke plan: {exc}")
        return tuple(errors)
    actual_digest = _projection_digest(projection)
    if actual_digest != contract["sha256"]:
        errors.append(
            f"grid ordered plan digest differs: {actual_digest} != "
            f"{contract['sha256']}"
        )
    if projection != contract["projection"]:
        errors.append("grid ordered plan projection differs from approved fixture")
    return tuple(dict.fromkeys(errors))


def smoke_plan_digest(
    root: Path,
    *,
    attempt: str,
    expected_study: str,
) -> str:
    """Return normalized exact digest of one planned smoke population."""

    return _projection_digest(
        smoke_plan_projection(
            root,
            attempt=attempt,
            expected_study=expected_study,
        )
    )


def smoke_plan_projection(
    root: Path,
    *,
    attempt: str,
    expected_study: str,
) -> dict[str, Any]:
    """Build the closed, ordered approved-smoke semantic projection."""

    root = Path(root).resolve(strict=True)
    attempt = validate_lineage_id(attempt)
    if expected_study not in {"pair_stability_v3", "pair_stability_v4"}:
        raise ValueError(f"unsupported projected study: {expected_study}")
    grid_dir = root / "00_grid" / attempt
    manifest = _read_json_strict(grid_dir / "manifest.json")
    if set(manifest) != GRID_MANIFEST_FIELDS:
        raise ValueError(
            "grid manifest fields mismatch; "
            f"missing={sorted(GRID_MANIFEST_FIELDS - set(manifest))}, "
            f"extra={sorted(set(manifest) - GRID_MANIFEST_FIELDS)}"
        )
    jobs = manifest["jobs"]
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_SCAN_COUNT:
        raise ValueError("grid manifest must contain exactly 64 ordered jobs")
    major_axes = _string_list(manifest["major_axes"], "major_axes")
    minor_axes = _string_list(manifest["minor_axes"], "minor_axes")
    seed_axis = str(manifest["scan_seed_axis"])
    all_axes = [*major_axes, *minor_axes, seed_axis]

    projected_jobs: list[dict[str, Any]] = []
    raw_commands: list[str] = []
    run_ids: list[str] = []
    for index, raw_job in enumerate(jobs):
        if not isinstance(raw_job, dict) or set(raw_job) != GRID_JOB_FIELDS:
            raise ValueError(f"grid job {index} fields do not match approved schema")
        run_id = str(raw_job["run_id"])
        if not run_id or "/" in run_id or ".." in Path(run_id).parts:
            raise ValueError(f"grid job {index} has invalid run_id")
        if run_id in run_ids:
            raise ValueError(f"duplicate grid run_id: {run_id}")
        run_ids.append(run_id)
        major_choices = _ordered_pairs(
            raw_job["major_choices"],
            major_axes,
            f"job {index} major_choices",
        )
        minor_choices = _ordered_pairs(
            raw_job["minor_choices"],
            minor_axes,
            f"job {index} minor_choices",
        )
        choices = _ordered_pairs(
            raw_job["choices"],
            all_axes,
            f"job {index} choices",
        )
        command = shlex.split(str(raw_job["command"]))
        normalized_command = _normalize_job_command(
            command,
            root=root,
            attempt=attempt,
            expected_study=expected_study,
            run_id=run_id,
        )
        raw_commands.append(str(raw_job["command"]))
        projected_jobs.append(
            {
                "run_id": run_id,
                "major_id": str(raw_job["major_id"]),
                "minor_id": str(raw_job["minor_id"]),
                "config_id": str(raw_job["config_id"]),
                "major_choices": major_choices,
                "minor_choices": minor_choices,
                "scan_seed": raw_job["scan_seed"],
                "seed_values": _sorted_pairs(raw_job["seed_values"]),
                "seed_overrides": _nested_sorted_pairs(raw_job["seed_overrides"]),
                "static_overrides": _nested_sorted_pairs(raw_job["static_overrides"]),
                "train_dir": _normalize_exact_path(
                    raw_job["train_dir"],
                    root / "01_train" / run_id,
                    root=root,
                    attempt=attempt,
                ),
                "validation_dir": _normalize_exact_path(
                    raw_job["validation_dir"],
                    root / "02_validation" / run_id,
                    root=root,
                    attempt=attempt,
                ),
                "train_attempt_dir": _normalize_exact_path(
                    raw_job["train_attempt_dir"],
                    root / "01_train" / run_id / attempt,
                    root=root,
                    attempt=attempt,
                ),
                "overrides": _normalize_job_overrides(
                    raw_job["overrides"],
                    root=root,
                    attempt=attempt,
                    expected_study=expected_study,
                    run_id=run_id,
                ),
                "command": normalized_command,
                "choices": choices,
                "tags": _string_list(raw_job["tags"], f"job {index} tags"),
                "submitted": _require_exact(raw_job["submitted"], False, "submitted"),
                "launcher": _require_exact(raw_job["launcher"], None, "launcher"),
                "launcher_job_id": _require_exact(
                    raw_job["launcher_job_id"],
                    None,
                    "launcher_job_id",
                ),
            }
        )

    commands = _validate_commands_file(
        grid_dir / "commands.sh",
        raw_commands=raw_commands,
        expected_study=expected_study,
        attempt=attempt,
    )
    _validate_job_files(grid_dir / "jobs", jobs)
    unblind = _project_unblind(
        grid_dir / "unblind.json",
        manifest=manifest,
        expected_study=expected_study,
    )
    snapshots = _ordered_pairs(
        manifest["config_snapshots"],
        ("train", "validation"),
        "config_snapshots",
    )
    return {
        "study": "pair_stability_v4",
        "stage": _require_exact(manifest["stage"], "00_grid", "grid stage"),
        "attempt_id": "<ATTEMPT_ID>",
        "config": _normalize_exact_path(
            manifest["config"],
            grid_dir / "train_config.yaml",
            root=root,
            attempt=attempt,
        ),
        "grid": _normalize_source_config_path(
            manifest["grid"],
            expected_study=expected_study,
            filename="smoke.yaml",
        ),
        "results_root": _normalize_exact_path(
            manifest["results_root"],
            root,
            root=root,
            attempt=attempt,
        ),
        "n_jobs": _require_exact(manifest["n_jobs"], 64, "n_jobs"),
        "grid_schema": _require_exact(
            manifest["grid_schema"],
            "major_minor_scan",
            "grid_schema",
        ),
        "major_axes": major_axes,
        "minor_axes": minor_axes,
        "scan_seed_axis": seed_axis,
        "major_grid": _ordered_pairs(
            manifest["major_grid"],
            major_axes,
            "major_grid",
        ),
        "minor_grid": _ordered_pairs(
            manifest["minor_grid"],
            minor_axes,
            "minor_grid",
        ),
        "scan_seeds": manifest["scan_seeds"],
        "scan_seed_rows": manifest["scan_seed_rows"],
        "axis_id_labels": _ordered_pairs(
            manifest["axis_id_labels"],
            all_axes,
            "axis_id_labels",
        ),
        "axis_overrides": _ordered_pairs(
            manifest["axis_overrides"],
            [*major_axes, *minor_axes],
            "axis_overrides",
        ),
        "config_snapshots": snapshots,
        "choice_validation": _nested_sorted_pairs(manifest["choice_validation"]),
        "seed_overrides": _nested_sorted_pairs(manifest["seed_overrides"]),
        "final_seed_sequences": _nested_sorted_pairs(
            manifest["final_seed_sequences"]
        ),
        "static_overrides": _nested_sorted_pairs(manifest["static_overrides"]),
        "champions": manifest["champions"],
        "champion_kinds": manifest["champion_kinds"],
        "champion_reference_metrics": manifest["champion_reference_metrics"],
        "final_replicates": _require_exact(
            manifest["final_replicates"],
            1,
            "final_replicates",
        ),
        "validation_config": _normalize_exact_path(
            manifest["validation_config"],
            grid_dir / "validation_config.yaml",
            root=root,
            attempt=attempt,
        ),
        "blinding": manifest["blinding"],
        "jobs": projected_jobs,
        "unblind": unblind,
        "commands": commands,
        "snapshot_sha256": {
            "grid.yaml": _config_snapshot_digest(
                grid_dir / "grid.yaml",
                kind="grid",
                expected_study=expected_study,
                root=root,
                attempt=attempt,
            ),
            "train_config.yaml": _config_snapshot_digest(
                grid_dir / "train_config.yaml",
                kind="train",
                expected_study=expected_study,
                root=root,
                attempt=attempt,
            ),
            "validation_config.yaml": _config_snapshot_digest(
                grid_dir / "validation_config.yaml",
                kind="validation",
                expected_study=expected_study,
                root=root,
                attempt=attempt,
            ),
        },
    }


def verify_smoke_plan_contract(contract: object) -> tuple[str, ...]:
    """Validate fixture schema, self-digest, and pinned generator inputs."""

    errors: list[str] = []
    if not isinstance(contract, dict):
        return ("contract is not an object",)
    expected_fields = {
        "schema_version",
        "normalization_version",
        "generator",
        "projection",
        "sha256",
    }
    if set(contract) != expected_fields:
        errors.append("contract fields mismatch")
        return tuple(errors)
    if contract.get("schema_version") != SMOKE_PLAN_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if contract.get("normalization_version") != SMOKE_PLAN_NORMALIZATION_VERSION:
        errors.append("normalization_version mismatch")
    projection = contract.get("projection")
    if not isinstance(projection, dict):
        errors.append("projection is not an object")
    elif contract.get("sha256") != _projection_digest(projection):
        errors.append("projection self-digest mismatch")
    generator = contract.get("generator")
    if not isinstance(generator, dict):
        errors.append("generator is not an object")
    else:
        expected_generator = smoke_plan_generator_provenance()
        if generator != expected_generator:
            errors.append("generator source/config/route provenance mismatch")
    return tuple(errors)


def smoke_plan_generator_provenance() -> dict[str, Any]:
    """Return current pinned inputs that authorize the smoke-plan fixture."""

    legacy = legacy_source_receipt(REPO_ROOT)
    configs = config_source_receipt(REPO_ROOT)
    return {
        "legacy_source_closure_sha256": legacy["closure_sha256"],
        "config_files": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in configs["files"]
        ],
        "routes_sha256": sha256_file(ROUTES_PATH),
        "blind_seed": EXPECTED_BLIND_SEED,
        "route_role": "screen_plan",
    }


def _projection_digest(projection: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json_strict(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ordered_pairs(
    value: object,
    order: Sequence[str],
    name: str,
) -> list[list[Any]]:
    if not isinstance(value, dict) or set(value) != set(order):
        raise ValueError(f"{name} keys differ from declared order")
    return [[key, value[key]] for key in order]


def _sorted_pairs(value: object) -> list[list[Any]]:
    if not isinstance(value, dict):
        raise ValueError("expected mapping")
    return [[str(key), value[key]] for key in sorted(value)]


def _nested_sorted_pairs(value: object) -> list[list[Any]]:
    if not isinstance(value, dict):
        raise ValueError("expected nested mapping")
    rows: list[list[Any]] = []
    for key in sorted(value):
        item = value[key]
        rows.append(
            [
                str(key),
                _nested_sorted_pairs(item) if isinstance(item, dict) else item,
            ]
        )
    return rows


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{name} must be a list of strings")
    return list(value)


def _normalize_exact_path(
    value: object,
    expected: Path,
    *,
    root: Path,
    attempt: str,
) -> str:
    path = Path(str(value))
    if path.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError(f"path {path} does not equal expected {expected}")
    text = str(path)
    root_text = str(root)
    if text == root_text:
        return "<RESULTS_ROOT>"
    relative = path.resolve(strict=False).relative_to(root)
    parts = [
        "<ATTEMPT_ID>" if part == attempt else part
        for part in relative.parts
    ]
    return str(Path("<RESULTS_ROOT>").joinpath(*parts))


def _normalize_source_config_path(
    value: object,
    *,
    expected_study: str,
    filename: str,
) -> str:
    expected_relative = Path(
        f"experiments/hooke/{expected_study}/configs/{filename}"
    )
    path = Path(str(value))
    if path.is_absolute():
        expected = REPO_ROOT / expected_relative
        if path.resolve(strict=False) != expected.resolve(strict=False):
            raise ValueError(f"source config path mismatch: {path}")
    elif path != expected_relative:
        raise ValueError(f"source config path mismatch: {path}")
    return f"experiments/hooke/pair_stability_v4/configs/{filename}"


def _normalize_job_overrides(
    value: object,
    *,
    root: Path,
    attempt: str,
    expected_study: str,
    run_id: str,
) -> list[str]:
    overrides = _string_list(value, "job overrides")
    normalized: list[str] = []
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"invalid CLI override: {override}")
        if key == "run.root":
            expected = root / "01_train"
            raw_value = _normalize_exact_path(
                raw_value,
                expected,
                root=root,
                attempt=attempt,
            )
        elif key == "run.run_id":
            if raw_value != f"{run_id}/{attempt}":
                raise ValueError("job run.run_id override mismatch")
            raw_value = f"{run_id}/<ATTEMPT_ID>"
        elif key in {"study.attempt_id"}:
            raw_value = str(
                _require_exact(raw_value, attempt, "study attempt override")
            ).replace(attempt, "<ATTEMPT_ID>")
        elif key in {"study.name", "experiment.name"}:
            _require_exact(raw_value, expected_study, f"{key} override")
            raw_value = "pair_stability_v4"
        elif key == "experiment.run_name":
            _require_exact(
                raw_value,
                f"{expected_study}_train",
                "experiment.run_name override",
            )
            raw_value = "pair_stability_v4_train"
        normalized.append(f"{key}={raw_value}")
    return normalized


def _normalize_job_command(
    command: Sequence[str],
    *,
    root: Path,
    attempt: str,
    expected_study: str,
    run_id: str,
) -> list[str]:
    if len(command) < 5 or command[:4] != ["python", "-u", "run.py", "--config"]:
        raise ValueError("job command prefix differs")
    config = _normalize_exact_path(
        command[4],
        root / "00_grid" / attempt / "train_config.yaml",
        root=root,
        attempt=attempt,
    )
    overrides = _normalize_job_overrides(
        list(command[5:]),
        root=root,
        attempt=attempt,
        expected_study=expected_study,
        run_id=run_id,
    )
    return ["python", "-u", "run.py", "--config", config, *overrides]


def _validate_commands_file(
    path: Path,
    *,
    raw_commands: Sequence[str],
    expected_study: str,
    attempt: str,
) -> list[str]:
    lines = path.read_text().splitlines()
    expected_header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# {expected_study} 00_grid attempt {attempt}",
        "",
    ]
    if lines[:5] != expected_header or lines[5:] != list(raw_commands):
        raise ValueError("commands.sh does not exactly match ordered manifest jobs")
    return ["<COMMAND_FROM_JOB>" for _ in raw_commands]


def _validate_job_files(directory: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
    expected = {f"{job['run_id']}.json": job for job in jobs}
    actual_names = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_names != set(expected):
        raise ValueError("grid jobs directory filename population mismatch")
    for filename, job in expected.items():
        if _read_json_strict(directory / filename) != job:
            raise ValueError(f"grid job file differs from manifest row: {filename}")


def _project_unblind(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    expected_study: str,
) -> dict[str, Any]:
    value = _read_json_strict(path)
    if set(value) != {"blind_seed", "original_grid", "axes"}:
        raise ValueError("unblind fields mismatch")
    _require_exact(value["blind_seed"], EXPECTED_BLIND_SEED, "unblind seed")
    original_grid = _normalize_source_config_path(
        value["original_grid"],
        expected_study=expected_study,
        filename="smoke.yaml",
    )
    axes = value["axes"]
    major_axes = _string_list(manifest["major_axes"], "major_axes")
    if not isinstance(axes, dict) or set(axes) != set(major_axes):
        raise ValueError("unblind axes mismatch")
    source_smoke = yaml.safe_load(
        (
            V3_STUDY_DIR.parent
            / expected_study
            / "configs"
            / "smoke.yaml"
        ).read_text()
    )
    projected_axes: list[list[Any]] = []
    for axis in major_axes:
        row = axes[axis]
        if not isinstance(row, dict) or set(row) != {
            "slot_to_value",
            "value_to_slot",
        }:
            raise ValueError(f"unblind {axis} fields mismatch")
        slot_to_value = row["slot_to_value"]
        value_to_slot = row["value_to_slot"]
        if not isinstance(slot_to_value, dict) or not isinstance(
            value_to_slot,
            dict,
        ):
            raise ValueError(f"unblind {axis} maps must be objects")
        if {value: slot for slot, value in slot_to_value.items()} != value_to_slot:
            raise ValueError(f"unblind {axis} maps are not bijective inverses")
        if set(value_to_slot) != set(source_smoke["major_grid"][axis]):
            raise ValueError(f"unblind {axis} semantic domain mismatch")
        projected_axes.append(
            [
                axis,
                {
                    "slot_to_value": _sorted_pairs(slot_to_value),
                    "value_to_slot": _sorted_pairs(value_to_slot),
                },
            ]
        )
    return {
        "blind_seed": EXPECTED_BLIND_SEED,
        "original_grid": original_grid,
        "axes": projected_axes,
    }


def _config_snapshot_digest(
    path: Path,
    *,
    kind: str,
    expected_study: str,
    root: Path,
    attempt: str,
) -> str:
    """Digest one snapshot after a closed identity-only V3→V4 transform."""

    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"config snapshot is not a mapping: {path}")
    projected = dict(payload)
    if kind == "grid":
        _require_exact(payload.get("study"), expected_study, "grid snapshot study")
        projected["study"] = "pair_stability_v4"
        projected["config"] = _normalize_exact_path(
            payload.get("config"),
            root / "00_grid" / attempt / "train_config.yaml",
            root=root,
            attempt=attempt,
        )
        projected["validation_config"] = _normalize_exact_path(
            payload.get("validation_config"),
            root / "00_grid" / attempt / "validation_config.yaml",
            root=root,
            attempt=attempt,
        )
        _require_exact(
            payload.get("results_root"),
            f"experiments/hooke/{expected_study}/results",
            "grid snapshot results_root",
        )
        projected["results_root"] = (
            "experiments/hooke/pair_stability_v4/results"
        )
    elif kind in {"train", "validation"}:
        expected_stage = "01_train" if kind == "train" else "02_validation"
        study = payload.get("study")
        if not isinstance(study, dict):
            raise ValueError(f"{kind} snapshot study is not a mapping")
        _require_exact(
            study.get("name"),
            expected_study,
            f"{kind} snapshot study.name",
        )
        projected_study = dict(study)
        projected_study["name"] = "pair_stability_v4"
        projected["study"] = projected_study
        run = payload.get("run")
        if not isinstance(run, dict):
            raise ValueError(f"{kind} snapshot run is not a mapping")
        _require_exact(
            run.get("root"),
            f"experiments/hooke/{expected_study}/results/{expected_stage}",
            f"{kind} snapshot run.root",
        )
        projected_run = dict(run)
        projected_run["root"] = (
            f"experiments/hooke/pair_stability_v4/results/{expected_stage}"
        )
        projected["run"] = projected_run
    else:
        raise ValueError(f"unsupported config snapshot kind: {kind}")
    canonical = _yaml_ordered_projection(projected)
    return _projection_digest({"snapshot": canonical})


def _yaml_ordered_projection(value: Any) -> Any:
    """Represent YAML mappings as ordered pairs without sorting their keys."""

    if isinstance(value, dict):
        return [
            [str(key), _yaml_ordered_projection(item)]
            for key, item in value.items()
        ]
    if isinstance(value, list):
        return [_yaml_ordered_projection(item) for item in value]
    return value


def _require_exact(value: Any, expected: Any, name: str) -> Any:
    if value != expected:
        raise ValueError(f"{name}={value!r}, expected {expected!r}")
    return value


def _read_json_for_audit(
    path: Path,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        errors.append(f"invalid JSON: {path}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path}")
        return {}
    return value
