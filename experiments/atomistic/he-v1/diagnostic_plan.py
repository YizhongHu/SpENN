"""Plan the immutable post-hoc ``he-v1-diagnostic-v1`` study.

The committed grid contains logical checkpoint identities and expected content
hashes.  An external source map supplies facility paths.  Planning reconciles
the two before writing a content-addressed manifest, so no evaluation can
silently restore a different checkpoint or a partial real-format directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import layout  # noqa: E402

GRID_SCHEMA = "he-v1-diagnostic-grid/v1"
SOURCE_SCHEMA = "he-v1-diagnostic-sources/v1"
PLAN_SCHEMA = "he-v1-diagnostic-plan/v1"
STUDY_TIMEZONE = "America/New_York"

GRID_KEYS = frozenset(
    {
        "schema",
        "study",
        "base_eval_config",
        "overlay_config",
        "source_git_sha",
        "checkpoints",
        "trajectory_protocols",
        "sensitivity_protocols",
        "factor_arms",
        "factor_common_configuration",
        "factor_reequilibrated",
        "checkpoint_diagnostics",
        "resources",
        "smoke_scale",
    }
)
CHECKPOINT_KEYS = frozenset(
    {"label", "completed_updates", "model_sha256", "manifest_sha256"}
)
TRAJECTORY_KEYS = frozenset(
    {"name", "comparison_kind", "seeds", "n_walkers", "n_draws", "burn_in", "stride"}
)
SENSITIVITY_KEYS = frozenset(
    {"name", "comparison_kind", "seed", "n_walkers", "n_draws", "burn_in", "stride"}
)
FACTOR_ARM_KEYS = frozenset(
    {"label", "seed", "b_ee", "c_electron_nucleus", "d_electron_nucleus"}
)
RESOURCE_KEYS = frozenset(
    {"partition", "stratum", "constraint", "timeout_min", "cpus", "mem_gb", "gpus"}
)
FACTOR_COMMON_KEYS = frozenset({"seed", "n_walkers", "burn_in", "stride"})
FACTOR_REEQUILIBRATED_KEYS = frozenset(
    {"n_walkers", "n_draws", "burn_in", "stride"}
)
CHECKPOINT_DIAGNOSTIC_KEYS = frozenset(
    {"seed", "n_walkers", "burn_in", "stride", "n_samples", "task_names"}
)
SMOKE_SCALE_KEYS = frozenset(
    {
        "n_walkers",
        "n_draws",
        "burn_in",
        "stride",
        "diagnostic_samples",
        "atlas_max_refinement_steps",
        "atlas_radii",
    }
)


class DiagnosticPlanError(ValueError):
    """A diagnostic grid or checkpoint source cannot be planned safely."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 content id of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with stable separators and key order."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_grid(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen diagnostic grid."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DiagnosticPlanError("diagnostic grid must be a mapping")
    missing = sorted(GRID_KEYS - set(payload))
    unknown = sorted(set(payload) - GRID_KEYS)
    if missing or unknown:
        raise DiagnosticPlanError(f"diagnostic grid keys mismatch: missing={missing}, unknown={unknown}")
    if payload["schema"] != GRID_SCHEMA:
        raise DiagnosticPlanError(f"diagnostic grid schema must be {GRID_SCHEMA!r}")
    if payload["study"] != "he-v1-diagnostic-v1":
        raise DiagnosticPlanError("diagnostic study identity is frozen as 'he-v1-diagnostic-v1'")
    if payload["base_eval_config"] != "experiments/atomistic/he-v1/configs/eval.yaml":
        raise DiagnosticPlanError("diagnostic base evaluation config changed")
    if payload["overlay_config"] != (
        "experiments/atomistic/he-v1/configs/diagnostic_eval.yaml"
    ):
        raise DiagnosticPlanError("diagnostic overlay config changed")
    if payload["source_git_sha"] != "418accf153368aab45586dc2a2cc97c18472691c":
        raise DiagnosticPlanError("diagnostic checkpoint source Git SHA changed")
    checkpoints = _mapping_sequence(payload["checkpoints"], "checkpoints", CHECKPOINT_KEYS)
    expected_checkpoints = [
        (
            "step_025000",
            25000,
            "7f897ee268e6f2261fb58f4136d08121258b31478da3bf31526f9ce7fea58b1f",
            "fbc5f20ac95da58479d0651b6ca7beb5f931b61123d330bc0dfac67982f77510",
        ),
        (
            "step_050000",
            50000,
            "9a8acc0e6f2ca86c405880ce5a43c322bfddb4352315c692e257aced7b0d07d7",
            "a4e77b36ec250a500014672b9128be3d18c0809e528d5b64dd0c1358c4a276ed",
        ),
    ]
    actual_checkpoints = [
        (
            checkpoint["label"],
            checkpoint["completed_updates"],
            checkpoint["model_sha256"],
            checkpoint["manifest_sha256"],
        )
        for checkpoint in checkpoints
    ]
    if actual_checkpoints != expected_checkpoints:
        raise DiagnosticPlanError("diagnostic checkpoints must be exactly 25k then 50k")
    for checkpoint in checkpoints:
        _require_text(checkpoint["label"], "checkpoint.label")
        _require_sha256(checkpoint["model_sha256"], "checkpoint.model_sha256")
        _require_sha256(checkpoint["manifest_sha256"], "checkpoint.manifest_sha256")

    trajectory = _mapping_sequence(
        payload["trajectory_protocols"], "trajectory_protocols", TRAJECTORY_KEYS
    )
    expected_trajectory = [
        (
            "primary_256x4096",
            "primary_headline",
            [1000, 1001, 1002, 1003],
            4096,
            256,
            100,
            20,
        ),
        (
            "long_1024x1024",
            "long_chain_diagnostic",
            [2000, 2001, 2002, 2003],
            1024,
            1024,
            100,
            20,
        ),
    ]
    actual_trajectory = [
        (
            row["name"],
            row["comparison_kind"],
            list(row["seeds"]),
            row["n_walkers"],
            row["n_draws"],
            row["burn_in"],
            row["stride"],
        )
        for row in trajectory
    ]
    if actual_trajectory != expected_trajectory:
        raise DiagnosticPlanError(
            "trajectory protocols must freeze four 256x4096 seeds 1000-1003 and "
            "four 1024x1024 seeds 2000-2003"
        )
    for protocol in trajectory:
        _validate_chain(protocol, seeds_key="seeds")

    sensitivity = _mapping_sequence(
        payload["sensitivity_protocols"], "sensitivity_protocols", SENSITIVITY_KEYS
    )
    expected_sensitivity = [
        ("burn_in_50", "burn_in_sensitivity", 3000, 256, 256, 50, 20),
        ("burn_in_200", "burn_in_sensitivity", 3001, 256, 256, 200, 20),
        ("stride_10", "stride_sensitivity", 3100, 256, 256, 100, 10),
        ("stride_40", "stride_sensitivity", 3101, 256, 256, 100, 40),
    ]
    actual_sensitivity = [
        (
            row["name"],
            row["comparison_kind"],
            row["seed"],
            row["n_walkers"],
            row["n_draws"],
            row["burn_in"],
            row["stride"],
        )
        for row in sensitivity
    ]
    if actual_sensitivity != expected_sensitivity:
        raise DiagnosticPlanError("burn-in and stride sensitivity arms changed")
    for protocol in sensitivity:
        _validate_chain(protocol, seeds_key="seed")

    factor_arms = _mapping_sequence(payload["factor_arms"], "factor_arms", FACTOR_ARM_KEYS)
    labels = [str(arm["label"]) for arm in factor_arms]
    if labels != [
        "baseline",
        "b_ee_minus_10pct",
        "b_ee_plus_10pct",
        "c_en_minus_10pct",
        "c_en_plus_10pct",
        "d_en_minus_10pct",
        "d_en_plus_10pct",
    ]:
        raise DiagnosticPlanError("factor arms or their frozen order changed")
    expected_factor_arms = [
        ("baseline", 4000, 1.0, 1.0, 1.0),
        ("b_ee_minus_10pct", 4001, 0.9, 1.0, 1.0),
        ("b_ee_plus_10pct", 4002, 1.1, 1.0, 1.0),
        ("c_en_minus_10pct", 4003, 1.0, 0.9, 1.0),
        ("c_en_plus_10pct", 4004, 1.0, 1.1, 1.0),
        ("d_en_minus_10pct", 4005, 1.0, 1.0, 0.9),
        ("d_en_plus_10pct", 4006, 1.0, 1.0, 1.1),
    ]
    actual_factor_arms = [
        (
            arm["label"],
            arm["seed"],
            arm["b_ee"],
            arm["c_electron_nucleus"],
            arm["d_electron_nucleus"],
        )
        for arm in factor_arms
    ]
    if actual_factor_arms != expected_factor_arms:
        raise DiagnosticPlanError("factor arm coordinates changed")
    for arm in factor_arms:
        _require_nonnegative_int(arm["seed"], f"factor arm {arm['label']} seed")
        for key in ("b_ee", "c_electron_nucleus", "d_electron_nucleus"):
            value = arm[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
                raise DiagnosticPlanError(f"factor arm {arm['label']} requires positive {key}")

    resources = payload["resources"]
    if not isinstance(resources, Mapping) or set(resources) != {"production", "smoke"}:
        raise DiagnosticPlanError("resources must define exactly production and smoke")
    for name, resource in resources.items():
        _require_exact_keys(resource, RESOURCE_KEYS, f"resources.{name}")
        for key in ("timeout_min", "cpus", "mem_gb", "gpus"):
            _require_positive_int(resource[key], f"resources.{name}.{key}")
    if resources["production"]["partition"] != "kozinsky_gpu" or resources["production"]["stratum"] != "a100":
        raise DiagnosticPlanError("production diagnostics are frozen to pinned A100 on kozinsky_gpu")
    if resources["smoke"]["partition"] != "gpu_test" or resources["smoke"]["stratum"] != "a100_mig":
        raise DiagnosticPlanError("smoke diagnostics are frozen to gpu_test A100 MIG")
    if dict(resources["production"]) != {
        "partition": "kozinsky_gpu",
        "stratum": "a100",
        "constraint": "a100",
        "timeout_min": 720,
        "cpus": 4,
        "mem_gb": 32,
        "gpus": 1,
    }:
        raise DiagnosticPlanError("production diagnostic resources changed")
    if dict(resources["smoke"]) != {
        "partition": "gpu_test",
        "stratum": "a100_mig",
        "constraint": None,
        "timeout_min": 120,
        "cpus": 4,
        "mem_gb": 32,
        "gpus": 1,
    }:
        raise DiagnosticPlanError("smoke diagnostic resources changed")

    common = payload["factor_common_configuration"]
    _require_exact_keys(common, FACTOR_COMMON_KEYS, "factor_common_configuration")
    _validate_snapshot(common, name="factor_common_configuration", seed=True)
    if tuple(common[key] for key in ("seed", "n_walkers", "burn_in", "stride")) != (
        5000,
        1024,
        100,
        20,
    ):
        raise DiagnosticPlanError("common-configuration factor protocol changed")

    reequilibrated = payload["factor_reequilibrated"]
    _require_exact_keys(
        reequilibrated,
        FACTOR_REEQUILIBRATED_KEYS,
        "factor_reequilibrated",
    )
    _validate_chain_shape(reequilibrated, name="factor_reequilibrated")
    if tuple(
        reequilibrated[key]
        for key in ("n_walkers", "n_draws", "burn_in", "stride")
    ) != (256, 128, 100, 20):
        raise DiagnosticPlanError("re-equilibrated factor protocol changed")

    diagnostics = payload["checkpoint_diagnostics"]
    _require_exact_keys(
        diagnostics,
        CHECKPOINT_DIAGNOSTIC_KEYS,
        "checkpoint_diagnostics",
    )
    _validate_snapshot(diagnostics, name="checkpoint_diagnostics", seed=True)
    _require_positive_int(diagnostics["n_samples"], "checkpoint_diagnostics.n_samples")
    task_names = diagnostics["task_names"]
    if not isinstance(task_names, Sequence) or isinstance(task_names, (str, bytes)):
        raise DiagnosticPlanError("checkpoint_diagnostics.task_names must be a sequence")
    if len(task_names) != len(set(task_names)) or not all(
        isinstance(name, str) and name.strip() for name in task_names
    ):
        raise DiagnosticPlanError(
            "checkpoint_diagnostics.task_names must be unique non-empty strings"
        )
    expected_diagnostic_tasks = [
        "he_radial_profiles",
        "he_en_numerical_atlas",
        "he_ee_ideal_vs_executed_numerical_atlas",
        "he_one_electron_tail_atlas",
        "he_center_of_mass_tail_atlas",
        "he_angular_shell_atlas",
        "full_model_antisymmetry",
        "spatial_exchange_symmetry",
        "rotation_consistency",
        "trace_equivariance",
        "feature_trace",
        "readout_trace",
    ]
    if list(task_names) != expected_diagnostic_tasks:
        raise DiagnosticPlanError("checkpoint diagnostic task graph changed")
    if tuple(
        diagnostics[key]
        for key in ("seed", "n_walkers", "burn_in", "stride", "n_samples")
    ) != (6000, 4096, 100, 20, 4096):
        raise DiagnosticPlanError("checkpoint diagnostic sampling protocol changed")

    smoke_scale = payload["smoke_scale"]
    _require_exact_keys(smoke_scale, SMOKE_SCALE_KEYS, "smoke_scale")
    for key in (
        "n_walkers",
        "n_draws",
        "burn_in",
        "stride",
        "diagnostic_samples",
        "atlas_max_refinement_steps",
    ):
        _require_positive_int(smoke_scale[key], f"smoke_scale.{key}")
    radii = smoke_scale["atlas_radii"]
    if not isinstance(radii, Sequence) or isinstance(radii, (str, bytes)) or not radii:
        raise DiagnosticPlanError("smoke_scale.atlas_radii must be non-empty")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) <= 0.0
        for value in radii
    ):
        raise DiagnosticPlanError("smoke_scale.atlas_radii must be positive")
    if tuple(
        smoke_scale[key]
        for key in (
            "n_walkers",
            "n_draws",
            "burn_in",
            "stride",
            "diagnostic_samples",
            "atlas_max_refinement_steps",
        )
    ) != (4, 2, 1, 1, 4, 4) or list(radii) != [2.0]:
        raise DiagnosticPlanError("smoke scale overrides changed")
    return dict(payload)


def load_sources(path: str | Path) -> Mapping[str, Path]:
    """Load an external mapping of logical checkpoint labels to directories."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "checkpoints"}:
        raise DiagnosticPlanError("source map must contain exactly schema and checkpoints")
    if payload["schema"] != SOURCE_SCHEMA or not isinstance(payload["checkpoints"], Mapping):
        raise DiagnosticPlanError(f"source map schema must be {SOURCE_SCHEMA!r}")
    return {str(label): Path(str(source)).resolve() for label, source in payload["checkpoints"].items()}


def reconcile_checkpoint_sources(
    grid: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Bind every logical checkpoint to one complete, content-matching source."""

    expected_labels = [str(checkpoint["label"]) for checkpoint in grid["checkpoints"]]
    if set(sources) != set(expected_labels):
        raise DiagnosticPlanError(
            f"source labels mismatch: expected={expected_labels}, actual={sorted(sources)}"
        )
    reconciled = []
    for expected in grid["checkpoints"]:
        label = str(expected["label"])
        source = sources[label]
        required = [source / "model.pt", source / "manifest.json", source / "COMPLETE"]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise DiagnosticPlanError(f"checkpoint {label} is incomplete; missing={missing}")
        model_sha = file_sha256(required[0])
        manifest_sha = file_sha256(required[1])
        if model_sha != expected["model_sha256"] or manifest_sha != expected["manifest_sha256"]:
            raise DiagnosticPlanError(
                f"checkpoint {label} content mismatch: model={model_sha}, manifest={manifest_sha}"
            )
        manifest = json.loads(required[1].read_text(encoding="utf-8"))
        if manifest.get("kind") != "tpen.checkpoint" or manifest.get("schema_version") != 2:
            raise DiagnosticPlanError(f"checkpoint {label} is not the required real format")
        if manifest.get("completed_updates") != expected["completed_updates"]:
            raise DiagnosticPlanError(f"checkpoint {label} completed_updates mismatch")
        files = manifest.get("files")
        provenance = manifest.get("provenance")
        if not isinstance(files, Mapping) or files.get("model") != "model.pt":
            raise DiagnosticPlanError(f"checkpoint {label} manifest does not bind model.pt")
        if not isinstance(provenance, Mapping) or provenance.get("git_sha") != grid["source_git_sha"]:
            raise DiagnosticPlanError(f"checkpoint {label} source git SHA mismatch")
        if not isinstance(provenance.get("tpen_version"), str) or not str(
            provenance["tpen_version"]
        ).strip():
            raise DiagnosticPlanError(f"checkpoint {label} lacks source TPEN version")
        if "resolved_config" not in files or not (source / str(files["resolved_config"])).is_file():
            raise DiagnosticPlanError(f"checkpoint {label} lacks its real-format resolved config")
        reconciled.append(
            {
                **dict(expected),
                "source_dir": str(source),
                "complete_sha256": file_sha256(required[2]),
                "checkpoint_schema_version": manifest["schema_version"],
                "checkpoint_kind": manifest["kind"],
                "source_git_sha": provenance.get("git_sha"),
                "source_tpen_version": provenance.get("tpen_version"),
            }
        )
    return reconciled


def expand_rows(grid: Mapping[str, Any], checkpoints: Sequence[Mapping[str, Any]], *, scale: str) -> list[dict[str, Any]]:
    """Expand the frozen grid into stable, independently collectable rows."""

    if scale not in {"production", "smoke"}:
        raise DiagnosticPlanError("scale must be 'production' or 'smoke'")
    resources = dict(grid["resources"][scale])
    smoke = dict(grid["smoke_scale"]) if scale == "smoke" else None
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        label = str(checkpoint["label"])
        for protocol in grid["trajectory_protocols"]:
            for seed in protocol["seeds"]:
                shape = _scaled_chain(protocol, smoke)
                rows.append(
                    _row(
                        checkpoint,
                        row_id=f"{label}-{protocol['name']}-seed{int(seed):04d}",
                        profile="retained_energy",
                        task_names=["retained_energy"],
                        protocol=str(protocol["name"]),
                        comparison_kind=str(protocol["comparison_kind"]),
                        seed=int(seed),
                        resources=resources,
                        **shape,
                    )
                )
        for protocol in grid["sensitivity_protocols"]:
            shape = _scaled_chain(protocol, smoke)
            rows.append(
                _row(
                    checkpoint,
                    row_id=f"{label}-{protocol['name']}-seed{int(protocol['seed']):04d}",
                    profile="retained_energy",
                    task_names=["retained_energy"],
                    protocol=str(protocol["name"]),
                    comparison_kind=str(protocol["comparison_kind"]),
                    seed=int(protocol["seed"]),
                    resources=resources,
                    **shape,
                )
            )
        common = dict(grid["factor_common_configuration"])
        common_shape = _scaled_snapshot(common, smoke)
        rows.append(
            _row(
                checkpoint,
                row_id=f"{label}-factor-common-configuration",
                profile="common_factor_response",
                task_names=["common_factor_response"],
                protocol="factor_common_configuration",
                comparison_kind="common_configuration",
                seed=int(common["seed"]),
                resources=resources,
                n_draws=1,
                record_capacity=common_shape["n_walkers"],
                diagnostic_samples=common_shape["n_walkers"],
                **common_shape,
            )
        )
        reeq = dict(grid["factor_reequilibrated"])
        for arm in grid["factor_arms"]:
            shape = _scaled_chain(reeq, smoke)
            factor_arm = {key: arm[key] for key in FACTOR_ARM_KEYS if key != "seed"}
            rows.append(
                _row(
                    checkpoint,
                    row_id=f"{label}-factor-reequilibrated-{arm['label']}",
                    profile="reequilibrated_energy",
                    task_names=["reequilibrated_energy"],
                    protocol=f"factor_reequilibrated/{arm['label']}",
                    comparison_kind="re_equilibrated",
                    seed=int(arm["seed"]),
                    resources=resources,
                    factor_arm=factor_arm,
                    **shape,
                )
            )
        diagnostics = dict(grid["checkpoint_diagnostics"])
        snapshot = _scaled_snapshot(diagnostics, smoke)
        task_names = list(diagnostics["task_names"])
        rows.append(
            _row(
                checkpoint,
                row_id=f"{label}-checkpoint-diagnostics",
                profile="checkpoint_diagnostics",
                task_names=task_names,
                protocol="checkpoint_diagnostics",
                comparison_kind="checkpoint_diagnostics",
                seed=int(diagnostics["seed"]),
                resources=resources,
                n_draws=1,
                record_capacity=snapshot["n_walkers"],
                diagnostic_samples=int(smoke["diagnostic_samples"] if smoke else diagnostics["n_samples"]),
                **snapshot,
            )
        )
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise DiagnosticPlanError("expanded diagnostic row ids are not unique")
    return rows


def build_manifest(
    grid: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    grid_path: str | Path,
    source_map_path: str | Path,
    scale: str,
    evaluation_git_sha: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one content-addressed immutable plan manifest."""

    _require_git_sha(evaluation_git_sha, "evaluation_git_sha")
    rows = expand_rows(grid, checkpoints, scale=scale)
    repo_root = STUDY_DIR.parents[2]
    production_grid = STUDY_DIR / "configs" / "production_grid.yaml"
    manifest: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "study": grid["study"],
        "scale": scale,
        "evaluation_git_sha": evaluation_git_sha,
        "created_at": created_at or datetime.now(ZoneInfo(STUDY_TIMEZONE)).isoformat(),
        "grid_path": str(Path(grid_path).resolve()),
        "grid_sha256": file_sha256(grid_path),
        "source_map_sha256": file_sha256(source_map_path),
        "base_eval_config": grid["base_eval_config"],
        "base_eval_config_sha256": file_sha256(repo_root / str(grid["base_eval_config"])),
        "overlay_config": grid["overlay_config"],
        "overlay_config_sha256": file_sha256(repo_root / str(grid["overlay_config"])),
        "production_grid_sha256_before": file_sha256(production_grid),
        "production_run_mutation_authorized": False,
        "checkpoint_reporting": "report_both_without_selection",
        "scale_overrides": dict(grid["smoke_scale"]) if scale == "smoke" else {},
        "checkpoints": list(checkpoints),
        "rows": rows,
    }
    identity = dict(manifest)
    identity.pop("created_at")
    manifest["plan_sha256"] = canonical_sha256(identity)
    return manifest


def write_plan(manifest: Mapping[str, Any], *, results_root: str | Path, attempt_id: str) -> Path:
    """Write one new plan attempt without replacing an existing attempt."""

    output = layout.plan_attempt_dir(results_root, attempt_id)
    output.mkdir(parents=True, exist_ok=False)
    layout.write_json(output / layout.MANIFEST_FILENAME, manifest)
    with (output / layout.ROWS_FILENAME).open("x", encoding="utf-8", newline="") as handle:
        fields = [
            "row_id",
            "checkpoint_label",
            "profile",
            "protocol",
            "comparison_kind",
            "seed",
            "n_walkers",
            "n_draws",
            "burn_in",
            "stride",
            "task_names",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in manifest["rows"]:
            writer.writerow({**{key: row[key] for key in fields[:-1]}, "task_names": ",".join(row["task_names"])})
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_PLAN), attempt_id)
    return output


def read_manifest(results_root: str | Path, attempt_id: str) -> dict[str, Any]:
    """Read one exact plan attempt and verify its content identity."""

    path = layout.manifest_path(results_root, attempt_id)
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic plan attempt does not exist: {path}")
    manifest = layout.read_json(path)
    if manifest.get("schema") != PLAN_SCHEMA:
        raise DiagnosticPlanError(f"unexpected diagnostic plan schema in {path}")
    identity = dict(manifest)
    claimed = identity.pop("plan_sha256", None)
    identity.pop("created_at", None)
    actual = canonical_sha256(identity)
    if claimed != actual:
        raise DiagnosticPlanError(f"diagnostic plan hash mismatch: claimed={claimed}, actual={actual}")
    return manifest


def row_by_id(manifest: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    """Return exactly one planned row."""

    matches = [row for row in manifest["rows"] if row["row_id"] == row_id]
    if len(matches) != 1:
        raise DiagnosticPlanError(f"expected one row {row_id!r}, found {len(matches)}")
    return dict(matches[0])


def _row(
    checkpoint: Mapping[str, Any],
    *,
    row_id: str,
    profile: str,
    task_names: Sequence[str],
    protocol: str,
    comparison_kind: str,
    seed: int,
    n_walkers: int,
    n_draws: int,
    burn_in: int,
    stride: int,
    record_capacity: int,
    diagnostic_samples: int,
    resources: Mapping[str, Any],
    factor_arm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "kind": "diagnostic_eval",
        "stage": layout.STAGE_EVAL,
        "profile": profile,
        "task_names": list(task_names),
        "protocol": protocol,
        "comparison_kind": comparison_kind,
        "seed": int(seed),
        "n_walkers": int(n_walkers),
        "n_draws": int(n_draws),
        "burn_in": int(burn_in),
        "stride": int(stride),
        "record_capacity": int(record_capacity),
        "diagnostic_samples": int(diagnostic_samples),
        "factor_arm": None if factor_arm is None else dict(factor_arm),
        "checkpoint_label": checkpoint["label"],
        "checkpoint_step": checkpoint["completed_updates"],
        "checkpoint_model_sha256": checkpoint["model_sha256"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "checkpoint_complete_sha256": checkpoint["complete_sha256"],
        "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "checkpoint_source_git_sha": checkpoint["source_git_sha"],
        "checkpoint_source_tpen_version": checkpoint["source_tpen_version"],
        "checkpoint_source_dir": checkpoint["source_dir"],
        "resources": dict(resources),
    }


def _scaled_chain(value: Mapping[str, Any], smoke: Mapping[str, Any] | None) -> dict[str, int]:
    n_walkers = int(smoke["n_walkers"] if smoke else value["n_walkers"])
    n_draws = int(smoke["n_draws"] if smoke else value["n_draws"])
    return {
        "n_walkers": n_walkers,
        "n_draws": n_draws,
        "burn_in": int(smoke["burn_in"] if smoke else value["burn_in"]),
        "stride": int(smoke["stride"] if smoke else value["stride"]),
        "record_capacity": n_walkers * n_draws,
        "diagnostic_samples": int(smoke["diagnostic_samples"] if smoke else min(n_walkers, 4096)),
    }


def _scaled_snapshot(value: Mapping[str, Any], smoke: Mapping[str, Any] | None) -> dict[str, int]:
    return {
        "n_walkers": int(smoke["n_walkers"] if smoke else value["n_walkers"]),
        "burn_in": int(smoke["burn_in"] if smoke else value["burn_in"]),
        "stride": int(smoke["stride"] if smoke else value["stride"]),
    }


def _validate_chain(value: Mapping[str, Any], *, seeds_key: str) -> None:
    _require_text(value["name"], "protocol.name")
    _require_text(value["comparison_kind"], "protocol.comparison_kind")
    seeds = value[seeds_key]
    if seeds_key == "seeds":
        if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds:
            raise DiagnosticPlanError("protocol seeds must be a non-empty sequence")
        for seed in seeds:
            _require_nonnegative_int(seed, "protocol seed")
    else:
        _require_nonnegative_int(seeds, "protocol seed")
    for key in ("n_walkers", "n_draws", "stride"):
        _require_positive_int(value[key], f"protocol.{key}")
    _require_nonnegative_int(value["burn_in"], "protocol.burn_in")


def _validate_chain_shape(value: Mapping[str, Any], *, name: str) -> None:
    for key in ("n_walkers", "n_draws", "stride"):
        _require_positive_int(value[key], f"{name}.{key}")
    _require_nonnegative_int(value["burn_in"], f"{name}.burn_in")


def _validate_snapshot(value: Mapping[str, Any], *, name: str, seed: bool) -> None:
    if seed:
        _require_nonnegative_int(value["seed"], f"{name}.seed")
    for key in ("n_walkers", "stride"):
        _require_positive_int(value[key], f"{name}.{key}")
    _require_nonnegative_int(value["burn_in"], f"{name}.burn_in")


def _mapping_sequence(value: Any, name: str, keys: frozenset[str]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise DiagnosticPlanError(f"{name} must be a non-empty sequence")
    rows = []
    for index, row in enumerate(value):
        _require_exact_keys(row, keys, f"{name}[{index}]")
        rows.append(dict(row))
    return rows


def _require_exact_keys(value: Any, keys: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise DiagnosticPlanError(f"{name} must be a mapping")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise DiagnosticPlanError(f"{name} keys mismatch: missing={missing}, unknown={unknown}")


def _require_text(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise DiagnosticPlanError(f"{name} must be non-empty")
    return text


def _require_sha256(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DiagnosticPlanError(f"{name} must be a lowercase SHA-256")
    return text


def _require_git_sha(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise DiagnosticPlanError(f"{name} must be a lowercase full Git SHA")
    return text


def _require_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DiagnosticPlanError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    resolved = _require_nonnegative_int(value, name)
    if resolved == 0:
        raise DiagnosticPlanError(f"{name} must be positive")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--scale", choices=("production", "smoke"), default="production")
    parser.add_argument("--evaluation-git-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    grid = load_grid(args.grid)
    sources = load_sources(args.sources)
    checkpoints = reconcile_checkpoint_sources(grid, sources)
    manifest = build_manifest(
        grid,
        checkpoints,
        grid_path=args.grid,
        source_map_path=args.sources,
        scale=args.scale,
        evaluation_git_sha=args.evaluation_git_sha,
    )
    write_plan(manifest, results_root=args.results_root, attempt_id=args.attempt_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiagnosticPlanError",
    "build_manifest",
    "canonical_sha256",
    "expand_rows",
    "file_sha256",
    "load_grid",
    "load_sources",
    "read_manifest",
    "reconcile_checkpoint_sources",
    "row_by_id",
    "write_plan",
]
