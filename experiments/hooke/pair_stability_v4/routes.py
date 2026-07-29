"""Define pinned V4-0 subprocess routes without importing legacy modules."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from roots import validate_lineage_id
from strict_data import StrictDataError, load_json, load_yaml

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
V3_STUDY_DIR = STUDY_DIR.parent / "pair_stability_v3"
ROUTES_PATH = STUDY_DIR / "legacy_routes_v1.json"
SOURCE_MANIFEST_PATH = STUDY_DIR / "legacy_source_v1.json"
ROUTES_SCHEMA_VERSION = "pair-stability-v4/legacy-routes/v1"
SOURCE_SCHEMA_VERSION = "pair-stability-v4/legacy-source/v1"
ROUTE_KINDS = frozenset({"local", "fanout"})
OUTPUT_ATTEMPT_RULES = frozenset({"explicit", "grid_input"})

ROLE_TO_STAGE = {
    "screen_plan": "00_grid",
    "screen_train": "01_train",
    "screen_eval": "02_validation",
    "screen_collect": "03_collect",
    "select": "04_select",
    "confirm_plan": "05_final_grid",
    "confirm_train": "06_final_train",
    "confirm_eval": "07_final_eval",
    "confirm_collect": "08_final_collect",
    "report": "09_final_report",
}
ROLE_TO_LEGACY_SCRIPT = {
    "screen_plan": "experiments/hooke/pair_stability_v3/plan.py",
    "screen_train": "experiments/hooke/pair_stability_v3/train.py",
    "screen_eval": "experiments/hooke/pair_stability_v3/validate.py",
    "screen_collect": "experiments/hooke/pair_stability_v3/collect.py",
    "select": "experiments/hooke/pair_stability_v3/select_champions.py",
    "confirm_plan": "experiments/hooke/pair_stability_v3/final_plan.py",
    "confirm_train": "experiments/hooke/pair_stability_v3/final_train.py",
    "confirm_eval": "experiments/hooke/pair_stability_v3/final_eval.py",
    "confirm_collect": "experiments/hooke/pair_stability_v3/final_collect.py",
    "report": "experiments/hooke/pair_stability_v3/final_report.py",
}
CONTROL_ONLY_INPUTS = {
    "confirm_plan": frozenset({"grid"}),
}
EXPECTED_LEGACY_SOURCE_PATHS = (
    *ROLE_TO_LEGACY_SCRIPT.values(),
    "experiments/hooke/pair_stability_v3/launch.py",
    "experiments/hooke/pair_stability_v3/plot.py",
    "experiments/hooke/pair_stability_v3/stats.py",
    "experiments/hooke/pair_stability_v3/utils/__init__.py",
    "experiments/hooke/pair_stability_v3/utils/ancestry.py",
    "experiments/hooke/pair_stability_v3/utils/config.py",
    "experiments/hooke/pair_stability_v3/utils/io.py",
    "experiments/hooke/pair_stability_v3/utils/layout.py",
    "experiments/hooke/pair_stability_v3/utils/naming.py",
    "experiments/hooke/pair_stability_v3/utils/overrides.py",
    "experiments/hooke/pair_stability_v3/utils/seeds.py",
    "experiments/hooke/pair_stability_v3/utils/time.py",
    "experiments/hooke/pair_stability_v3/configs/smoke.yaml",
    "experiments/hooke/pair_stability_v3/configs/pair_stability.yaml",
    "experiments/hooke/pair_stability_v3/configs/pair_validation.yaml",
)
ROUTE_FIELDS = frozenset(
    {
        "logical_role",
        "physical_stage",
        "legacy_script",
        "kind",
        "required_input_attempts",
        "required_configs",
        "output_attempt_rule",
        "arguments",
    }
)
ROUTE_PLACEHOLDERS = frozenset(
    {"{results_root}", "{output_attempt}", "{repo_root}"}
)
RUNTIME_PATHS = (
    "run.py",
    "pyproject.toml",
    "uv.lock",
)
ROUTE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class LegacyStageRoute:
    """Describe one versioned V4-0 subprocess route without importing v3."""

    logical_role: str
    physical_stage: str
    legacy_script: str
    kind: str
    required_input_attempts: tuple[str, ...]
    required_configs: tuple[str, ...]
    output_attempt_rule: str
    arguments: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LegacyStageRoute":
        """Build and validate one route from serialized route data."""

        unknown = set(data) - ROUTE_FIELDS
        missing = ROUTE_FIELDS - set(data)
        if unknown or missing:
            raise ValueError(
                f"route fields mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        route = cls(
            logical_role=_required_string(data, "logical_role"),
            physical_stage=_required_string(data, "physical_stage"),
            legacy_script=_required_string(data, "legacy_script"),
            kind=_required_string(data, "kind"),
            required_input_attempts=_string_tuple(
                data.get("required_input_attempts"), "required_input_attempts"
            ),
            required_configs=_string_tuple(
                data.get("required_configs"), "required_configs"
            ),
            output_attempt_rule=_required_string(data, "output_attempt_rule"),
            arguments=_string_tuple(data.get("arguments"), "arguments"),
        )
        route.validate()
        return route

    def validate(self) -> None:
        """Validate closed route fields and placeholder use."""

        expected_stage = ROLE_TO_STAGE.get(self.logical_role)
        if expected_stage is None:
            raise ValueError(f"unknown V4-0 logical role: {self.logical_role!r}")
        if self.physical_stage != expected_stage:
            raise ValueError(
                f"role {self.logical_role!r} must map to {expected_stage!r}, "
                f"not {self.physical_stage!r}"
            )
        expected_script = ROLE_TO_LEGACY_SCRIPT[self.logical_role]
        if self.legacy_script != expected_script:
            raise ValueError(
                f"role {self.logical_role!r} must invoke {expected_script!r}, "
                f"not {self.legacy_script!r}"
            )
        if self.kind not in ROUTE_KINDS:
            raise ValueError(f"unknown route kind: {self.kind!r}")
        if self.output_attempt_rule not in OUTPUT_ATTEMPT_RULES:
            raise ValueError(
                f"unknown output attempt rule: {self.output_attempt_rule!r}"
            )
        if self.logical_role == "screen_train":
            if self.output_attempt_rule != "grid_input":
                raise ValueError("screen_train must use grid_input output rule")
        elif self.output_attempt_rule != "explicit":
            raise ValueError(
                f"{self.logical_role} must use explicit output attempt rule"
            )
        _require_unique(self.required_input_attempts, "required_input_attempts")
        _require_unique(self.required_configs, "required_configs")
        for name in (
            *self.required_input_attempts,
            *self.required_configs,
        ):
            if ROUTE_KEY_PATTERN.fullmatch(name) is None:
                raise ValueError(f"invalid route key: {name!r}")
        _validate_repository_relative(self.legacy_script, "legacy_script")
        if "--wait-job" in self.arguments or "--smoke" in self.arguments:
            raise ValueError("routes may not use legacy --wait-job or --smoke")

        seen_inputs: list[str] = []
        seen_configs: list[str] = []
        for argument in self.arguments:
            token = _placeholder(argument)
            if token is None:
                if "{" in argument or "}" in argument:
                    raise ValueError(
                        f"partial or unknown route placeholder: {argument!r}"
                    )
                continue
            kind, name = token
            if kind == "input":
                seen_inputs.append(name)
            elif kind == "config":
                seen_configs.append(name)
        expected_inputs = set(seen_inputs) | set(
            CONTROL_ONLY_INPUTS.get(self.logical_role, ())
        )
        if expected_inputs != set(self.required_input_attempts):
            raise ValueError(
                f"input placeholders/control inputs {sorted(expected_inputs)!r} "
                f"do not match declared {self.required_input_attempts!r}"
            )
        if sorted(seen_configs) != sorted(self.required_configs):
            raise ValueError(
                f"config placeholders {seen_configs!r} do not match declared "
                f"{self.required_configs!r}"
            )
        _require_unique(seen_inputs, "input placeholders")
        _require_unique(seen_configs, "config placeholders")
        if self.arguments.count("{results_root}") != 1:
            raise ValueError("every route must contain results_root exactly once")
        expected_output_count = 0 if self.output_attempt_rule == "grid_input" else 1
        if self.arguments.count("{output_attempt}") != expected_output_count:
            raise ValueError(
                f"{self.logical_role} must contain output_attempt "
                f"{expected_output_count} time(s)"
            )
        expected_repo_count = 1 if self.kind == "fanout" else 0
        if self.arguments.count("{repo_root}") != expected_repo_count:
            raise ValueError(
                f"{self.logical_role} must contain repo_root "
                f"{expected_repo_count} time(s)"
            )


def load_routes(path: Path = ROUTES_PATH) -> Mapping[str, LegacyStageRoute]:
    """Load exactly one unique route for every V4-0 logical stage."""

    payload = _read_json_object(path)
    if set(payload) != {"schema_version", "routes"}:
        raise ValueError(f"invalid route manifest fields: {sorted(payload)}")
    if payload.get("schema_version") != ROUTES_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported route schema: {payload.get('schema_version')!r}"
        )
    rows = payload.get("routes")
    if not isinstance(rows, list):
        raise ValueError("route manifest routes must be a list")
    routes: dict[str, LegacyStageRoute] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each route must be an object")
        route = LegacyStageRoute.from_dict(row)
        if route.logical_role in routes:
            raise ValueError(f"duplicate route: {route.logical_role}")
        routes[route.logical_role] = route
    if set(routes) != set(ROLE_TO_STAGE):
        raise ValueError(
            f"route role set mismatch; missing={sorted(set(ROLE_TO_STAGE) - set(routes))}, "
            f"extra={sorted(set(routes) - set(ROLE_TO_STAGE))}"
        )
    if len({route.physical_stage for route in routes.values()}) != len(routes):
        raise ValueError("physical stage mappings must be one-to-one")
    if {route.legacy_script for route in routes.values()} != set(
        ROLE_TO_LEGACY_SCRIPT.values()
    ):
        raise ValueError("route scripts do not match the pinned ten-stage closure")
    return MappingProxyType(routes)


def render_legacy_argv(
    route: LegacyStageRoute,
    *,
    results_root: Path,
    output_attempt: str,
    input_attempts: Mapping[str, str],
    config_paths: Mapping[str, Path],
    repo_root: Path,
) -> tuple[str, ...]:
    """Render complete pinned legacy argv from typed V4-owned inputs."""

    route.validate()
    root = Path(results_root)
    repo_root = Path(repo_root).resolve(strict=True)
    if not root.is_absolute():
        raise ValueError("results_root must be absolute")
    output_attempt = validate_lineage_id(output_attempt)
    normalized_inputs = {
        str(key): validate_lineage_id(str(value))
        for key, value in input_attempts.items()
    }
    if set(normalized_inputs) != set(route.required_input_attempts):
        raise ValueError(
            f"{route.logical_role} input attempts mismatch; "
            f"expected={list(route.required_input_attempts)!r}, "
            f"received={sorted(normalized_inputs)!r}"
        )
    normalized_configs = {
        str(key): Path(value).resolve(strict=False)
        for key, value in config_paths.items()
    }
    if set(normalized_configs) != set(route.required_configs):
        raise ValueError(
            f"{route.logical_role} config paths mismatch; "
            f"expected={list(route.required_configs)!r}, "
            f"received={sorted(normalized_configs)!r}"
        )
    if route.output_attempt_rule == "grid_input":
        grid_attempt = normalized_inputs.get("grid")
        if output_attempt != grid_attempt:
            raise ValueError(
                "screen_train output attempt must equal its grid input attempt"
            )

    script = repo_root / route.legacy_script
    _require_beneath(script.resolve(strict=False), repo_root, "legacy script")
    rendered: list[str] = []
    for argument in route.arguments:
        token = _placeholder(argument)
        if token is None:
            rendered.append(argument)
            continue
        kind, name = token
        if kind == "fixed":
            values = {
                "results_root": str(root),
                "output_attempt": output_attempt,
                "repo_root": str(repo_root),
            }
            rendered.append(values[name])
        elif kind == "input":
            rendered.append(normalized_inputs[name])
        elif kind == "config":
            rendered.append(str(normalized_configs[name]))
        else:  # pragma: no cover - _placeholder is closed
            raise AssertionError(kind)
    return (sys.executable, "-B", str(script), *rendered)


def verify_legacy_source_manifest(
    repo_root: Path,
    *,
    manifest_path: Path = SOURCE_MANIFEST_PATH,
) -> tuple[str, ...]:
    """Return missing or digest-mismatched files in pinned v3 closure."""

    errors: list[str] = []
    try:
        manifest = load_legacy_source_manifest(manifest_path)
    except ValueError as exc:
        return (str(exc),)
    repo_root = Path(repo_root).resolve(strict=True)
    for entry in manifest["files"]:
        relative_path = str(entry["path"])
        path = repo_root / relative_path
        try:
            _require_beneath(path.resolve(strict=False), repo_root, "source path")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing pinned legacy source: {relative_path}")
            continue
        actual = _sha256_file(path)
        if actual != entry["sha256"]:
            errors.append(
                f"legacy source digest mismatch: {relative_path}: "
                f"{actual} != {entry['sha256']}"
            )
    return tuple(errors)


def load_legacy_source_manifest(path: Path = SOURCE_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate the pinned v3 study-closure manifest."""

    payload = _read_json_object(path)
    if set(payload) != {"schema_version", "study", "files"}:
        raise ValueError(f"invalid legacy source manifest fields: {sorted(payload)}")
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported legacy source schema: {payload.get('schema_version')!r}"
        )
    if payload.get("study") != "pair_stability_v3":
        raise ValueError("legacy source manifest study must be pair_stability_v3")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("legacy source manifest files must be a nonempty list")
    paths: list[str] = []
    normalized: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("legacy source entry requires only path and sha256")
        relative_path = _required_string(row, "path")
        digest = _required_string(row, "sha256")
        _validate_repository_relative(relative_path, "legacy source path")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA-256 for {relative_path}")
        paths.append(relative_path)
        normalized.append({"path": relative_path, "sha256": digest})
    _require_unique(paths, "legacy source paths")
    if set(paths) != set(EXPECTED_LEGACY_SOURCE_PATHS):
        raise ValueError(
            "legacy source closure mismatch; "
            f"missing={sorted(set(EXPECTED_LEGACY_SOURCE_PATHS) - set(paths))}, "
            f"extra={sorted(set(paths) - set(EXPECTED_LEGACY_SOURCE_PATHS))}"
        )
    return {**payload, "files": normalized}


def legacy_source_receipt(repo_root: Path) -> dict[str, Any]:
    """Return verified path digests and one aggregate v3 closure digest."""

    errors = verify_legacy_source_manifest(repo_root)
    if errors:
        raise ValueError("; ".join(errors))
    manifest = load_legacy_source_manifest()
    files = [dict(row) for row in manifest["files"]]
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "manifest_path": str(SOURCE_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": _sha256_file(SOURCE_MANIFEST_PATH),
        "closure_sha256": _aggregate_digest(files),
        "files": files,
    }


def runtime_source_receipt(repo_root: Path) -> dict[str, Any]:
    """Return provenance-only digests for evolving runtime dependencies."""

    repo_root = Path(repo_root).resolve(strict=True)
    tracked = _git_lines(repo_root, ["ls-files"])
    selected = sorted(
        path
        for path in tracked
        if path in RUNTIME_PATHS
        or (path.startswith("experiments/toolkit/") and path.endswith(".py"))
        or (path.startswith("spenn/") and path.endswith(".py"))
    )
    if not selected:
        raise RuntimeError("runtime source closure is empty")
    missing = [path for path in selected if not (repo_root / path).is_file()]
    if missing:
        raise RuntimeError(f"runtime source closure has missing files: {missing}")
    files = [
        {"path": relative, "sha256": _sha256_file(repo_root / relative)}
        for relative in selected
    ]
    git_status = _git_lines(
        repo_root, ["status", "--short", "--untracked-files=all"]
    )
    git_commit = _git_value(repo_root, ["rev-parse", "HEAD"])
    if not git_commit:
        raise RuntimeError("git rev-parse HEAD returned an empty commit")
    git_branch = _git_value(repo_root, ["branch", "--show-current"])
    receipt: dict[str, Any] = {
        "schema_version": "pair-stability-v4/runtime-source/v1",
        "closure_sha256": _aggregate_digest(files),
        "n_files": len(files),
        "git_commit": git_commit,
        "git_branch": git_branch or "(detached)",
        "dirty": bool(git_status),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "uv_project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
    }
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        receipt.update(
            {
                "torch_version": None,
                "torch_cuda_version": None,
                "cuda_available": False,
            }
        )
    else:
        receipt.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "torch_cuda_version": getattr(
                    getattr(torch, "version", None), "cuda", None
                ),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
    return receipt


def config_source_receipt(repo_root: Path) -> dict[str, Any]:
    """Return hashes for v4-owned configs and semantic source equivalence."""

    repo_root = Path(repo_root).resolve(strict=True)
    config_names = ("smoke.yaml", "pair_stability.yaml", "pair_validation.yaml")
    files = []
    for name in config_names:
        path = STUDY_DIR / "configs" / name
        files.append(
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256_file(path),
            }
        )
    errors = verify_v4_config_copies(repo_root)
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": "pair-stability-v4/config-source/v1",
        "closure_sha256": _aggregate_digest(files),
        "files": files,
    }


def verify_v4_config_copies(repo_root: Path) -> tuple[str, ...]:
    """Return semantic differences beyond approved v3-to-v4 substitutions."""

    repo_root = Path(repo_root).resolve(strict=True)
    errors: list[str] = []
    for name in ("smoke.yaml", "pair_stability.yaml", "pair_validation.yaml"):
        v3_path = V3_STUDY_DIR / "configs" / name
        v4_path = STUDY_DIR / "configs" / name
        try:
            v3 = load_yaml(v3_path)
            v4 = load_yaml(v4_path)
        except (OSError, StrictDataError) as exc:
            errors.append(f"cannot read config pair {name}: {exc}")
            continue
        expected = _approved_v4_config(v3, name=name)
        if expected != v4:
            errors.append(f"v4 config differs beyond approved substitutions: {name}")
            continue
        expected_text = (
            v3_path.read_text()
            .replace("pair_stability_v3", "pair_stability_v4")
            .replace("V3", "V4")
        )
        if expected_text != v4_path.read_text():
            errors.append(
                f"v4 config text/order differs beyond approved substitutions: {name}"
            )
    return tuple(errors)


def require_launcher_environment(
    route: LegacyStageRoute,
    repo_root: Path,
) -> None:
    """Reject fan-out dispatch outside expected Submitit launcher environment."""

    if route.kind != "fanout":
        return
    expected = (Path(repo_root) / ".venv-submitit").resolve(strict=False)
    executable = Path(sys.executable).resolve()
    if executable != expected and expected not in executable.parents:
        raise RuntimeError(
            "fan-out V4-0 dispatch must start inside .venv-submitit; "
            f"current executable is {executable}"
        )
    if "--wait-job" in route.arguments:
        raise RuntimeError("V4-0 fan-out routes forbid legacy --wait-job")


def _approved_v4_config(value: Any, *, name: str) -> Any:
    if not isinstance(value, dict):
        return value
    expected = json.loads(json.dumps(value))
    if name == "smoke.yaml":
        expected["study"] = "pair_stability_v4"
        expected["config"] = (
            "experiments/hooke/pair_stability_v4/configs/pair_stability.yaml"
        )
        expected["validation_config"] = (
            "experiments/hooke/pair_stability_v4/configs/pair_validation.yaml"
        )
        expected["results_root"] = "experiments/hooke/pair_stability_v4/results"
        return expected
    expected["study"]["name"] = "pair_stability_v4"
    stage = "01_train" if name == "pair_stability.yaml" else "02_validation"
    expected["run"]["root"] = (
        f"experiments/hooke/pair_stability_v4/results/{stage}"
    )
    return expected


def _placeholder(value: str) -> tuple[str, str] | None:
    if value in ROUTE_PLACEHOLDERS:
        return ("fixed", value[1:-1])
    if value.startswith("{input:") and value.endswith("}"):
        name = value[len("{input:") : -1]
        if name:
            return ("input", name)
    if value.startswith("{config:") and value.endswith("}"):
        name = value[len("{config:") : -1]
        if name:
            return ("config", name)
    return None


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain nonempty strings")
    return tuple(value)


def _require_unique(values: Sequence[str], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates")


def _validate_repository_relative(value: str, name: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative: {value}")


def _require_beneath(path: Path, root: Path, name: str) -> None:
    if path == root or root not in path.parents:
        raise ValueError(f"{name} escapes repository root: {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = load_json(Path(path))
    except (OSError, StrictDataError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_digest(files: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(files, key=lambda row: row["path"]):
        digest.update(entry["path"].encode())
        digest.update(b"\0")
        digest.update(entry["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _git_lines(repo_root: Path, arguments: Sequence[str]) -> list[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed with {result.returncode}: {detail}"
        )
    return [line for line in result.stdout.splitlines() if line]


def _git_value(repo_root: Path, arguments: Sequence[str]) -> str:
    return "\n".join(_git_lines(repo_root, arguments))
