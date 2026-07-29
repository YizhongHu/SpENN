"""Derived source read-set and checkpoint evidence for V4 references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from strict_data import iter_jsonl, load_json

EVIDENCE_INPUT_SCHEMA_VERSION = (
    "pair-stability-v4/reference-evidence-inputs/v2"
)
CHECKPOINT_EVIDENCE_SCHEMA_VERSION = (
    "pair-stability-v4/checkpoint-evidence/v1"
)
FANOUT_ATTEMPTS = {
    "01_train": "train",
    "02_validation": "validation",
    "06_final_train": "final_train",
    "07_final_eval": "final_eval",
}


def source_snapshot(path: Path) -> tuple[int, int, int, str]:
    """Return the source file facts relevant to an immutable freeze."""

    stat = path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_mode,
        _sha256_file(path),
    )


def discovery_anchor_paths(
    root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[Path, ...]:
    """Return fixed files that discover the complete reference read set."""

    grid = root / "00_grid" / attempts["grid"]
    paths = [
        grid / name
        for name in (
            "manifest.json",
            "unblind.json",
            "grid.yaml",
            "train_config.yaml",
            "validation_config.yaml",
        )
    ]
    for stage, attempt_key in FANOUT_ATTEMPTS.items():
        plan = root / stage / "stage_plans" / attempts[attempt_key]
        paths.extend(
            plan / name
            for name in (
                "stage_manifest.json",
                "tasks.jsonl",
                "execution_records.jsonl",
            )
        )
    paths.extend(
        (
            root
            / "03_collect"
            / attempts["collection"]
            / "collection_report.json",
            root
            / "03_collect"
            / attempts["collection"]
            / "summary.csv",
            root
            / "04_select"
            / attempts["selection"]
            / "selection_report.json",
            root
            / "04_select"
            / attempts["selection"]
            / "champions.csv",
            root
            / "05_final_grid"
            / attempts["final_grid"]
            / "final_jobs.csv",
            root
            / "05_final_grid"
            / attempts["final_grid"]
            / "manifest.json",
            root
            / "05_final_grid"
            / attempts["final_grid"]
            / "manifest.yaml",
            root
            / "08_final_collect"
            / attempts["final_collect"]
            / "manifest.yaml",
            root
            / "09_final_report"
            / attempts["report"]
            / "final_report.json",
        )
    )
    canonical = tuple(
        sorted(
            (_required_regular_file(path, root=root) for path in paths),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if len(canonical) != len(set(canonical)):
        raise ValueError("reference discovery anchors are duplicated")
    return canonical


def evidence_input_receipt(
    root: Path,
    *,
    attempts: Mapping[str, str],
    protected_paths: Sequence[Path],
) -> dict[str, Any]:
    """Snapshot the exact derived protected and audit-evidence read set."""

    evidence_paths, checkpoint_contracts, directory_projections = (
        _evidence_input_contract(
            root,
            attempts=attempts,
        )
    )
    roles: dict[Path, str] = {}
    for path in protected_paths:
        canonical = _required_regular_file(path, root=root)
        roles[canonical] = "protected_artifact"
    for path in evidence_paths:
        canonical = _required_regular_file(path, root=root)
        if canonical in roles:
            raise ValueError(f"reference read-set roles overlap: {canonical}")
        roles[canonical] = "audit_evidence"
    rows: list[dict[str, Any]] = []
    for path in sorted(
        roles,
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        stat = path.stat()
        rows.append(
            {
                "role": roles[path],
                "source_path": path.relative_to(root).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "mode": int(stat.st_mode),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema_version": EVIDENCE_INPUT_SCHEMA_VERSION,
        "file_count": len(rows),
        "files": rows,
        "checkpoint_contracts": checkpoint_contracts,
        "directory_projections": directory_projections,
        "directory_aggregate_sha256": _canonical_sha256(
            directory_projections
        ),
        "aggregate_sha256": _canonical_sha256(rows),
    }


def _evidence_input_contract(
    root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[
    tuple[Path, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paths: list[Path] = []
    checkpoint_contracts: list[dict[str, Any]] = []
    projected_directories: list[tuple[str, Path]] = [
        (
            "grid_jobs",
            root / "00_grid" / attempts["grid"] / "jobs",
        ),
        (
            "final_grid_jobs",
            root / "05_final_grid" / attempts["final_grid"] / "jobs",
        ),
    ]
    for stage, attempt_key in FANOUT_ATTEMPTS.items():
        tasks_path = (
            root
            / stage
            / "stage_plans"
            / attempts[attempt_key]
            / "tasks.jsonl"
        )
        for task in iter_jsonl(tasks_path):
            if not isinstance(task, dict):
                raise ValueError(f"{stage} task row is not an object")
            task_id = str(task.get("task_id") or "")
            run_id = _safe_component(
                task.get("run_id"),
                f"{stage} evidence run_id",
            )
            result_value = task.get("result_dir")
            if not isinstance(result_value, str) or not result_value:
                raise ValueError(f"{stage} task has no result_dir")
            result_dir = Path(result_value)
            expected_result_dir = (
                root / stage / run_id / attempts[attempt_key]
            )
            if result_dir != expected_result_dir:
                raise ValueError(
                    f"{stage} evidence result_dir differs from task identity"
                )
            paths.extend(
                (
                    result_dir / "status.json",
                    result_dir / "launcher_status.json",
                    result_dir / "metrics.jsonl",
                    result_dir / "metadata.json",
                    result_dir / "run_start.json",
                )
            )
            if stage in {"01_train", "02_validation"}:
                paths.append(result_dir / "source_grid_attempt.json")
            else:
                paths.extend(
                    (
                        result_dir / "source_final_grid_attempt.json",
                        result_dir / "source_final_job.json",
                        result_dir / "source_champion.json",
                    )
                )
            if stage == "02_validation":
                paths.append(result_dir / "source_train_attempt.json")
            if stage in {"02_validation", "07_final_eval"}:
                diagnostics_index = (
                    result_dir / "diagnostics" / "index.json"
                )
                paths.append(diagnostics_index)
                projected_directories.extend(
                    _diagnostic_output_directories(
                        root,
                        stage=stage,
                        result_dir=result_dir,
                        index_path=diagnostics_index,
                    )
                )
            if stage == "06_final_train":
                paths.append(result_dir / "selected_checkpoint.json")
            if stage == "07_final_eval":
                paths.extend(
                    (
                        result_dir / "source_final_train_attempt.json",
                        result_dir / "evaluated_checkpoint.json",
                    )
                )

            completion = task.get("completion")
            if not isinstance(completion, dict):
                raise ValueError(f"{stage} task completion is not an object")
            status_path = Path(str(completion.get("status_path") or ""))
            if status_path != result_dir / "status.json":
                raise ValueError(
                    f"{stage} evidence status path differs from task identity"
                )
            logs = task.get("logs")
            if logs != [str(result_dir / "launcher_status.json")]:
                raise ValueError(
                    f"{stage} evidence launcher path differs from task identity"
                )
            if (
                completion.get("policy")
                == "status_completed_with_checkpoint"
            ):
                pointer = Path(
                    str(completion.get("checkpoint_path") or "")
                )
                if pointer != result_dir / "checkpoints" / "latest.json":
                    raise ValueError(
                        f"{stage} evidence checkpoint path differs "
                        "from task identity"
                    )
                contract, checkpoint_paths = _checkpoint_evidence_contract(
                    root,
                    task_id=task_id,
                    pointer=pointer,
                )
                checkpoint_contracts.append(contract)
                paths.extend(checkpoint_paths)

    report_dir = root / "09_final_report" / attempts["report"]
    report = _read_json_object(report_dir / "final_report.json")
    figures = report.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError("final report figure contract is empty")
    paths.extend(
        report_dir / "figures" / _safe_nested_path(name)
        for name in figures
    )
    canonical = [
        _required_regular_file(path, root=root)
        for path in paths
    ]
    if len(set(canonical)) != len(canonical):
        raise ValueError("raw audit evidence input paths are duplicated")
    ordered_paths = tuple(
        sorted(canonical, key=lambda path: path.relative_to(root).as_posix())
    )
    ordered_contracts = sorted(
        checkpoint_contracts,
        key=lambda row: str(row["task_id"]),
    )
    directory_projections = [
        _directory_projection(root, role=role, directory=directory)
        for role, directory in sorted(
            projected_directories,
            key=lambda row: (row[1].relative_to(root).as_posix(), row[0]),
        )
    ]
    projection_paths = [
        str(row["source_path"]) for row in directory_projections
    ]
    if len(set(projection_paths)) != len(projection_paths):
        raise ValueError("directory projection paths are duplicated")
    return ordered_paths, ordered_contracts, directory_projections


def _diagnostic_output_directories(
    root: Path,
    *,
    stage: str,
    result_dir: Path,
    index_path: Path,
) -> list[tuple[str, Path]]:
    """Resolve task output directories declared by one diagnostic index."""

    index = _read_json_object(index_path)
    tasks = index.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"diagnostic task index is invalid: {index_path}")
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for row in tasks:
        if not isinstance(row, dict):
            raise ValueError(f"diagnostic task row is invalid: {index_path}")
        name = _safe_component(
            row.get("name"),
            f"{stage} diagnostic task name",
        )
        if name in names:
            raise ValueError(f"diagnostic task names are duplicated: {index_path}")
        names.add(name)
        output_value = row.get("output_dir")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError(
                f"diagnostic output directory is invalid: {index_path}"
            )
        output = Path(output_value)
        expected = result_dir / name
        if output != expected:
            raise ValueError(
                f"diagnostic output directory differs from task identity: "
                f"{index_path}"
            )
        _required_directory(output, root=root)
        result.append((f"{stage}_diagnostic_output", output))
    return result


def _directory_projection(
    root: Path,
    *,
    role: str,
    directory: Path,
) -> dict[str, Any]:
    """Snapshot recursive entry names and filesystem types for one directory."""

    canonical = _required_directory(directory, root=root)
    entries: list[dict[str, str]] = []
    for path in canonical.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"directory projection contains a symlink: {path}"
            )
        if path.is_file():
            entry_type = "file"
        elif path.is_dir():
            entry_type = "directory"
        else:
            raise ValueError(
                f"directory projection contains a special entry: {path}"
            )
        entries.append(
            {
                "path": path.relative_to(canonical).as_posix(),
                "type": entry_type,
            }
        )
    entries.sort(key=lambda row: str(row["path"]))
    return {
        "role": role,
        "source_path": canonical.relative_to(root).as_posix(),
        "entries": entries,
        "entries_sha256": _canonical_sha256(entries),
    }


def _checkpoint_evidence_contract(
    root: Path,
    *,
    task_id: str,
    pointer: Path,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Resolve the exact checkpoint files read by the terminal audit."""

    pointer = _required_regular_file(pointer, root=root)
    try:
        pointer_value = _read_json_object(pointer)
        step = int(pointer_value["step"])
        directory_name = str(pointer_value["checkpoint_dir"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint pointer {pointer}: {exc}") from exc
    if (
        Path(directory_name).is_absolute()
        or ".." in Path(directory_name).parts
        or directory_name != f"step_{step:06d}"
    ):
        raise ValueError(f"unsafe checkpoint directory in {pointer}")
    concrete = pointer.parent / directory_name
    if concrete.is_symlink() or not concrete.is_dir():
        raise ValueError(f"checkpoint directory is unavailable: {concrete}")
    complete = _required_regular_file(concrete / "COMPLETE", root=root)
    manifest_path = _required_regular_file(
        concrete / "manifest.json",
        root=root,
    )
    manifest = _read_json_object(manifest_path)
    if manifest.get("step") != step:
        raise ValueError(f"checkpoint manifest step differs: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(
            f"checkpoint manifest files are invalid: {manifest_path}"
        )
    payloads: list[Path] = []
    for value in files.values():
        relative = _safe_nested_path(value)
        payloads.append(
            _required_regular_file(concrete / relative, root=root)
        )
    if len(set(payloads)) != len(payloads):
        raise ValueError(
            f"checkpoint manifest payloads are duplicated: {manifest_path}"
        )

    def relative(path: Path) -> str:
        return path.relative_to(root).as_posix()

    contract = {
        "schema_version": CHECKPOINT_EVIDENCE_SCHEMA_VERSION,
        "task_id": task_id,
        "pointer_path": relative(pointer),
        "pointer_projection": pointer_value,
        "step": step,
        "checkpoint_dir": relative(concrete),
        "manifest_projection": manifest,
        "payload_paths": sorted(relative(path) for path in payloads),
        "file_sha256": {
            relative(path): _sha256_file(path)
            for path in (
                pointer,
                complete,
                manifest_path,
                *sorted(payloads),
            )
        },
    }
    return contract, (
        pointer,
        complete,
        manifest_path,
        *sorted(payloads),
    )


def _required_regular_file(path: Path, *, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"read-set input is not a regular file: {path}")
    canonical = path.resolve(strict=True)
    if canonical == root or root not in canonical.parents:
        raise ValueError(f"read-set input escapes lineage root: {path}")
    return canonical


def _required_directory(path: Path, *, root: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"read-set directory is unavailable: {path}")
    canonical = path.resolve(strict=True)
    if canonical == root or root not in canonical.parents:
        raise ValueError(f"read-set directory escapes lineage root: {path}")
    return canonical


def _safe_component(value: object, label: str) -> str:
    text = str(value or "")
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
    ):
        raise ValueError(f"invalid {label}: {text!r}")
    return text


def _read_json_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_component(value: object, name: str) -> str:
    text = str(value or "")
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"invalid {name}: {text!r}")
    return text


def _safe_nested_path(value: object) -> Path:
    path = Path(str(value))
    if (
        not str(value)
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"unsafe nested artifact path: {value!r}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
