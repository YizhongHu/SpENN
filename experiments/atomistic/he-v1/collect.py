"""Assemble planned He-v1 rows into one table keyed by immutable identity.

Two failures this stage exists to prevent, both of which have happened in this
repository or its receipts:

a missing value is not a zero and not a blank
    Every cell states its own presence. A blank cell parses to NaN and is then
    silently dropped, so a median over two of nine rows renders exactly like a
    median over nine; a zero is worse, because it participates. Aggregates
    therefore always carry ``n_present``/``n_absent``.

rows are joined on identity, not on position
    A row is identified by its run id, its launch attempt, the hash of the
    checkpoint it ran, the hash of its resolved config, its seed, and its GPU
    stratum -- requested AND delivered. A delivered/requested mismatch fails
    the row here as it does in the allocation, and an evaluation whose restored
    checkpoint does not hash to its training row's retained checkpoint fails
    too, because that number belongs to a different model than the one claimed.

Gating is delegated in full to ``gates.evaluate_atom_gates`` (layer L2); this
module adds no gate logic of its own. Before the tolerances are predeclared in
H-F1 every value gate legitimately reports ``absent`` with its observed value
retained -- that is the honest state of an ungated run, not a defect, and it is
never patched over with a placeholder threshold.

This module imports no ``tpen`` (``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import yaml

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import absence  # noqa: E402
import launch as launch_stage  # noqa: E402
import layout  # noqa: E402
import plan as plan_stage  # noqa: E402


def _load_gates() -> ModuleType:
    """Load layer L2's ``gates`` module by path.

    The study directory is not an importable package (its name contains a
    hyphen), so the checked-in experiment code loads its siblings by file
    location. ``gates.py`` is owned by another layer and is imported, never
    modified.
    """

    path = STUDY_DIR / "gates.py"
    spec = importlib.util.spec_from_file_location("he_v1_gates", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gates module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gates = _load_gates()

#: Energy columns retained for every evaluation row. ``local_energy_stderr`` is
#: an IID standard error and is labelled as such wherever it is rendered; it is
#: never presented as an MCSE.
ENERGY_METRIC_KEYS: tuple[str, ...] = (
    "local_energy_mean",
    "local_energy_stderr",
    "local_energy_variance",
    "local_energy_n_finite",
    "local_energy_nonfinite_count",
    "reference_energy",
    "energy_error",
    "energy_abs_error",
)

#: Every metric the gates read, retained whether or not it gated.
GATE_METRIC_KEYS: tuple[str, ...] = tuple(gates.ATOM_GATE_METRIC_KEYS)

COLLECTED_FILENAME = "collected.json"
ROWS_CSV = "rows.csv"
GATES_CSV = "gates.csv"


class CollectError(RuntimeError):
    """The collected rows do not form a table that can be reported honestly."""


def read_metrics_jsonl(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Read one ``metrics.jsonl`` into namespaced and flat mappings.

    Returns
    -------
    namespaced : dict
        ``"<namespace>/<key>" -> value`` for every logged record.
    flat : dict
        ``"<key>" -> value`` for keys that appear under exactly one namespace.
    ambiguous : list of str
        Keys logged under more than one namespace with differing values. They
        are excluded from ``flat`` rather than resolved by guesswork: picking
        one silently would attribute a number to the wrong task.
    """

    path = Path(path)
    namespaced: dict[str, Any] = {}
    by_key: dict[str, list[tuple[str, Any]]] = {}
    if not path.is_file():
        return {}, {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        record = json.loads(text)
        namespace = str(record.get("namespace") or "").strip("/")
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            full = f"{namespace}/{key}" if namespace else str(key)
            namespaced[full] = value
            by_key.setdefault(str(key), []).append((namespace, value))
    flat: dict[str, Any] = {}
    ambiguous: list[str] = []
    for key, entries in by_key.items():
        values = {json.dumps(value, sort_keys=True) for _namespace, value in entries}
        if len(values) == 1:
            flat[key] = entries[0][1]
        else:
            ambiguous.append(key)
    return namespaced, flat, sorted(ambiguous)


def file_sha256(path: str | Path) -> str | Any:
    """Return the SHA-256 of one file, or :data:`absence.ABSENT` when missing."""

    path = Path(path)
    if not path.is_file():
        return absence.ABSENT
    return plan_stage.file_sha256(path)


def directory_sha256(path: str | Path) -> str | Any:
    """Return a content hash of a directory tree, or absent when missing.

    The hash covers relative paths and file contents in sorted order, so it
    identifies the checkpoint rather than the moment it was written.
    """

    root = Path(path)
    if not root.is_dir():
        return absence.ABSENT
    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(plan_stage.file_sha256(file_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_gate_spec(path: str | Path | None, *, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the tolerance spec, from a file when given, else from the plan.

    An empty spec is a legitimate state, not a missing input: the tolerance
    numbers are predeclared in H-F1, and until then every value gate reports
    ``absent`` with its observed value retained.
    """

    if path is None:
        return dict(manifest.get("gate_spec") or {})
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise CollectError(f"gate spec {path} is not a mapping")
    return dict(payload)


def collect_row(
    row: Mapping[str, Any],
    *,
    results_root: Path,
    plan_attempt_id: str,
    manifest: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    extra_metric_keys: Sequence[str] = (),
    hash_checkpoints: bool = True,
) -> dict[str, Any]:
    """Collect one planned row into a table row with explicit absence."""

    row_id = str(row["row_id"])
    kind = str(row["kind"])
    result_dir = layout.row_dir(results_root, str(row["stage"]), row_id, plan_attempt_id)
    run_dir = result_dir / row_id
    reasons: list[str] = []

    receipt = _read_json_or_absent(result_dir / "allocation_receipt.json", reasons)
    row_record = _read_json_or_absent(result_dir / "row.json", reasons)
    metadata = _read_json_or_absent(run_dir / "metadata.json", reasons)
    status = _read_json_or_absent(run_dir / "status.json", reasons)
    namespaced, flat, ambiguous = read_metrics_jsonl(run_dir / "metrics.jsonl")
    if ambiguous:
        reasons.append(f"metric keys logged under several namespaces: {ambiguous}")
    if not namespaced:
        reasons.append("no metrics were logged")

    requested_stratum = str(row["resources"]["stratum"])
    delivered_device = _receipt_field(receipt, "delivered_device")
    delivered_matches = _receipt_field(receipt, "delivered_matches_requested")
    if absence.is_absent(delivered_matches):
        reasons.append("no allocation receipt: the delivered GPU was never verified")
    elif delivered_matches is not True:
        reasons.append(
            f"delivered device {delivered_device!r} does not match requested stratum "
            f"{requested_stratum!r}"
        )

    checkpoint_step = row.get("checkpoint_step")
    checkpoint_hash: Any = absence.ABSENT
    checkpoint_dir: Any = absence.ABSENT
    if kind == "eval":
        checkpoint_dir = launch_stage.checkpoint_dir_for_eval_row(
            results_root, row, plan_attempt_id, manifest=manifest
        )
        checkpoint_hash = (
            directory_sha256(checkpoint_dir) if hash_checkpoints else absence.ABSENT
        )
        if hash_checkpoints and absence.is_absent(checkpoint_hash):
            reasons.append(f"restored checkpoint is missing: {checkpoint_dir}")

    identity = {
        "row_id": row_id,
        "kind": kind,
        "seed": int(row["seed"]),
        "checkpoint_step": absence.cell(checkpoint_step),
        "chain": absence.cell(row.get("chain")),
        "chain_seed": absence.cell(row.get("chain_seed")),
        "run_id": absence.cell(_json_field(metadata, "run_id")),
        "plan_attempt_id": str(plan_attempt_id),
        "plan_hash": str(manifest["plan_hash"]),
        "launch_attempt_id": absence.cell(_json_field(row_record, "launch_attempt_id")),
        "job_id": absence.cell(_receipt_field(receipt, "job_id")),
        "hostname": absence.cell(_receipt_field(receipt, "hostname")),
        "requested_stratum": requested_stratum,
        "requested_constraint": str(row["resources"].get("constraint") or ""),
        "delivered_device": absence.cell(delivered_device),
        "partition": str(row["resources"]["partition"]),
        "config_sha256": absence.cell(file_sha256(run_dir / "resolved_config.yaml")),
        "checkpoint_dir": absence.cell(
            None if absence.is_absent(checkpoint_dir) else str(checkpoint_dir)
        ),
        "checkpoint_sha256": absence.cell(checkpoint_hash),
    }

    metric_keys = list(dict.fromkeys([*ENERGY_METRIC_KEYS, *GATE_METRIC_KEYS, *extra_metric_keys]))
    metrics = {key: absence.cell(flat.get(key)) for key in metric_keys}

    gate_rows: list[dict[str, Any]] = []
    if kind == "eval":
        for outcome in gates.evaluate_atom_gates(flat, spec=gate_spec):
            gate_rows.append(
                {
                    "name": outcome.name,
                    "status": outcome.status,
                    "value": absence.cell(outcome.value),
                    "threshold": absence.cell(outcome.threshold),
                    "reason": outcome.reason,
                }
            )
        failed_gates = [gate["name"] for gate in gate_rows if gate["status"] == "fail"]
        if failed_gates:
            reasons.append(f"failed gates: {failed_gates}")

    run_status = _json_field(status, "status")
    if absence.is_absent(run_status):
        reasons.append("no run status.json: the row did not finish")
    elif str(run_status) != "completed":
        reasons.append(f"run status is {run_status!r}")

    return {
        "identity": identity,
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "run_dir": str(run_dir),
        "result_dir": str(result_dir),
        "run_status": absence.cell(run_status),
        "metrics": metrics,
        "gates": gate_rows,
        "gate_counts": _gate_counts(gate_rows),
        "ambiguous_metric_keys": ambiguous,
        "namespaced_metric_count": len(namespaced),
    }


def cross_check_checkpoint_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail rows whose restored checkpoint disagrees across chains.

    Four chains over one checkpoint must have restored the same bytes. If two
    chains of the same (seed, step) hash differently, at least one number is
    attributed to a model that did not produce it.
    """

    by_checkpoint: dict[tuple[int, Any], dict[str, str]] = {}
    for row in rows:
        identity = row["identity"]
        if identity["kind"] != "eval":
            continue
        digest = absence.cell_value(identity["checkpoint_sha256"])
        if absence.is_absent(digest):
            continue
        key = (identity["seed"], absence.cell_value(identity["checkpoint_step"]))
        by_checkpoint.setdefault(key, {})[identity["row_id"]] = str(digest)
    for key, digests in by_checkpoint.items():
        distinct = set(digests.values())
        if len(distinct) > 1:
            for row in rows:
                identity = row["identity"]
                if identity["kind"] != "eval":
                    continue
                if (identity["seed"], absence.cell_value(identity["checkpoint_step"])) != key:
                    continue
                row["status"] = "fail"
                row["reasons"].append(
                    f"chains over seed={key[0]} step={key[1]} restored differing checkpoint "
                    f"hashes: {sorted(distinct)}"
                )


def require_unique_identities(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject two collected rows that claim the same identity."""

    seen: dict[tuple[Any, ...], str] = {}
    for row in rows:
        identity = row["identity"]
        key = (
            identity["row_id"],
            identity["plan_attempt_id"],
            absence.cell_value(identity["run_id"]),
        )
        if key in seen:
            raise CollectError(
                f"rows {seen[key]!r} and {identity['row_id']!r} share identity {key!r}"
            )
        seen[key] = str(identity["row_id"])


def summarize(rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]) -> dict[str, Any]:
    """Aggregate evaluation-row metrics, keeping coverage visible."""

    eval_rows = [row for row in rows if row["identity"]["kind"] == "eval"]
    summaries: dict[str, Any] = {}
    for key in keys:
        values = [absence.cell_value(row["metrics"].get(key, absence.cell(None))) for row in eval_rows]
        summaries[key] = absence.summarize_values(values).to_dict()
    return summaries


def collect(
    *,
    results_root: Path,
    plan_attempt_id: str,
    collect_attempt_id: str,
    gate_spec: Mapping[str, Any],
    gate_spec_source: str,
    extra_metric_keys: Sequence[str] = (),
    hash_checkpoints: bool = True,
) -> dict[str, Any]:
    """Collect one plan attempt into a durable table."""

    manifest = plan_stage.read_manifest(results_root, plan_attempt_id)
    rows = [
        collect_row(
            row,
            results_root=results_root,
            plan_attempt_id=plan_attempt_id,
            manifest=manifest,
            gate_spec=gate_spec,
            extra_metric_keys=extra_metric_keys,
            hash_checkpoints=hash_checkpoints,
        )
        for row in manifest["rows"]
    ]
    require_unique_identities(rows)
    cross_check_checkpoint_identity(rows)
    metric_keys = list(dict.fromkeys([*ENERGY_METRIC_KEYS, *GATE_METRIC_KEYS, *extra_metric_keys]))
    collected = {
        "schema_version": plan_stage.SCHEMA_VERSION,
        "study": str(manifest["study"]),
        "plan_attempt_id": str(plan_attempt_id),
        "plan_hash": str(manifest["plan_hash"]),
        "collect_attempt_id": str(collect_attempt_id),
        "gate_spec": dict(gate_spec),
        "gate_spec_declared": bool(gate_spec),
        "gate_spec_source": gate_spec_source,
        "checkpoint_hashing": bool(hash_checkpoints),
        "metric_keys": metric_keys,
        "n_rows": len(rows),
        "n_pass": sum(1 for row in rows if row["status"] == "pass"),
        "n_fail": sum(1 for row in rows if row["status"] == "fail"),
        "summaries": summarize(rows, keys=metric_keys),
        "rows": rows,
    }
    return collected


def write_collected(collected: Mapping[str, Any], *, results_root: Path) -> Path:
    """Write the collected table, its CSV, and the per-gate CSV."""

    attempt_id = str(collected["collect_attempt_id"])
    directory = layout.collect_attempt_dir(results_root, attempt_id)
    layout.write_json(directory / COLLECTED_FILENAME, dict(collected))
    _write_rows_csv(directory / ROWS_CSV, collected)
    _write_gates_csv(directory / GATES_CSV, collected)
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_COLLECT), attempt_id)
    return directory


def _write_rows_csv(path: Path, collected: Mapping[str, Any]) -> None:
    metric_keys = list(collected["metric_keys"])
    fieldnames = [
        "row_id",
        "kind",
        "status",
        "seed",
        "checkpoint_step",
        "chain",
        "run_id",
        "job_id",
        "partition",
        "requested_stratum",
        "requested_constraint",
        "delivered_device",
        "config_sha256",
        "checkpoint_sha256",
        "run_status",
        *metric_keys,
        "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in collected["rows"]:
            identity = row["identity"]
            record = {
                "row_id": identity["row_id"],
                "kind": identity["kind"],
                "status": row["status"],
                "seed": identity["seed"],
                "checkpoint_step": _render_cell(identity["checkpoint_step"]),
                "chain": _render_cell(identity["chain"]),
                "run_id": _render_cell(identity["run_id"]),
                "job_id": _render_cell(identity["job_id"]),
                "partition": identity["partition"],
                "requested_stratum": identity["requested_stratum"],
                "requested_constraint": identity["requested_constraint"],
                "delivered_device": _render_cell(identity["delivered_device"]),
                "config_sha256": _render_cell(identity["config_sha256"]),
                "checkpoint_sha256": _render_cell(identity["checkpoint_sha256"]),
                "run_status": _render_cell(row["run_status"]),
                "reasons": "; ".join(row["reasons"]) or "none",
            }
            for key in metric_keys:
                record[key] = _render_cell(row["metrics"].get(key, absence.cell(None)))
            writer.writerow(record)


def _write_gates_csv(path: Path, collected: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "gate", "status", "value", "threshold", "reason"],
        )
        writer.writeheader()
        for row in collected["rows"]:
            for gate_row in row["gates"]:
                writer.writerow(
                    {
                        "row_id": row["identity"]["row_id"],
                        "gate": gate_row["name"],
                        "status": gate_row["status"],
                        "value": _render_cell(gate_row["value"]),
                        "threshold": _render_cell(gate_row["threshold"]),
                        "reason": gate_row["reason"],
                    }
                )


def _render_cell(cell: Any) -> str:
    return absence.render(absence.cell_value(cell))


def _gate_counts(gate_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"pass": 0, "fail": 0, "absent": 0}
    for gate_row in gate_rows:
        counts[str(gate_row["status"])] = counts.get(str(gate_row["status"]), 0) + 1
    return counts


def _read_json_or_absent(path: Path, reasons: list[str]) -> Any:
    if not path.is_file():
        reasons.append(f"missing artifact: {path.name}")
        return absence.ABSENT
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        reasons.append(f"unreadable artifact {path.name}: {exc}")
        return absence.ABSENT


def _json_field(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return absence.present_or_absent(payload.get(key))
    return absence.ABSENT


def _receipt_field(receipt: Any, key: str) -> Any:
    return _json_field(receipt, key)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse collect command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, help="Durable study results root.")
    parser.add_argument("--plan-attempt-id", default=None, help="Plan attempt (defaults to latest).")
    parser.add_argument("--collect-attempt-id", default=None, help="Explicit collect attempt id.")
    parser.add_argument(
        "--gate-spec",
        default=None,
        help="Tolerance spec YAML; defaults to the spec recorded in the plan manifest.",
    )
    parser.add_argument(
        "--metric-key",
        action="append",
        default=[],
        help="Additional metric key to retain per row; repeatable.",
    )
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Skip checkpoint hashing; the hash then renders as absent, never as matching.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Collect one plan attempt."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    plan_attempt_id = layout.resolve_attempt_id(results_root, layout.STAGE_PLAN, args.plan_attempt_id)
    manifest = plan_stage.read_manifest(results_root, plan_attempt_id)
    gate_spec = load_gate_spec(args.gate_spec, manifest=manifest)
    collected = collect(
        results_root=results_root,
        plan_attempt_id=plan_attempt_id,
        collect_attempt_id=args.collect_attempt_id or plan_stage.now_attempt_id(),
        gate_spec=gate_spec,
        gate_spec_source=str(args.gate_spec) if args.gate_spec else "plan_manifest",
        extra_metric_keys=args.metric_key,
        hash_checkpoints=not args.skip_checkpoint_hash,
    )
    directory = write_collected(collected, results_root=results_root)
    print(
        f"[he-v1] collected {collected['n_rows']} rows "
        f"({collected['n_pass']} pass, {collected['n_fail']} fail) into {directory}"
    )
    if not collected["gate_spec_declared"]:
        print(
            "[he-v1] no tolerances declared: value gates report 'absent' with observed "
            "values retained (thresholds are predeclared in H-F1)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
