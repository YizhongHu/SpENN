"""Plan and execute one-shot NN-QMC rows on ALCF Polaris.

The module owns the Polaris-specific contract.  A manifest describes logical
rows; :func:`size_production_request` turns those rows into a legal ``prod``
request; and the PBS template launches one ordinary Python process per GPU.
The worker deliberately knows nothing about any particular baseline package,
so a manifest may name codes which are not installed in the selected runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import yaml


SCHEMA = "nnqmc-polaris-manifest/v1"
PROFILE_SCHEMA = "nnqmc-polaris-runtime-profile/v1"
GPU_PER_NODE = 4


class PolarisSubmissionError(ValueError):
    """A manifest or scheduler request cannot satisfy the Polaris contract."""


def gpu_for_local_rank(local_rank: int) -> int:
    """Return Polaris's reversed local GPU id for an MPI rank."""

    if isinstance(local_rank, bool) or not isinstance(local_rank, int) or not 0 <= local_rank < GPU_PER_NODE:
        raise PolarisSubmissionError("device-binding constraint: local rank must be in 0..3")
    return GPU_PER_NODE - 1 - local_rank


def row_gpu_slot(row_index: int) -> tuple[int, int]:
    """Return ``(node_index, local_gpu_id)`` for a zero-based row rank."""

    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise PolarisSubmissionError("row constraint: rank must be a non-negative integer")
    return row_index // GPU_PER_NODE, gpu_for_local_rank(row_index % GPU_PER_NODE)


@dataclass(frozen=True)
class Destination:
    """One routed ``prod`` destination and its jointly valid limits."""

    name: str
    min_nodes: int
    max_nodes: int
    max_walltime_seconds: int


DESTINATIONS = (
    Destination("small", 10, 24, 3 * 60 * 60),
    Destination("medium", 25, 99, 6 * 60 * 60),
    Destination("large", 100, 496, 24 * 60 * 60),
)


@dataclass(frozen=True)
class SubmissionRequest:
    """A fully resolved PBS request, including the routed destination."""

    queue: str
    destination: str
    nodes: int
    walltime_seconds: int
    row_count: int

    @property
    def walltime(self) -> str:
        """Return PBS ``HH:MM:SS`` walltime text."""

        hours, remainder = divmod(self.walltime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-friendly request data."""

        return {**asdict(self), "walltime": self.walltime}


def parse_walltime(value: str | int | float) -> int:
    """Parse a positive PBS walltime string or a number of seconds."""

    if isinstance(value, bool):
        raise PolarisSubmissionError("walltime constraint: boolean is not a duration")
    if isinstance(value, (int, float)):
        seconds = int(value)
        if seconds != value:
            raise PolarisSubmissionError("walltime constraint: duration must be integral seconds")
    else:
        text = str(value).strip()
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise PolarisSubmissionError(
                f"walltime constraint: {value!r} is not HH:MM:SS or MM:SS"
            )
        try:
            numbers = [int(part) for part in parts]
        except ValueError as exc:
            raise PolarisSubmissionError(
                f"walltime constraint: {value!r} is not numeric"
            ) from exc
        if any(part < 0 for part in numbers):
            raise PolarisSubmissionError("walltime constraint: duration cannot be negative")
        if len(numbers) == 2:
            minutes, seconds = numbers
            hours = 0
        else:
            hours, minutes, seconds = numbers
        if minutes >= 60 or seconds >= 60:
            raise PolarisSubmissionError(
                f"walltime constraint: {value!r} has invalid minute/second fields"
            )
        seconds = hours * 3600 + minutes * 60 + seconds
    if seconds <= 0:
        raise PolarisSubmissionError("walltime constraint: duration must be positive")
    return seconds


def size_production_request(row_count: int, per_row_walltime: str | int | float) -> SubmissionRequest:
    """Choose the smallest legal ``prod`` destination for independent rows.

    Polaris ``prod`` is a router, not a 10--496-node execution queue.  The
    node and walltime limits below are a joint table.  A row requiring seven
    hours therefore forces a 100-node request even when there is only one row.

    Parameters
    ----------
    row_count : int
        Number of independent one-GPU rows in the wave.
    per_row_walltime : str, int, or float
        Required walltime for every row, as PBS text or seconds.

    Returns
    -------
    SubmissionRequest
        A request that one and only one ``prod`` destination accepts.

    Raises
    ------
    PolarisSubmissionError
        If row count, walltime, or the joint destination shape is impossible.
    """

    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise PolarisSubmissionError("row-count constraint: at least one row is required")
    seconds = parse_walltime(per_row_walltime)
    required_gpu_nodes = math.ceil(row_count / GPU_PER_NODE)
    if required_gpu_nodes > DESTINATIONS[-1].max_nodes:
        raise PolarisSubmissionError(
            "node-capacity constraint: "
            f"{row_count} rows need {required_gpu_nodes} nodes at four GPUs/node, "
            f"but prod ends at {DESTINATIONS[-1].max_nodes} nodes"
        )

    for destination in DESTINATIONS:
        nodes = max(destination.min_nodes, required_gpu_nodes)
        if nodes <= destination.max_nodes and seconds <= destination.max_walltime_seconds:
            return SubmissionRequest("prod", destination.name, nodes, seconds, row_count)

    if seconds > DESTINATIONS[-1].max_walltime_seconds:
        raise PolarisSubmissionError(
            "walltime constraint: "
            f"each row needs {seconds}s, exceeding prod/large's 86400s cap; "
            "rows are one-shot and may not rely on restart"
        )
    # This is the important joint-shape refusal.  It names both axes rather
    # than exposing PBS's opaque "all possible destinations" error.
    destination = next(
        (item for item in DESTINATIONS if seconds <= item.max_walltime_seconds),
        DESTINATIONS[-1],
    )
    nodes = max(destination.min_nodes, required_gpu_nodes)
    raise PolarisSubmissionError(
        "joint-routing constraint: "
        f"{nodes} nodes with {seconds}s walltime fit no prod destination; "
        f"{destination.name} allows {destination.min_nodes}-{destination.max_nodes} "
        f"nodes and at most {destination.max_walltime_seconds}s"
    )


def validate_production_request(nodes: int, walltime: str | int | float) -> str:
    """Validate a concrete ``prod`` shape and return its destination name.

    This check is intentionally public so callers validating an operator's
    requested ``select`` cannot accidentally reproduce PBS's opaque rejection.
    """

    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes <= 0:
        raise PolarisSubmissionError("node-count constraint: nodes must be a positive integer")
    seconds = parse_walltime(walltime)
    if nodes < DESTINATIONS[0].min_nodes or nodes > DESTINATIONS[-1].max_nodes:
        raise PolarisSubmissionError(
            "node-count constraint: prod accepts 10-496 nodes, got " + str(nodes)
        )
    for destination in DESTINATIONS:
        if destination.min_nodes <= nodes <= destination.max_nodes:
            if seconds <= destination.max_walltime_seconds:
                return destination.name
            raise PolarisSubmissionError(
                "joint-routing constraint: "
                f"prod/{destination.name} permits {destination.min_nodes}-{destination.max_nodes} "
                f"nodes but at most {destination.max_walltime_seconds}s, not {seconds}s"
            )
    raise PolarisSubmissionError(
        f"joint-routing constraint: {nodes} nodes with {seconds}s matches no prod destination"
    )


def size_debug_request(row_count: int, per_row_walltime: str | int | float) -> SubmissionRequest:
    """Size a small receipt-backed ``debug`` validation request."""

    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise PolarisSubmissionError("row-count constraint: at least one row is required")
    seconds = parse_walltime(per_row_walltime)
    nodes = math.ceil(row_count / GPU_PER_NODE)
    if nodes > 2:
        raise PolarisSubmissionError("debug node-capacity constraint: debug accepts at most 2 nodes")
    if seconds > 60 * 60:
        raise PolarisSubmissionError("debug walltime constraint: debug accepts at most 3600s")
    return SubmissionRequest("debug", "debug", max(1, nodes), seconds, row_count)


def size_request(queue: str, row_count: int, per_row_walltime: str | int | float) -> SubmissionRequest:
    """Select a request for the production path or its short validation path."""

    if queue == "prod":
        return size_production_request(row_count, per_row_walltime)
    if queue == "debug":
        return size_debug_request(row_count, per_row_walltime)
    raise PolarisSubmissionError(
        f"queue constraint: {queue!r} is unsupported; use prod or receipt-backed debug"
    )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolarisSubmissionError(f"manifest constraint: {label} must be a mapping")
    return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a row manifest without requiring installed codes."""

    manifest_path = Path(path)
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PolarisSubmissionError(f"manifest constraint: cannot read {manifest_path}") from exc
    root = _require_mapping(payload, "root")
    expected_keys = {"schema", "facility", "runtime", "rows"}
    if set(root) != expected_keys:
        raise PolarisSubmissionError(
            f"manifest constraint: root keys must be {sorted(expected_keys)}, got {sorted(root)}"
        )
    if root["schema"] != SCHEMA or root["facility"] != "polaris":
        raise PolarisSubmissionError("manifest constraint: schema/facility is not Polaris v1")
    runtime = _require_mapping(root["runtime"], "runtime")
    runtime_keys = {"ferminet_root", "ferminet_branch", "ferminet_commit"}
    if set(runtime) != runtime_keys:
        raise PolarisSubmissionError(
            f"manifest constraint: runtime keys must be {sorted(runtime_keys)}"
        )
    for key in runtime_keys:
        if not isinstance(runtime[key], str) or not runtime[key].strip():
            raise PolarisSubmissionError(f"manifest constraint: runtime.{key} must be non-empty")
    rows = root["rows"]
    if not isinstance(rows, list) or not rows:
        raise PolarisSubmissionError("manifest constraint: rows must be a non-empty list")

    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    row_keys = {"id", "code", "ansatz", "system", "seed", "steps", "command"}
    for index, raw in enumerate(rows):
        row = dict(_require_mapping(raw, f"rows[{index}]"))
        if set(row) != row_keys:
            raise PolarisSubmissionError(
                f"manifest constraint: rows[{index}] keys must be {sorted(row_keys)}"
            )
        row_id = row["id"]
        if not isinstance(row_id, str) or not row_id.strip():
            raise PolarisSubmissionError(f"manifest constraint: rows[{index}].id must be non-empty")
        if row_id in seen:
            raise PolarisSubmissionError(f"manifest constraint: duplicate row id {row_id!r}")
        seen.add(row_id)
        for key in ("code", "ansatz", "system"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise PolarisSubmissionError(f"manifest constraint: rows[{index}].{key} must be non-empty")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise PolarisSubmissionError(f"manifest constraint: rows[{index}].seed must be an integer")
        if isinstance(row["steps"], bool) or not isinstance(row["steps"], int) or row["steps"] <= 0:
            raise PolarisSubmissionError(f"manifest constraint: rows[{index}].steps must be positive")
        command = row["command"]
        if not isinstance(command, list) or not command or not all(isinstance(token, str) for token in command):
            raise PolarisSubmissionError(
                f"manifest constraint: rows[{index}].command must be a non-empty string list"
            )
        lowered = " ".join(command).lower()
        if any(token in lowered for token in ("mpi4py", "torchrun", "ddp", "mpirun", "mpiexec")):
            raise PolarisSubmissionError(
                f"launcher constraint: rows[{index}] embeds MPI/DDP; mpiexec belongs only to the PBS template"
            )
        normalized_rows.append(row)
    return {**dict(root), "runtime": dict(runtime), "rows": normalized_rows}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a marker atomically; an existing marker means a restart."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise PolarisSubmissionError(
            f"restart detected: one-shot marker already exists at {path}"
        ) from exc


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PolarisSubmissionError(f"preflight git check failed for {root}: {exc}") from exc
    return result.stdout.strip()


def _probe_jax() -> dict[str, Any]:
    """Import JAX only after the caller has established GPU visibility."""

    try:
        import jax  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - the message is environment-specific
        raise PolarisSubmissionError(f"preflight JAX check failed: {exc}") from exc
    devices = list(jax.devices())
    if not devices:
        raise PolarisSubmissionError("preflight device constraint: JAX reported no devices")
    platforms = {str(device.platform) for device in devices}
    if platforms != {"gpu"}:
        raise PolarisSubmissionError(
            f"preflight device constraint: expected GPU-only JAX devices, got {sorted(platforms)}"
        )
    kinds = sorted({str(device.device_kind) for device in devices})
    return {
        "jax_version": str(jax.__version__),
        "device_count": len(devices),
        "device_kind": kinds[0] if len(kinds) == 1 else kinds,
        "device_kinds": kinds,
    }


def run_preflight(
    manifest: Mapping[str, Any],
    *,
    results_root: str | Path,
    interpreter: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the fatal, once-per-job checks before any row launches."""

    env = os.environ if environ is None else environ
    root = Path(results_root)
    _write_once(root / "preflight.started.json", {"started_at": _utc_now()})
    python = str(interpreter or sys.executable)
    if Path(python).resolve() != Path(sys.executable).resolve() and interpreter is None:
        raise PolarisSubmissionError("preflight interpreter constraint: unresolved interpreter changed")
    if sys.version_info < (3, 10):
        raise PolarisSubmissionError(f"preflight interpreter constraint: Python {sys.version} is too old")

    runtime = _require_mapping(manifest["runtime"], "runtime")
    ferminet_root = Path(str(runtime["ferminet_root"]))
    if not ferminet_root.is_dir():
        raise PolarisSubmissionError(f"preflight ferminet tree constraint: missing {ferminet_root}")
    branch = _git(ferminet_root, "branch", "--show-current")
    expected_branch = str(runtime["ferminet_branch"])
    if branch != expected_branch:
        raise PolarisSubmissionError(
            f"preflight branch constraint: ferminet is on {branch!r}, expected {expected_branch!r}"
        )
    head = _git(ferminet_root, "rev-parse", "HEAD")
    expected_commit = str(runtime["ferminet_commit"])
    if not head.startswith(expected_commit):
        raise PolarisSubmissionError(
            f"preflight commit constraint: ferminet HEAD {head} does not match {expected_commit}"
        )
    status = _git(ferminet_root, "status", "--porcelain")
    if status:
        raise PolarisSubmissionError(
            f"preflight clean-tree constraint: ferminet tree is dirty ({status.splitlines()[0]!r})"
        )

    # The template exports this before invoking the worker.  Refuse a job that
    # forgot the setting, rather than silently accepting JAX's 75% prealloc.
    if env.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise PolarisSubmissionError(
            "preflight memory constraint: XLA_PYTHON_CLIENT_PREALLOCATE must be false"
        )
    jax_data = _probe_jax()
    record = {
        "schema": "nnqmc-polaris-preflight/v1",
        "interpreter": str(Path(python).resolve()),
        "python_version": sys.version,
        "ferminet_branch": branch,
        "ferminet_commit": head,
        "ferminet_clean": True,
        **jax_data,
        "completed_at": _utc_now(),
    }
    (root / "preflight.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _bound_device() -> dict[str, Any]:
    """Record the visible device after the launcher has set its binding."""

    data = _probe_jax()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible not in {"0", "1", "2", "3"}:
        raise PolarisSubmissionError(
            "device-binding constraint: CUDA_VISIBLE_DEVICES must be one local Polaris GPU id"
        )
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        uuids = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PolarisSubmissionError(f"device-binding constraint: nvidia-smi failed: {exc}") from exc
    if len(uuids) != 1:
        raise PolarisSubmissionError(
            f"device-binding constraint: expected one visible GPU UUID, got {len(uuids)}"
        )
    return {
        "cuda_visible_devices": visible,
        "device_uuid": uuids[0],
        "device_kinds": data["device_kinds"],
        "jax_version": data["jax_version"],
    }


def run_row(manifest: Mapping[str, Any], row_index: int, *, results_root: str | Path) -> int:
    """Run one row and always leave a terminal record when it returns."""

    rows = manifest["rows"]
    if not isinstance(row_index, int) or row_index < 0 or row_index >= len(rows):
        raise PolarisSubmissionError(f"row constraint: index {row_index} is outside manifest")
    row = dict(rows[row_index])
    row_dir = Path(results_root) / "rows" / row["id"]
    _write_once(
        row_dir / "started.json",
        {
            "schema": "nnqmc-polaris-row-start/v1",
            "row_id": row["id"],
            "code": row["code"],
            "ansatz": row["ansatz"],
            "system": row["system"],
            "seed": row["seed"],
            "steps": row["steps"],
            "started_at": _utc_now(),
        },
    )
    result: dict[str, Any] = {
        "schema": "nnqmc-polaris-row-result/v1",
        "row_id": row["id"],
        "code": row["code"],
        "ansatz": row["ansatz"],
        "system": row["system"],
        "seed": row["seed"],
        "steps": row["steps"],
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }
    try:
        preflight_path = Path(results_root) / "preflight.json"
        if not preflight_path.is_file():
            raise PolarisSubmissionError("ordering constraint: preflight.json is missing")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        result["ferminet_branch"] = preflight["ferminet_branch"]
        result["ferminet_commit"] = preflight["ferminet_commit"]
        if str(result["xla_preallocate"]).lower() != "false":
            raise PolarisSubmissionError(
                "memory constraint: XLA_PYTHON_CLIENT_PREALLOCATE must be false for every row"
            )
        result["device"] = _bound_device()
        command = [sys.executable if token == "{python}" else token for token in row["command"]]
        result["command"] = command
        result["command_text"] = shlex.join(command)
        completed = subprocess.run(command, check=False, env=dict(os.environ))
        result["exit_code"] = completed.returncode
        result["status"] = "complete" if completed.returncode == 0 else "failed"
        return_code = completed.returncode
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return_code = 1
    result["finished_at"] = _utc_now()
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (row_dir / "terminal.json").write_text(
        json.dumps({"schema": "nnqmc-polaris-terminal/v1", "row_id": row["id"], "status": result["status"], "exit_code": return_code, "finished_at": result["finished_at"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return return_code


def render_template(template: str | Path, request: SubmissionRequest, output: str | Path) -> Path:
    """Render scheduler shape fields into the tracked PBS template."""

    template_path = Path(template)
    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "@QUEUE@": request.queue,
        "@DESTINATION@": request.destination,
        "@NODES@": str(request.nodes),
        "@WALLTIME@": request.walltime,
        "@ROW_COUNT@": str(request.row_count),
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    if "@" in text:
        raise PolarisSubmissionError("template constraint: unresolved scheduler marker")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    output_path.chmod(0o750)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    size = subparsers.add_parser("size")
    size.add_argument("--rows", type=int, required=True)
    size.add_argument("--walltime", required=True)
    manifest = subparsers.add_parser("validate-manifest")
    manifest.add_argument("path", type=Path)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--results-root", type=Path, required=True)
    row = subparsers.add_parser("row")
    row.add_argument("--manifest", type=Path, required=True)
    row.add_argument("--row-index", type=int, required=True)
    row.add_argument("--results-root", type=Path, required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--walltime", required=True)
    plan.add_argument("--results-root", type=Path, required=True)
    plan.add_argument("--template", type=Path, default=Path(__file__).with_name("templates") / "polaris_production.pbs")
    plan.add_argument("--queue", choices=("prod", "debug"), default="prod")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one of the small planner/worker entrypoints."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "size":
            print(json.dumps(size_production_request(args.rows, args.walltime).to_dict(), indent=2))
        elif args.command == "validate-manifest":
            manifest = load_manifest(args.path)
            print(json.dumps({"rows": len(manifest["rows"]), "schema": manifest["schema"]}))
        elif args.command == "preflight":
            print(json.dumps(run_preflight(load_manifest(args.manifest), results_root=args.results_root), indent=2))
        elif args.command == "row":
            return run_row(load_manifest(args.manifest), args.row_index, results_root=args.results_root)
        elif args.command == "plan":
            manifest = load_manifest(args.manifest)
            request = size_request(args.queue, len(manifest["rows"]), args.walltime)
            output = render_template(args.template, request, args.results_root / "scheduler" / "polaris.pbs")
            (args.results_root / "scheduler" / "request.json").write_text(
                json.dumps({"manifest": str(args.manifest.resolve()), **request.to_dict()}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"script": str(output), **request.to_dict()}, indent=2))
    except PolarisSubmissionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
