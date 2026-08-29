"""Verify a DeepQMC-on-Polaris environment and the seed override it depends on.

This module exists because two things in the Polaris DeepQMC port fail
*silently*, and both would otherwise be discovered only after a batch of runs
had been published.

**An import test is not a GPU test.** Importing ``jax`` and ``deepqmc``
succeeds on a Polaris login node, where the GPUs are not user-usable at all.
The ``env`` subcommand therefore asserts a visible GPU device rather than a
successful import, and it is meant to be run inside a PBS allocation. On a
login node it will correctly fail; that is the point.

**A rejected ``task.seed`` override looks exactly like a successful one.**
Hydra accepts an override on the command line and DeepQMC starts training
either way. If the override does not reach the config, every "seed spread" row
silently runs the same seed, produces near-identical energies, and reads as a
legitimate result. The ``seed`` subcommand reads the value back out of the
run's own ``training/.hydra/config.yaml`` -- the file the run actually
resolved -- rather than trusting the command line that requested it.

That check has one trap worth stating, because the obvious version of it
returns a blank and reads as "no seed key". ``task:`` is the first line of that
file and ``seed`` nests roughly fifty lines below it, so::

    grep -A2 task: config.yaml | grep seed     # prints nothing; the key IS there

This module parses the YAML instead, and reports the line number and the raw
line so a reader can confirm it against the file by eye.

Examples
--------
::

    # inside a PBS allocation, after XLA_PYTHON_CLIENT_PREALLOCATE=false
    python -m experiments.baselines.check_polaris_deepqmc_env env \\
        --expect-prefix /home/rhu/.venvs/deepqmc-jax083-edf373e7 \\
        --expect-jax 0.8.3

    python -m experiments.baselines.check_polaris_deepqmc_env seed \\
        --run-dir /eagle/HetRxnEnergy/rhu/runs/<run> --expect-seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

# Relative to the run directory Hydra was pointed at with ``hydra.run.dir``.
HYDRA_CONFIG_RELPATH = Path("training") / ".hydra" / "config.yaml"


class EnvCheckError(RuntimeError):
    """Raised when a checked property of the environment or run is not as required.

    Carries the partial report alongside the message. A failed required check
    does not invalidate the evidence gathered before it -- knowing WHICH
    interpreter and WHICH commit failed is most of the diagnostic value -- so
    that evidence travels with the exception instead of being discarded.
    """

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report


def _git_commit(source_root: Path) -> dict[str, str]:
    """Resolve the commit and cleanliness of a source checkout.

    Parameters
    ----------
    source_root : Path
        Directory containing the ``.git`` of the DeepQMC checkout.

    Returns
    -------
    dict of str to str
        Keys ``commit``, ``subject`` and ``dirty``. ``dirty`` is the porcelain
        status of *tracked* files only, so an untracked build artefact does not
        masquerade as a modified source tree.
    """
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source_root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "subject": run("log", "--format=%s", "-1"),
        # Empty string means a clean tracked tree.
        "dirty": run("status", "--porcelain=v1", "--untracked-files=no"),
    }


def backend_platform_version(jax_module: Any) -> dict[str, Any]:
    """Resolve the backend's platform version without ever raising.

    The accessor for this moved between JAX releases, and reaching for the wrong
    one is fatal in a way that is out of all proportion to the value: on jax
    0.8.3 ``jax.extend`` raises ``AttributeError`` from a deprecation shim, which
    took down an entire GPU environment check on Polaris (PBS 7571666) and cost
    the interpreter, device and provenance evidence along with it. Optional
    evidence must never be able to do that, so every route is attempted in turn
    and the failures are reported as data.

    Parameters
    ----------
    jax_module : module
        The imported ``jax`` module.

    Returns
    -------
    dict
        ``platform_version`` (None if no route worked), ``via`` naming the route
        that succeeded, and ``attempts`` listing what each failed route said.
        The attempts are kept even on success: knowing which API this JAX build
        answers to is itself useful when comparing two facilities.
    """
    def _via_device_client() -> Any:
        # Works across the range this project uses, because a device always
        # carries the client that produced it.
        return jax_module.devices()[0].client.platform_version

    def _via_extend() -> Any:
        import jax.extend  # noqa: PLC0415 -- probing availability deliberately

        return jax.extend.backend.get_backend().platform_version

    def _via_xla_bridge() -> Any:
        from jax.lib import xla_bridge  # noqa: PLC0415

        return xla_bridge.get_backend().platform_version

    attempts: list[str] = []
    routes = (
        ("device.client", _via_device_client),
        ("jax.extend.backend", _via_extend),
        ("jax.lib.xla_bridge", _via_xla_bridge),
    )
    for name, route in routes:
        try:
            value = route()
        except Exception as exc:  # noqa: BLE001 -- any failure is just a dead route
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        return {"platform_version": value, "via": name, "attempts": attempts}
    return {"platform_version": None, "via": None, "attempts": attempts}


def loaded_cuda_libraries() -> dict[str, str]:
    """Report the CUDA shared objects this process has actually mapped.

    Pinning ``jaxlib`` to an equal version on two facilities does NOT establish
    that both load an equal CUDA runtime: the ``cuda12`` wheels vendor their own
    runtime, and which ``.so`` wins at load time is a property of the process,
    not of the wheel version. Since that is the most likely source of a small
    unexplained cross-facility energy difference, it is recorded rather than
    inferred.

    Returns
    -------
    dict of str to str
        Library soname to the resolved path it was loaded from. Empty on
        platforms without ``/proc/self/maps`` (macOS, for instance), which is
        reported as an empty mapping rather than raised, because this is
        evidence-gathering and not a checked assertion.
    """
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return {}
    return parse_proc_maps(maps.read_text())


#: Shared-object prefixes worth recording from a process memory map.
CUDA_SONAME_PREFIXES = (
    "libcudart",
    "libcublas",
    "libcudnn",
    "libcufft",
    "libcusolver",
    "libnvrtc",
)


def parse_proc_maps(text: str) -> dict[str, str]:
    """Extract CUDA shared-object mappings from ``/proc/self/maps`` content.

    Split out as a pure function so it is testable on a machine without
    ``/proc``. It previously lived inline and survived a mutation
    (``startswith`` to ``endswith``) undetected, because the only test that could
    run off-Linux asserted the return TYPE and never the parsing.

    Parameters
    ----------
    text : str
        Contents of a ``/proc/<pid>/maps`` file.

    Returns
    -------
    dict of str to str
        Library soname to the resolved path it was first mapped from.
    """
    interesting = CUDA_SONAME_PREFIXES
    found: dict[str, str] = {}
    for line in text.splitlines():
        # /proc/self/maps is `address perms offset dev inode pathname`, and the
        # PATHNAME MAY CONTAIN SPACES. Taking the last whitespace-separated field
        # silently dropped any library under a path like "/opt/cuda libs/",
        # reporting {} rather than an error -- an absent mapping that reads as
        # "nothing loaded". Split on the fixed five leading fields instead.
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if not path.startswith("/"):
            continue
        name = path.rsplit("/", 1)[-1]
        if name.startswith(interesting):
            found.setdefault(name, path)
    return dict(sorted(found.items()))


def gpu_memory_high_water() -> list[dict[str, Any]]:
    """Report peak device memory per visible GPU, in bytes and MiB.

    Only meaningful when ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is set before
    JAX initialises. With preallocation on, JAX takes ~75% of the card up front
    and every configuration reports the same number, so the reading measures the
    allocator's policy rather than the workload.

    Returns
    -------
    list of dict
        One entry per device. ``peak_bytes_in_use`` is absent on devices whose
        backend does not expose memory statistics, and is reported as ``None``
        rather than omitted, so a missing reading cannot be mistaken for zero.
    """
    import jax

    stats: list[dict[str, Any]] = []
    for device in jax.devices():
        entry: dict[str, Any] = {"id": device.id, "kind": device.device_kind}
        try:
            memory = device.memory_stats() or {}
        except Exception as exc:  # backend without memory stats
            entry["error"] = str(exc)
            memory = {}
        peak = memory.get("peak_bytes_in_use")
        entry["peak_bytes_in_use"] = peak
        entry["peak_mib"] = None if peak is None else round(peak / 1024 / 1024, 1)
        entry["preallocate_env"] = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
        # THIS PROCESS ONLY. Device memory statistics are per-process, so a
        # checker launched after a training run reports the checker's own peak,
        # not the run's. On Polaris PBS 7571675 that produced peak_mib 0.0 with
        # preallocate correctly set to 'false' and no warning attached, which
        # reads like a real measurement of a run that used no memory. It is not a
        # measurement of that run at all.
        entry["scope"] = "this process only; not any earlier process on this device"
        if peak == 0:
            entry["warning"] = (
                "peak_bytes_in_use is 0: this process did no device work, so this "
                "reading says NOTHING about a training run in another process. "
                "Measure the training process itself (see gpu-mem-samples.csv)"
            )
        elif entry["preallocate_env"] != "false":
            # Stated inline so a reader of the job log cannot take the number at
            # face value without also seeing why it may be meaningless.
            entry["warning"] = (
                "XLA_PYTHON_CLIENT_PREALLOCATE is not 'false'; this reading likely "
                "reflects JAX preallocation (~75% of the card), not the workload"
            )
        stats.append(entry)
    return stats


def check_env(
    expect_prefix: str | None,
    expect_jax: str | None,
    expect_commit: str | None,
    source_root: Path | None,
    require_gpu: bool,
) -> dict[str, Any]:
    """Report the interpreter, JAX build, visible devices and DeepQMC provenance.

    Parameters
    ----------
    expect_prefix : str or None
        If given, ``sys.prefix`` must equal this. Guards against the job having
        picked up a different venv than the one it names -- the failure mode
        where a run is attributed to the wrong environment.
    expect_jax : str or None
        If given, ``jax.__version__`` must equal this.
    expect_commit : str or None
        If given, the DeepQMC checkout must be at this commit. A prefix match is
        accepted so an abbreviated SHA can be passed.
    source_root : Path or None
        DeepQMC checkout to interrogate. Defaults to the directory containing
        the imported ``deepqmc`` package, which is what an editable install
        points at.
    require_gpu : bool
        When true, a GPU-platform device must be visible. Leave true for any
        check inside a PBS allocation; a login node has no usable GPU and
        failing there is correct behaviour, not a bug.

    Returns
    -------
    dict
        The collected facts, suitable for printing as JSON into a job log.

    Raises
    ------
    EnvCheckError
        If any requested expectation is not met.
    """
    import jax  # imported lazily so --help works outside the venv
    import deepqmc

    devices = jax.devices()
    report: dict[str, Any] = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        # base_prefix identifies the interpreter the venv was built on top of.
        # On Polaris this should be the absolute facility conda interpreter,
        # which is what makes the vendored-CUDA wheel stack survive facility
        # CUDA changes.
        "base_prefix": sys.base_prefix,
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "device_count": len(devices),
        "device_kinds": sorted({d.device_kind for d in devices}),
        "platforms": sorted({d.platform for d in devices}),
        "deepqmc_file": deepqmc.__file__,
    }

    # OPTIONAL EVIDENCE, collected defensively and deliberately AFTER the report
    # dict exists. None of it may abort the run: these fields are useful context,
    # while the interpreter, jax version, device kind and DeepQMC commit above are
    # the evidence the acceptance criteria actually require. Letting a nice-to-have
    # field take down a required one is how PBS 7571666 produced a 0-byte
    # env-check.json.
    for key, collect in (
        # The backend's own statement of the CUDA runtime and driver it talks to
        # -- the artefact, rather than the wheel version it was inferred from.
        ("backend_platform_version", lambda: backend_platform_version(jax)),
        ("loaded_cuda_libraries", loaded_cuda_libraries),
        ("gpu_memory", gpu_memory_high_water),
    ):
        try:
            report[key] = collect()
        except Exception as exc:  # noqa: BLE001 -- context must never be fatal
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}

    root = source_root
    if root is None:
        # An editable install leaves __file__ inside the checkout, so walking up
        # from the package directory finds the source root. A NON-editable or
        # unusually shallow install may have fewer than three parents, which
        # previously raised IndexError BEFORE the provenance try block -- so the
        # function failed to produce the report it promises, for a reason that
        # has nothing to do with the environment being wrong.
        parents = Path(deepqmc.__file__).resolve().parents
        root = parents[2] if len(parents) > 2 else None
    if root is None:
        report["deepqmc_source"] = {
            "root": None,
            "error": (
                f"cannot derive a source root from {deepqmc.__file__!r}: fewer than "
                "three parent directories. Pass --source-root explicitly."
            ),
        }
        root = Path(deepqmc.__file__)  # keep the commit assertion below meaningful
    try:
        report["deepqmc_source"] = {"root": str(root), **_git_commit(root)}
    except (subprocess.CalledProcessError, OSError) as exc:
        # Recorded rather than raised: a missing checkout is only fatal when the
        # caller actually asked for a commit assertion, handled below.
        report["deepqmc_source"] = {"root": str(root), "error": str(exc)}

    failures: list[str] = []
    if expect_prefix is not None and sys.prefix != expect_prefix:
        failures.append(f"sys.prefix {sys.prefix!r} != expected {expect_prefix!r}")
    if expect_jax is not None and jax.__version__ != expect_jax:
        failures.append(f"jax {jax.__version__!r} != expected {expect_jax!r}")
    if require_gpu and not any(d.platform == "gpu" for d in devices):
        failures.append(f"no GPU device visible to JAX; devices={devices!r}")
    if expect_commit is not None:
        actual = report["deepqmc_source"].get("commit")
        if actual is None:
            failures.append(
                f"could not read DeepQMC commit: {report['deepqmc_source'].get('error')}"
            )
        elif not actual.startswith(expect_commit):
            failures.append(f"DeepQMC commit {actual!r} != expected {expect_commit!r}")

    report["ok"] = not failures
    report["failures"] = failures
    if failures:
        raise EnvCheckError("; ".join(failures), report=report)
    return report


def _is_key(text: str, key: str) -> bool:
    """True when ``text`` opens the mapping key ``key``, in any YAML spelling.

    YAML permits ``key:``, ``key :``, ``"key":`` and ``'key':`` for the same
    mapping key. A scanner matching only the first shape rejects valid files.
    """
    stripped = text.lstrip()
    for opener in (key, f'"{key}"', f"'{key}'"):
        if stripped.startswith(opener):
            rest = stripped[len(opener) :].lstrip()
            if rest.startswith(":"):
                return True
    return False


def _is_top_level_key(line: str, key: str) -> bool:
    """True when ``line`` opens ``key`` at column zero (no leading whitespace)."""
    return bool(line) and not line[0].isspace() and _is_key(line, key)


def read_seed(run_dir: Path) -> dict[str, Any]:
    """Read ``task.seed`` back out of a run's own resolved Hydra config.

    The value is taken from ``training/.hydra/config.yaml`` inside the run
    directory. That file is what the run resolved, so it reflects whether an
    override was actually applied -- unlike the command line, which records only
    what was requested.

    Parameters
    ----------
    run_dir : Path
        The directory passed to ``hydra.run.dir``.

    Returns
    -------
    dict
        ``config_path``, ``seed``, ``line_number`` and ``raw_line`` -- the last
        two so the caller can quote the file rather than paraphrase it.

    Raises
    ------
    EnvCheckError
        If the config is missing, or carries no ``task.seed`` key.
    """
    import yaml

    config_path = run_dir / HYDRA_CONFIG_RELPATH
    if not config_path.is_file():
        raise EnvCheckError(f"no Hydra config at {config_path}")

    text = config_path.read_text()
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # Callers catch EnvCheckError. A raw ParserError escaping here reaches
        # main's broad handler and is reported, but a DIRECT caller of read_seed
        # gets an exception type the function never documents.
        raise EnvCheckError(f"{config_path} is not valid YAML: {exc}") from exc
    if parsed is not None and not isinstance(parsed, dict):
        # A valid top-level list or string previously raised AttributeError from
        # .get() -- a type error masquerading as a bug rather than as bad input.
        raise EnvCheckError(
            f"{config_path} is valid YAML but its top level is "
            f"{type(parsed).__name__}, not a mapping"
        )
    task = (parsed or {}).get("task")
    if not isinstance(task, dict) or "seed" not in task:
        raise EnvCheckError(
            f"{config_path} has no task.seed key "
            f"(task keys: {sorted(task) if isinstance(task, dict) else task!r})"
        )
    seed = task["seed"]

    # Locate the literal line so the evidence can quote the file rather than
    # paraphrase it. The scan is bounded to the top-level `task:` block on
    # purpose. In the reference config `seed` happens to be unique, so an
    # unbounded search would agree today -- but `ansatz:`, `hamil:` and
    # `logging:` are sibling top-level blocks, and a future ansatz carrying its
    # own `seed:` would silently make an unbounded search quote the wrong line
    # while still reporting a number.
    lines = text.splitlines()
    line_number: int | None = None
    raw_line: str | None = None
    in_task = False
    seed_line_count = 0
    for index, line in enumerate(lines, start=1):
        # Accept the shapes YAML permits for the same key, not just the one the
        # reference config happens to use: `task:`, `task :`, `"task":`,
        # `'task':`. Matching only the literal `task:` made the scanner reject
        # valid configurations, which is a FALSE REFUSAL -- as bad as a missing
        # check and harder to notice, because tests supply the shapes the author
        # expects.
        if _is_top_level_key(line, "task"):
            in_task = True
            continue
        if in_task:
            # A non-indented, non-blank line ends the top-level `task:` block.
            if line and not line[0].isspace():
                break
            if _is_key(line.strip(), "seed"):
                seed_line_count += 1
                if line_number is None:
                    line_number = index
                    raw_line = line

    # yaml.safe_load silently keeps the LAST duplicate key while the raw-line scan
    # above quotes the FIRST, so a duplicated `seed:` produced a value and a
    # quoted line that DISAGREE, with no error. Since the whole point of this
    # function is to quote the file as evidence, contradictory evidence is worse
    # than no evidence.
    # The whole contract of this function is to QUOTE THE FILE as evidence. Inline
    # flow-style YAML (`task: {seed: 7}`) parses fine and yields a seed, but the
    # line scan finds no `seed:` line of its own, so it previously returned
    # line_number=None and raw_line=None alongside a confident value -- a
    # successful check with nothing to show for it. A4's evidence requirement
    # says "quoted FROM THE FILE", and None is not a quotation.
    if seed_line_count > 1:
        raise EnvCheckError(
            f"{config_path} has {seed_line_count} `seed:` lines inside the top-level "
            "task block; YAML keeps the last and the quoted line would be the first, "
            "so the reported value and its evidence would disagree"
        )

    result = {
        "config_path": str(config_path),
        "seed": seed,
        "line_number": line_number,
        "raw_line": raw_line,
        # A4 requires the value be quoted FROM THE FILE. Some legitimate YAML
        # shapes -- inline flow mappings, merge keys, anchors -- carry no
        # `seed:` line of their own, so the value is real but unquotable.
        # Refusing those was a false refusal; reporting a bare None was a
        # silent gap. State it instead, so a caller can see that the value
        # is verified and the EVIDENCE is not available.
        "evidence_available": line_number is not None,
    }
    if line_number is None:
        result["warning"] = (
            f"task.seed is {seed!r} but no `seed:` line exists in the top-level "
            "task block to quote (inline flow mapping, anchor or merge key?). "
            "The value is verified; the file-quoted evidence A4 asks for is not "
            "available from this config."
        )
    return result


def check_seed(run_dir: Path, expect_seed: int) -> dict[str, Any]:
    """Assert that a requested seed override reached the run's resolved config.

    Parameters
    ----------
    run_dir : Path
        The directory passed to ``hydra.run.dir``.
    expect_seed : int
        The seed the launcher asked for on the command line.

    Returns
    -------
    dict
        The result of :func:`read_seed` plus ``expected`` and ``ok``.

    Raises
    ------
    EnvCheckError
        If the config's seed differs from the requested one. This is the failure
        that otherwise produces a whole seed-spread of identical rows.
    """
    found = read_seed(run_dir)
    found["expected"] = expect_seed
    found["ok"] = found["seed"] == expect_seed
    if not found["ok"]:
        raise EnvCheckError(
            f"task.seed in {found['config_path']} is {found['seed']!r}, "
            f"but {expect_seed!r} was requested: the override did not take"
        )
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns 0 on success, 1 on a failed check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="report interpreter, JAX, devices, DeepQMC commit")
    env.add_argument("--expect-prefix", default=None)
    env.add_argument("--expect-jax", default=None)
    env.add_argument("--expect-commit", default=None)
    env.add_argument("--source-root", type=Path, default=None)
    env.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="do not require a visible GPU (login-node inspection only)",
    )

    seed = sub.add_parser("seed", help="verify task.seed in the run's resolved config")
    seed.add_argument("--run-dir", type=Path, required=True)
    seed.add_argument("--expect-seed", type=int, required=True)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(json.dumps({"ok": False, "error": f"invalid arguments (argparse exit {exc.code})"}, indent=2))
        raise
    try:
        if args.command == "env":
            report = check_env(
                expect_prefix=args.expect_prefix,
                expect_jax=args.expect_jax,
                expect_commit=args.expect_commit,
                source_root=args.source_root,
                require_gpu=not args.allow_no_gpu,
            )
        else:
            report = check_seed(args.run_dir, args.expect_seed)
    except EnvCheckError as exc:
        payload: dict[str, Any] = {"ok": False, "error": str(exc)}
        # Emit whatever WAS established before the check failed.
        if getattr(exc, "report", None):
            payload["report"] = exc.report
        print(json.dumps(payload, indent=2))
        return 1
    except Exception as exc:  # noqa: BLE001
        # THE ARTEFACT MUST NEVER BE EMPTY. The caller redirects stdout into
        # env-check.json, so an uncaught exception writes its traceback to
        # stderr and leaves a 0-byte file -- which is what PBS 7571666 produced,
        # and a 0-byte file records nothing about which environment failed.
        # Any unanticipated failure still yields valid, parseable JSON.
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
