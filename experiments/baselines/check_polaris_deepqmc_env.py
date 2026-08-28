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
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# Relative to the run directory Hydra was pointed at with ``hydra.run.dir``.
HYDRA_CONFIG_RELPATH = Path("training") / ".hydra" / "config.yaml"


class EnvCheckError(RuntimeError):
    """Raised when a checked property of the environment or run is not as required."""


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
    interesting = ("libcudart", "libcublas", "libcudnn", "libcufft", "libcusolver", "libnvrtc")
    found: dict[str, str] = {}
    for line in maps.read_text().splitlines():
        # Path is the last whitespace-separated field, when present at all.
        path = line.rsplit(" ", 1)[-1]
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
        if entry["preallocate_env"] != "false":
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
        # from the package directory finds the source root.
        root = Path(deepqmc.__file__).resolve().parents[2]
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
        raise EnvCheckError("; ".join(failures))
    return report


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
    parsed = yaml.safe_load(text)
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
    for index, line in enumerate(lines, start=1):
        if line.startswith("task:"):
            in_task = True
            continue
        if in_task:
            # A non-indented, non-blank line ends the top-level `task:` block.
            if line and not line[0].isspace():
                break
            if line.strip().startswith("seed:"):
                line_number = index
                raw_line = line
                break

    return {
        "config_path": str(config_path),
        "seed": seed,
        "line_number": line_number,
        "raw_line": raw_line,
    }


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


# --- A5 pre-registered comparison ------------------------------------------
# These thresholds were fixed BEFORE any Polaris energy existed (Task
# Orchestrator note a5-preregistered-criterion-2026-08-28) so the result cannot
# choose its own criterion. They are encoded here rather than applied by hand
# because a narrated verdict drifts from its numbers: a label reading "sane"
# beside a difference that is not is a failure this program has already seen.
SANE_HARTREE = 1e-4
BROKEN_HARTREE = 1e-3
# Exact non-relativistic, infinite-nuclear-mass He ground state. Aznabaev,
# Bekbaev and Korobov, arXiv:1810.11288 Table 3, attributing to Schwartz (2006);
# recorded as confirmed in NNQMC-REFERENCE-ENERGIES.md.
HE_EXACT_HARTREE = -2.903724377034119598


def compare_energy(
    polaris_energy: float,
    polaris_stderr: float,
    reference_energy: float,
    exact_energy: float | None = HE_EXACT_HARTREE,
    steps_observed: int | None = None,
    steps_expected: int | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered A5 criterion to a Polaris energy.

    Parameters
    ----------
    polaris_energy, polaris_stderr : float
        The Polaris row's tail-mean energy and its blocked standard error, in
        hartree, from the same estimator used for the reference row.
    reference_energy : float
        The Cannon comparator's energy in hartree.
    exact_energy : float or None
        Exact ground-state energy used for the variational check. Pass ``None``
        to skip that check for a system with no exact value.
    steps_observed, steps_expected : int or None
        Row lengths. A short row is BROKEN regardless of its energy, because a
        truncated run's tail mean is not the quantity being compared.

    Returns
    -------
    dict
        ``delta``, ``verdict`` (``sane`` / ``investigate`` / ``broken``) and
        ``reasons``. The verdict is derived, never passed in.
    """
    delta = polaris_energy - reference_energy
    magnitude = abs(delta)
    reasons: list[str] = []

    # Any single disqualifier forces BROKEN regardless of how small delta is.
    if not all(map(_finite, (polaris_energy, polaris_stderr))):
        reasons.append("energy or stderr is not finite")
    if steps_expected is not None and (steps_observed or 0) < steps_expected:
        reasons.append(
            f"row is short: {steps_observed} of {steps_expected} steps recorded"
        )
    if exact_energy is not None and polaris_stderr > 0:
        # A variational energy cannot lie below the exact ground state. This
        # program has produced below-exact energies four times from too-short
        # tails, so it is a live failure mode rather than a hypothetical.
        below_by = exact_energy - polaris_energy
        if below_by > 3 * polaris_stderr:
            reasons.append(
                f"energy is {below_by:.3e} Ha below exact, more than 3 sigma "
                f"({3 * polaris_stderr:.3e} Ha): variationally impossible"
            )

    if reasons:
        verdict = "broken"
    elif magnitude >= BROKEN_HARTREE:
        verdict = "broken"
        reasons.append(f"|delta| {magnitude:.3e} Ha >= {BROKEN_HARTREE:.0e}")
    elif magnitude >= SANE_HARTREE:
        verdict = "investigate"
        reasons.append(
            f"|delta| {magnitude:.3e} Ha is in [{SANE_HARTREE:.0e}, "
            f"{BROKEN_HARTREE:.0e}): report as a finding, do NOT call this verified"
        )
    else:
        verdict = "sane"
        reasons.append(f"|delta| {magnitude:.3e} Ha < {SANE_HARTREE:.0e}")

    return {
        "polaris_energy_hartree": polaris_energy,
        "polaris_stderr_hartree": polaris_stderr,
        "reference_energy_hartree": reference_energy,
        "exact_energy_hartree": exact_energy,
        "delta_hartree": delta,
        "abs_delta_hartree": magnitude,
        "thresholds": {"sane_below": SANE_HARTREE, "broken_at_or_above": BROKEN_HARTREE},
        "verdict": verdict,
        "reasons": reasons,
    }


def _finite(value: float) -> bool:
    """True when ``value`` is a finite real number."""
    return isinstance(value, (int, float)) and math.isfinite(value)


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

    cmp_ = sub.add_parser("compare", help="apply the pre-registered A5 criterion")
    cmp_.add_argument("--polaris-energy", type=float, required=True)
    cmp_.add_argument("--polaris-stderr", type=float, required=True)
    cmp_.add_argument("--reference-energy", type=float, required=True)
    cmp_.add_argument("--exact-energy", type=float, default=HE_EXACT_HARTREE)
    cmp_.add_argument("--steps-observed", type=int, default=None)
    cmp_.add_argument("--steps-expected", type=int, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "env":
            report = check_env(
                expect_prefix=args.expect_prefix,
                expect_jax=args.expect_jax,
                expect_commit=args.expect_commit,
                source_root=args.source_root,
                require_gpu=not args.allow_no_gpu,
            )
        elif args.command == "seed":
            report = check_seed(args.run_dir, args.expect_seed)
        else:
            report = compare_energy(
                polaris_energy=args.polaris_energy,
                polaris_stderr=args.polaris_stderr,
                reference_energy=args.reference_energy,
                exact_energy=args.exact_energy,
                steps_observed=args.steps_observed,
                steps_expected=args.steps_expected,
            )
            print(json.dumps(report, indent=2))
            # A non-sane verdict must not exit 0: a green exit beside an
            # investigate verdict is how a finding gets lost in a job log.
            return 0 if report["verdict"] == "sane" else 1
    except EnvCheckError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
