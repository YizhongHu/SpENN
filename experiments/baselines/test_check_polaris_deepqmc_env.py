"""Tests for the DeepQMC-on-Polaris seed check.

The fixture in ``testdata/deepqmc_hydra_config.yaml`` is a byte copy of a real
``training/.hydra/config.yaml`` from Cannon run
``dqmc-he-default-200k-seedspread-42610526_1``, taken 2026-08-28. It is used
verbatim rather than reduced to a three-line stub because the property under
test is precisely the one a stub destroys: in the real file ``task:`` is line 1
and ``seed:`` is line 50, forty-nine lines apart, across a 283-line config with
four top-level blocks. A stub would pass whatever the reader wrote it to pass.

The ``env`` subcommand is not covered here. It requires jax, a GPU and a
DeepQMC checkout, none of which exist in this suite; its evidence is the PBS job
log instead.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from experiments.baselines.check_polaris_deepqmc_env import (
    HE_EXACT_HARTREE,
    HYDRA_CONFIG_RELPATH,
    EnvCheckError,
    check_seed,
    backend_platform_version,
    compare_energy,
    loaded_cuda_libraries,
    read_seed,
)

FIXTURE = Path(__file__).parent / "testdata" / "deepqmc_hydra_config.yaml"


def _run_dir(tmp_path: Path, config_text: str | None = None) -> Path:
    """Materialise a run directory holding a Hydra config at the real relpath."""
    run_dir = tmp_path / "run"
    target = run_dir / HYDRA_CONFIG_RELPATH
    target.parent.mkdir(parents=True)
    if config_text is None:
        shutil.copyfile(FIXTURE, target)
    else:
        target.write_text(config_text)
    return run_dir


def test_fixture_still_has_the_shape_that_makes_this_check_necessary() -> None:
    """Guard the fixture itself: if it is ever trimmed, the other tests weaken."""
    lines = FIXTURE.read_text().splitlines()
    assert lines[0] == "task:"
    seed_lines = [i for i, line in enumerate(lines, 1) if line.strip().startswith("seed:")]
    assert seed_lines == [50], seed_lines
    # Far enough below `task:` that a two-line-context grep misses it entirely,
    # which is the exact way this check gets skipped in practice.
    assert seed_lines[0] - 1 > 2
    top_level = [line for line in lines if line and not line[0].isspace()]
    assert top_level == ["task:", "logging:", "ansatz:", "hamil:"], top_level


def test_read_seed_reports_value_line_number_and_raw_line(tmp_path: Path) -> None:
    found = read_seed(_run_dir(tmp_path))
    assert found["seed"] == 1
    # The line number and raw line are what let a receipt quote the file.
    assert found["line_number"] == 50
    assert found["raw_line"] == "  seed: 1"


def test_check_seed_passes_when_the_override_took(tmp_path: Path) -> None:
    result = check_seed(_run_dir(tmp_path), expect_seed=1)
    assert result["ok"] is True
    assert result["expected"] == 1


def test_check_seed_raises_when_the_override_did_not_take(tmp_path: Path) -> None:
    """The whole point: a requested seed that never reached the config must fail.

    Without this the run trains happily on the default seed and the row looks
    legitimate.
    """
    with pytest.raises(EnvCheckError) as excinfo:
        check_seed(_run_dir(tmp_path), expect_seed=7)
    message = str(excinfo.value)
    assert "did not take" in message
    # Both numbers must appear, or the failure cannot be diagnosed from the log.
    assert "1" in message and "7" in message


def test_missing_config_is_an_error_not_a_silent_pass(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EnvCheckError, match="no Hydra config"):
        read_seed(empty)


def test_config_without_a_task_seed_key_is_an_error(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, "task:\n  steps: 200000\nansatz:\n  name: default\n")
    with pytest.raises(EnvCheckError, match="no task.seed key"):
        read_seed(run_dir)


def test_seed_outside_the_task_block_is_not_mistaken_for_task_seed(tmp_path: Path) -> None:
    """A sibling block's own `seed:` must not be reported as `task.seed`.

    An unbounded search for the first `seed:` line would quote the ansatz block
    here and still return a plausible-looking line number.
    """
    text = "ansatz:\n  seed: 99\ntask:\n  steps: 10\n  seed: 4\n"
    found = read_seed(_run_dir(tmp_path, text))
    assert found["seed"] == 4
    assert found["line_number"] == 5
    assert found["raw_line"] == "  seed: 4"


# --- A5 pre-registered comparison -------------------------------------------
# The comparator is Cannon run dqmc-he-39358341/default: 20000 steps, batch 4096,
# seed 0, code_commit edf373e7, on an A100-SXM4-80GB.
CANNON_HE_DEFAULT_20K = -2.9036863634347916
CANNON_HE_DEFAULT_20K_STDERR = 1.6026224325770225e-05


def test_a_matching_energy_is_sane() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 2e-5,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "sane"


def test_the_investigate_band_is_not_silently_absorbed_into_either_verdict() -> None:
    """A difference between 1e-4 and 1e-3 is a finding, not a pass and not a fail."""
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 5e-4,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "investigate"


def test_a_third_decimal_difference_is_broken() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 2e-3,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "broken"


def test_below_exact_is_broken_even_when_the_delta_is_tiny() -> None:
    """The disqualifier must outrank a small delta, not be averaged with it.

    An energy can sit very close to the Cannon row and still be variationally
    impossible. This program has produced below-exact energies four times from
    too-short tails, so this is the check that must not be conditional on delta.
    """
    below = HE_EXACT_HARTREE - 1e-3
    result = compare_energy(
        polaris_energy=below,
        polaris_stderr=1e-6,
        reference_energy=below,  # delta is exactly zero
    )
    assert result["abs_delta_hartree"] == 0.0
    assert result["verdict"] == "broken"
    assert any("variationally impossible" in r for r in result["reasons"])


def test_a_short_row_is_broken_regardless_of_energy() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        steps_observed=11000,
        steps_expected=20000,
    )
    assert result["verdict"] == "broken"
    assert any("short" in r for r in result["reasons"])


def test_non_finite_energy_is_broken() -> None:
    result = compare_energy(
        polaris_energy=float("nan"),
        polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "broken"


def test_verdict_is_derived_from_the_numbers_it_reports() -> None:
    """Guard against a label drifting from the value printed beside it."""
    for offset in (0.0, 5e-5, 2e-4, 5e-3):
        result = compare_energy(
            polaris_energy=CANNON_HE_DEFAULT_20K + offset,
            polaris_stderr=CANNON_HE_DEFAULT_20K_STDERR,
            reference_energy=CANNON_HE_DEFAULT_20K,
        )
        magnitude = result["abs_delta_hartree"]
        expected = (
            "sane" if magnitude < 1e-4 else "investigate" if magnitude < 1e-3 else "broken"
        )
        assert result["verdict"] == expected, (offset, result)


# --- Optional-evidence collection must never be fatal ------------------------
# Regression for PBS 7571666 on Polaris: check_env reached
# jax.extend.backend.get_backend() to read an OPTIONAL context field. On jax
# 0.8.3 `jax.extend` raises AttributeError from a deprecation shim, which aborted
# the whole environment check and left a 0-byte env-check.json -- losing the
# interpreter, device kind and DeepQMC commit evidence that the acceptance
# criteria actually require, none of which had anything to do with the failure.


class _StubDevice:
    def __init__(self, client: object | None) -> None:
        self.client = client
        self.id = 0
        self.device_kind = "stub"
        self.platform = "gpu"

    def memory_stats(self) -> dict[str, int]:
        # A fresh process that has done no device work: the exact case whose
        # zero reading must not be mistaken for a training run's high-water.
        return {"peak_bytes_in_use": 0}


class _StubClient:
    platform_version = "cuda 12090"


class _JaxWithoutExtend:
    """Reproduces jax 0.8.3: touching `.extend` raises, as the real shim does."""

    def __init__(self, devices: list[_StubDevice]) -> None:
        self._devices = devices

    def devices(self) -> list[_StubDevice]:
        return self._devices

    def __getattr__(self, name: str):
        # The exact message the real deprecation shim produced in the job log.
        raise AttributeError(f"module 'jax' has no attribute {name!r}")


def test_platform_version_is_read_through_the_device_client() -> None:
    result = backend_platform_version(_JaxWithoutExtend([_StubDevice(_StubClient())]))
    assert result["platform_version"] == "cuda 12090"
    assert result["via"] == "device.client"


def test_missing_jax_extend_does_not_raise() -> None:
    """The exact Polaris failure: every route dead, and it must still return."""
    stub = _JaxWithoutExtend([_StubDevice(None)])  # client is None -> attribute error
    result = backend_platform_version(stub)
    assert result["platform_version"] is None
    assert result["via"] is None
    # The failures are reported as data rather than raised, so the job log says
    # why the field is absent instead of losing everything else.
    assert result["attempts"], result
    assert any("device.client" in a for a in result["attempts"])


def test_no_devices_at_all_does_not_raise() -> None:
    result = backend_platform_version(_JaxWithoutExtend([]))
    assert result["platform_version"] is None
    assert any("IndexError" in a for a in result["attempts"]), result["attempts"]


def test_loaded_cuda_libraries_never_raises_off_linux() -> None:
    """Returns an empty mapping where /proc/self/maps does not exist."""
    assert isinstance(loaded_cuda_libraries(), dict)


# --- The artefact must never be empty ---------------------------------------
# The job pipes this command's stdout into env-check.json. An uncaught exception
# writes a traceback to stderr and leaves a 0-byte file, which records nothing
# about WHICH environment failed -- the outcome of PBS 7571666. Guarding the
# optional collectors is not sufficient on its own: the report is serialised at
# the end, so any late raise still empties the file. These tests assert on what
# actually reaches stdout, which is what becomes the artefact.


def _stdout_json(capsys) -> dict:
    out = capsys.readouterr().out
    assert out.strip(), "nothing was written to stdout: the artefact would be 0 bytes"
    return json.loads(out)


def test_unexpected_exception_still_writes_parseable_json(capsys, monkeypatch) -> None:
    """An error nobody anticipated must still leave a usable artefact."""
    import experiments.baselines.check_polaris_deepqmc_env as mod

    def boom(**_kwargs):
        raise AttributeError("module 'jax' has no attribute 'extend'")

    monkeypatch.setattr(mod, "check_env", boom)
    assert mod.main(["env"]) == 1
    payload = _stdout_json(capsys)
    assert payload["ok"] is False
    assert "no attribute 'extend'" in payload["error"]
    # The traceback is retained: knowing where it failed is the diagnostic value.
    assert "traceback" in payload


def test_a_failed_required_check_still_reports_what_was_established(
    capsys, monkeypatch
) -> None:
    """A wrong jax version must not discard the interpreter and commit evidence."""
    import experiments.baselines.check_polaris_deepqmc_env as mod

    partial = {"prefix": "/some/venv", "jax": "0.9.2", "ok": False, "failures": ["jax mismatch"]}

    def failing(**_kwargs):
        raise mod.EnvCheckError("jax '0.9.2' != expected '0.8.3'", report=partial)

    monkeypatch.setattr(mod, "check_env", failing)
    assert mod.main(["env"]) == 1
    payload = _stdout_json(capsys)
    assert payload["ok"] is False
    # The partial evidence survives the failure.
    assert payload["report"]["prefix"] == "/some/venv"
    assert payload["report"]["jax"] == "0.9.2"


def test_seed_failure_also_writes_json_rather_than_a_traceback(
    capsys, tmp_path
) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    from experiments.baselines.check_polaris_deepqmc_env import main

    assert main(["seed", "--run-dir", str(empty), "--expect-seed", "7"]) == 1
    payload = _stdout_json(capsys)
    assert payload["ok"] is False


# --- check_env's own failure path, driven for real ---------------------------
# The tests above monkeypatch check_env wholesale, so they establish that main()
# FORWARDS a partial report but not that check_env ATTACHES one. That gap was
# found by mutation: removing `report=report` from the raise killed no test.
# These drive the real function with stub jax/deepqmc modules instead.


class _StubJax:
    __version__ = "0.9.2"

    def devices(self):
        return [_StubDevice(_StubClient())]


def _install_stub_modules(monkeypatch, tmp_path: Path) -> None:
    """Make `import jax` and `import deepqmc` inside check_env resolve to stubs."""
    import types

    deepqmc = types.ModuleType("deepqmc")
    # An editable install leaves __file__ inside the checkout; check_env walks up
    # two parents from it, so the path needs that much depth to be realistic.
    pkg = tmp_path / "src" / "deepqmc"
    pkg.mkdir(parents=True)
    deepqmc.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "jax", _StubJax())
    monkeypatch.setitem(sys.modules, "deepqmc", deepqmc)


def test_check_env_attaches_the_partial_report_to_its_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A required-check failure must carry the evidence gathered before it."""
    _install_stub_modules(monkeypatch, tmp_path)
    from experiments.baselines.check_polaris_deepqmc_env import EnvCheckError, check_env

    with pytest.raises(EnvCheckError) as excinfo:
        check_env(
            expect_prefix=None,
            expect_jax="0.8.3",  # stub reports 0.9.2, so this must fail
            expect_commit=None,
            source_root=None,
            require_gpu=False,
        )
    report = excinfo.value.report
    assert report is not None, "the partial report was discarded on failure"
    # The evidence established BEFORE the failing assertion survives it.
    assert report["jax"] == "0.9.2"
    assert report["python"] == sys.version.split()[0]
    assert report["ok"] is False
    assert any("0.8.3" in f for f in report["failures"])


def test_check_env_requires_a_gpu_when_asked(monkeypatch, tmp_path: Path) -> None:
    """require_gpu must actually fail on a CPU-only backend, not just record it."""
    _install_stub_modules(monkeypatch, tmp_path)
    from experiments.baselines.check_polaris_deepqmc_env import EnvCheckError, check_env

    cpu_device = _StubDevice(_StubClient())
    cpu_device.platform = "cpu"
    monkeypatch.setattr(sys.modules["jax"], "devices", lambda: [cpu_device])
    with pytest.raises(EnvCheckError, match="no GPU device visible"):
        check_env(
            expect_prefix=None,
            expect_jax=None,
            expect_commit=None,
            source_root=None,
            require_gpu=True,
        )


# --- Cross-facility criterion, the second pre-registered test ----------------
# Separate from physics validity and TIGHTER than it: 3 combined sigma is about
# 6.8e-5 Ha against the locked 1e-4 SANE threshold, so a result can pass the
# physics test and fail this one.
CANNON_STDERR = 1.6026224325770225e-05


def test_agreement_within_combined_sigma_is_consistent() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 2e-5,
        polaris_stderr=CANNON_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        reference_stderr=CANNON_STDERR,
    )
    assert result["cross_facility"]["verdict"] == "consistent"
    assert result["cross_facility"]["delta_in_sigma"] < 3


def test_the_cross_facility_test_is_tighter_than_the_physics_threshold() -> None:
    """A delta can pass SANE and still fail cross-facility. That is the point.

    If these two ever agree on every input, one of them is redundant and the
    pre-registration of both was pointless.
    """
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K + 9e-5,  # under the 1e-4 SANE line
        polaris_stderr=CANNON_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        reference_stderr=CANNON_STDERR,
    )
    assert result["verdict"] == "sane"
    assert result["cross_facility"]["verdict"] in ("marginal", "inconsistent")


def test_mismatched_estimator_windows_refuse_to_combine_sigmas(monkeypatch) -> None:
    """Two sigmas from different tail windows are not the same quantity.

    The emitted record schema cannot express this -- a 10000-step tail and a
    750-step tail both serialise as estimator "training_tail" -- so the check
    has to live where the windows are still known.
    """
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        reference_stderr=CANNON_STDERR,
        tail_steps=750,
        reference_tail_steps=10000,
    )
    assert result["estimator_windows"]["comparable"] is False
    assert result["cross_facility"]["verdict"] == "not_evaluated"
    assert "not the same quantity" in result["cross_facility"]["reason"]


def test_matching_windows_are_marked_comparable() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        reference_stderr=CANNON_STDERR,
        tail_steps=10000,
        reference_tail_steps=10000,
    )
    assert result["estimator_windows"]["comparable"] is True
    assert result["cross_facility"]["verdict"] == "consistent"


def test_absent_reference_stderr_is_not_evaluated_rather_than_passed() -> None:
    """A missing input must not read as agreement."""
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["cross_facility"]["verdict"] == "not_evaluated"


def test_exit_status_reflects_the_cross_facility_verdict(capsys) -> None:
    """A physics-SANE result that fails cross-facility must NOT exit 0.

    Found by mutation: removing the cross_facility term from main's exit
    condition killed no test, because every existing test asserted on
    compare_energy's RETURN VALUE and none on main's EXIT STATUS. The verdict
    logic and the wiring that acts on it are two links, and only one was tested.
    A green exit is what a job script keys on, so this is the link that matters
    operationally.
    """
    from experiments.baselines.check_polaris_deepqmc_env import main

    # 9e-5 Ha: inside the locked 1e-4 SANE threshold, but ~4 combined sigma.
    rc = main([
        "compare",
        "--polaris-energy", repr(CANNON_HE_DEFAULT_20K + 9e-5),
        "--polaris-stderr", repr(CANNON_STDERR),
        "--reference-energy", repr(CANNON_HE_DEFAULT_20K),
        "--reference-stderr", repr(CANNON_STDERR),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "sane"
    assert payload["cross_facility"]["verdict"] in ("marginal", "inconsistent")
    assert rc == 1, "exit 0 would report the weaker of the two criteria as the answer"


def test_exit_status_is_zero_when_both_criteria_pass(capsys) -> None:
    from experiments.baselines.check_polaris_deepqmc_env import main

    rc = main([
        "compare",
        "--polaris-energy", repr(CANNON_HE_DEFAULT_20K + 1e-5),
        "--polaris-stderr", repr(CANNON_STDERR),
        "--reference-energy", repr(CANNON_HE_DEFAULT_20K),
        "--reference-stderr", repr(CANNON_STDERR),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "sane"
    assert payload["cross_facility"]["verdict"] == "consistent"
    assert rc == 0


def test_a_low_z_is_labelled_as_ordinary_not_as_tight_agreement() -> None:
    """Guard against "consistent at 0.21 sigma" being requoted as sub-error-bar agreement.

    E|Z| for one draw is sqrt(2/pi) = 0.798, so |z| < 1 is the ordinary outcome.
    Claiming the facilities agree better than their error bars is a statement
    about a distribution of z, and one pair cannot support it.
    """
    result = compare_energy(
        polaris_energy=-2.9036814794540406,
        polaris_stderr=1.6173570569821275e-05,
        reference_energy=-2.9036863634347916,
        reference_stderr=1.6026224325770225e-05,
    )
    cross = result["cross_facility"]
    assert cross["verdict"] == "consistent"
    assert round(cross["delta_in_sigma"], 4) == 0.2145
    # The number travels with the caveat that bounds how it may be read.
    assert cross["expected_abs_z_for_one_draw"] == pytest.approx(0.7979, abs=1e-4)
    assert "NOT evidence of sub-error-bar agreement" in cross["interpretation"]


# --- Gaps found by an independent verifier ----------------------------------
# All three of these were reported against a SHA whose own mutation grid was
# clean. My mutants were chosen by me and so inherited my blind spots; these
# were chosen by someone who had not built the thing.


def test_a_non_finite_reference_energy_is_broken_not_sane(capsys) -> None:
    """A nan comparator must not read as agreement.

    nan comparisons are all False, so `magnitude >= BROKEN` and `magnitude >=
    SANE` both failed and control fell through to the sane branch. A missing or
    corrupt comparator therefore produced verdict "sane" with delta nan -- a
    wrong verdict in the reassuring direction. My own tests only ever passed
    finite reference values, so the input was never exercised.
    """
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_STDERR,
        reference_energy=float("nan"),
    )
    assert result["verdict"] == "broken"
    assert any("not finite" in r for r in result["reasons"])


def test_a_non_finite_polaris_stderr_is_broken() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=float("inf"),
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "broken"


def test_zero_peak_memory_carries_the_checker_only_warning(monkeypatch) -> None:
    """The zero-peak warning had NO test: a mutant flipping `== 0` to `< 0` survived.

    That warning is the only thing stopping a fresh checker process's 0 MiB
    reading from being quoted as a training run's memory high-water, which is
    how E6 was nearly recorded as satisfied on a meaningless number.
    """
    import experiments.baselines.check_polaris_deepqmc_env as mod

    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setitem(sys.modules, "jax", _StubJax())
    stats = mod.gpu_memory_high_water()
    assert stats[0]["peak_bytes_in_use"] == 0
    assert "warning" in stats[0], "a zero peak must say it measures nothing"
    assert "says NOTHING" in stats[0]["warning"]
    assert stats[0]["scope"].startswith("this process only")


def test_bad_arguments_still_emit_json_not_an_empty_artefact(capsys) -> None:
    """argparse exits before main's handler, leaving stdout empty.

    The job pipes stdout into a file, so a typo in the JOB SCRIPT would have
    produced exactly the 0-byte artefact PBS 7571666 produced. My claim that any
    exception yields JSON was too broad; argparse was outside it.
    """
    from experiments.baselines.check_polaris_deepqmc_env import main

    with pytest.raises(SystemExit):
        main(["definitely-not-a-subcommand"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "invalid arguments" in payload["error"]


# --- Degenerate-input sweep -------------------------------------------------
# Found by asking, systematically, what each verdict function returns for absent,
# nan, inf, zero, negative and wrong-typed inputs -- not by mutation. No mutant
# could reach these: mutants probe the code you wrote against the inputs you
# thought of, and no test had ever supplied a negative uncertainty.


def test_a_negative_polaris_stderr_is_broken_not_sane() -> None:
    """A negative standard error is a corrupt input, not a small one.

    It passes the finite check -- it IS a finite number -- and then sails through
    every threshold, because the thresholds only look at |delta|.
    """
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=-1.6e-05,
        reference_energy=CANNON_HE_DEFAULT_20K,
    )
    assert result["verdict"] == "broken"
    assert any("negative" in r for r in result["reasons"])


def test_a_negative_reference_stderr_does_not_yield_consistent() -> None:
    """Combining SQUARES the stderr, so a sign error produces a plausible sigma.

    Squaring is exactly the operation that hides the corruption: the resulting
    combined sigma looks entirely ordinary and the verdict came back "consistent".
    """
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        reference_stderr=-CANNON_STDERR,
    )
    assert result["cross_facility"]["verdict"] == "not_evaluated"
    assert "unusable uncertainties" in result["cross_facility"]["reason"]


def test_a_non_finite_reference_stderr_is_not_evaluated_with_a_reason() -> None:
    result = compare_energy(
        polaris_energy=CANNON_HE_DEFAULT_20K,
        polaris_stderr=CANNON_STDERR,
        reference_energy=CANNON_HE_DEFAULT_20K,
        reference_stderr=float("nan"),
    )
    assert result["cross_facility"]["verdict"] == "not_evaluated"
    assert "unusable uncertainties" in result["cross_facility"]["reason"]


def test_degenerate_inputs_never_reach_the_permissive_branch() -> None:
    """A verdict function falling through to its permissive branch on degenerate
    input is worse than one that crashes, because the crash is legible.

    This sweeps the whole degenerate space in one assertion so a future input
    kind cannot be added without someone deciding what it should do.
    """
    nan, inf = float("nan"), float("inf")
    degenerate = [
        dict(polaris_energy=nan, polaris_stderr=CANNON_STDERR, reference_energy=CANNON_HE_DEFAULT_20K),
        dict(polaris_energy=CANNON_HE_DEFAULT_20K, polaris_stderr=nan, reference_energy=CANNON_HE_DEFAULT_20K),
        dict(polaris_energy=CANNON_HE_DEFAULT_20K, polaris_stderr=CANNON_STDERR, reference_energy=nan),
        dict(polaris_energy=inf, polaris_stderr=CANNON_STDERR, reference_energy=CANNON_HE_DEFAULT_20K),
        dict(polaris_energy=CANNON_HE_DEFAULT_20K, polaris_stderr=CANNON_STDERR, reference_energy=inf),
        dict(polaris_energy=CANNON_HE_DEFAULT_20K, polaris_stderr=-CANNON_STDERR, reference_energy=CANNON_HE_DEFAULT_20K),
    ]
    for kwargs in degenerate:
        assert compare_energy(**kwargs)["verdict"] == "broken", kwargs
