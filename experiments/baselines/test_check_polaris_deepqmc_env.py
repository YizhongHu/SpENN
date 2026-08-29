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
    HYDRA_CONFIG_RELPATH,
    EnvCheckError,
    check_seed,
    backend_platform_version,
    loaded_cuda_libraries,
    parse_proc_maps,
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


# The comparator is Cannon run dqmc-he-39358341/default: 20000 steps, batch 4096,
# seed 0, code_commit edf373e7, on an A100-SXM4-80GB.


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


# Separate from physics validity and TIGHTER than it: 3 combined sigma is about
# 6.8e-5 Ha against the locked 1e-4 SANE threshold, so a result can pass the
# physics test and fail this one.


# --- Gaps found by an independent verifier ----------------------------------
# All three of these were reported against a SHA whose own mutation grid was
# clean. My mutants were chosen by me and so inherited my blind spots; these
# were chosen by someone who had not built the thing.


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


# Found by asking, systematically, what each verdict function returns for absent,
# nan, inf, zero, negative and wrong-typed inputs -- not by mutation. No mutant
# could reach these: mutants probe the code you wrote against the inputs you
# thought of, and no test had ever supplied a negative uncertainty.


# --- Second verifier's findings ---------------------------------------------


MAPS_SAMPLE = """\
7f0e00000000-7f0e00021000 r--p 00000000 08:01 1234 /usr/lib/x86_64-linux-gnu/libc.so.6
7f0e10000000-7f0e10a00000 r-xp 00000000 08:01 5678 /venv/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
7f0e20000000-7f0e21000000 r-xp 00000000 08:01 9012 /opt/cuda libs/libcublas.so.12
7f0e30000000-7f0e31000000 rw-p 00000000 00:00 0 [heap]
7f0e40000000-7f0e41000000 r-xp 00000000 08:01 3456 /venv/site-packages/nvidia/cudnn/lib/libcudnn.so.9
"""


def test_maps_parser_finds_cuda_libraries() -> None:
    found = parse_proc_maps(MAPS_SAMPLE)
    assert found["libcudart.so.12"].endswith("/nvidia/cuda_runtime/lib/libcudart.so.12")
    assert found["libcudnn.so.9"].endswith("/nvidia/cudnn/lib/libcudnn.so.9")
    # libc is not a CUDA library and must not be collected.
    assert not any(k.startswith("libc.so") for k in found)


def test_maps_parser_handles_a_path_containing_a_space() -> None:
    """A mapped path with a space was silently DROPPED, reporting {} not an error.

    /proc/self/maps is `address perms offset dev inode pathname` and the pathname
    may contain spaces, so taking the last whitespace-separated field loses it.
    An absent mapping reads as "nothing loaded", which is the reassuring
    direction.
    """
    found = parse_proc_maps(MAPS_SAMPLE)
    assert found["libcublas.so.12"] == "/opt/cuda libs/libcublas.so.12"


def test_maps_parser_ignores_anonymous_and_pseudo_mappings() -> None:
    assert parse_proc_maps("7f0e30000000-7f0e31000000 rw-p 00000000 00:00 0 [heap]\n") == {}
    assert parse_proc_maps("7f0e30000000-7f0e31000000 rw-p 00000000 00:00 0\n") == {}


def test_invalid_yaml_raises_env_check_error_not_a_parser_error(tmp_path: Path) -> None:
    """Callers catch EnvCheckError; a raw yaml.ParserError is undocumented here."""
    run_dir = _run_dir(tmp_path, "task:\n  seed: [unclosed\n")
    with pytest.raises(EnvCheckError, match="not valid YAML"):
        read_seed(run_dir)


def test_a_non_mapping_top_level_is_an_error_not_an_attribute_error(tmp_path: Path) -> None:
    """A valid top-level list previously raised AttributeError from .get()."""
    run_dir = _run_dir(tmp_path, "- one\n- two\n")
    with pytest.raises(EnvCheckError, match="not a mapping"):
        read_seed(run_dir)


def test_duplicate_seed_keys_are_refused_rather_than_reported_inconsistently(
    tmp_path: Path,
) -> None:
    """YAML keeps the LAST duplicate; the raw-line scan quotes the FIRST.

    So a duplicated key produced a value and a quoted line that DISAGREE, with no
    error. For a function whose whole purpose is to quote the file as evidence,
    contradictory evidence is worse than none.
    """
    run_dir = _run_dir(tmp_path, "task:\n  seed: 1\n  steps: 10\n  seed: 7\n")
    with pytest.raises(EnvCheckError, match="seed:. lines"):
        read_seed(run_dir)


@pytest.mark.parametrize(
    "text",
    [
        "task:\n  seed: 7\n",
        "task :\n  seed: 7\n",
        '"task":\n  seed: 7\n',
        "'task':\n  seed: 7\n",
        "task:\n  seed : 7\n",
        'task:\n  "seed": 7\n',
    ],
    ids=["plain", "task-spaced", "task-double-quoted", "task-single-quoted",
         "seed-spaced", "seed-quoted"],
)
def test_valid_yaml_spellings_of_the_same_key_are_not_refused(
    tmp_path: Path, text: str
) -> None:
    """YAML permits several spellings of one key; refusing them is a FALSE REFUSAL.

    Found by an independent verifier asked specifically to hunt refusals that fire
    on legitimate input. A false refusal is as bad as a missing check and harder
    to notice, because tests supply the shapes the author expects -- and every
    test here used the single spelling the reference config happens to use.
    """
    found = read_seed(_run_dir(tmp_path, text))
    assert found["seed"] == 7
    assert found["evidence_available"] is True
    assert found["line_number"] == 2


def test_a_yaml_merge_key_seed_is_reported_not_refused(tmp_path: Path) -> None:
    """A merge key resolves task.seed with no `seed:` line inside the task block.

    Named by an independent verifier alongside the spelling cases. It is valid
    YAML and a legitimate Hydra shape, so refusing it would reject a real config;
    the value is verified and the absent evidence is stated.
    """
    text = "defaults: &d\n  seed: 7\ntask:\n  <<: *d\n"
    found = read_seed(_run_dir(tmp_path, text))
    assert found["seed"] == 7
    assert found["evidence_available"] is False


def test_a_seed_with_no_quoteable_line_is_reported_not_refused(tmp_path: Path) -> None:
    """Inline flow style carries a real value and no line to quote.

    Refusing it rejected a valid config; returning a bare None was a silent gap.
    The value is verified and the missing evidence is stated explicitly.
    """
    found = read_seed(_run_dir(tmp_path, "task: {seed: 7, steps: 10}\n"))
    assert found["seed"] == 7
    assert found["evidence_available"] is False
    assert "not" in found["warning"] and "available" in found["warning"]


def test_a_normal_block_style_seed_still_yields_its_line(tmp_path: Path) -> None:
    """The negative control for the test above: the ordinary shape must pass."""
    found = read_seed(_run_dir(tmp_path))
    assert found["line_number"] == 50
    assert found["raw_line"] == "  seed: 1"
