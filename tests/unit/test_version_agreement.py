"""The four recorded version sources must agree with each other.

``7693ead`` ("Release v0.3.0") moved ``pyproject.toml``, ``tpen/__init__.py`` and
the ``uv.lock`` root entry but left ``README.md`` naming the previous release.
Nothing executable asserted the sources agreed, so nothing failed; the drift was
found later and patched out of band by the unrelated commit ``e41a044``. The
canonical bump before it, ``0684d07`` (v0.2.3), had touched all four. This module
is the assertion that was missing.

``uv.lock`` is in scope for a reason that is not cosmetic: the cluster
environment is built with ``uv sync --extra cpu --locked``, which fails closed on
a stale lock. A bump that moves ``pyproject.toml`` and leaves the lock's ``tpen``
entry behind does not merely print a wrong version -- no scheduler job can start
at all.

No expected version is written anywhere in this module, deliberately. Hardcoding
one would create a fifth literal with its own drift, which is the defect rather
than the fix. The invariant under test is AGREEMENT, not any particular value.
"""

from __future__ import annotations

import re
from pathlib import Path

import tpen

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``The current Hooke integration release is `v0.3.1`; release notes live in``
README_RELEASE_PATTERN = re.compile(
    r"^The current Hooke integration release is `v(?P<version>[^`]+)`;",
    re.MULTILINE,
)


def _sole_match(pattern: re.Pattern[str], text: str, source: str) -> str:
    """Return the single capture of ``pattern`` in ``text`` or fail loudly.

    A second match means the file grew another version literal that this test
    would otherwise read past, which is the failure mode the module exists to
    prevent -- so ambiguity is an error, not a first-match-wins situation.
    """

    matches = pattern.findall(text)
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one version literal in {source}, found {len(matches)}")
    return matches[0]


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_table = re.search(r"^\[project\]$(?P<body>.*?)^\[", text, re.MULTILINE | re.DOTALL)
    if project_table is None:
        raise AssertionError("pyproject.toml has no [project] table")
    return _sole_match(
        re.compile(r'^version = "([^"]+)"$', re.MULTILINE),
        project_table.group("body"),
        "pyproject.toml [project]",
    )


def _uv_lock_version() -> str:
    text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    blocks = [block for block in text.split("[[package]]") if re.search(r'^name = "tpen"$', block, re.MULTILINE)]
    if len(blocks) != 1:
        raise AssertionError(f"expected exactly one uv.lock package named 'tpen', found {len(blocks)}")
    return _sole_match(
        re.compile(r'^version = "([^"]+)"$', re.MULTILINE),
        blocks[0],
        "uv.lock [[package]] tpen",
    )


def _readme_version() -> str:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return _sole_match(README_RELEASE_PATTERN, text, "README.md versioning section")


def test_all_four_version_sources_agree() -> None:
    """Package metadata, runtime metadata, the lock entry and the README agree.

    Reported together rather than as four separate asserts: when they disagree,
    the useful diagnostic is which one is the outlier, and a single assert that
    prints all four gives that immediately.
    """

    sources = {
        "pyproject.toml [project] version": _pyproject_version(),
        "tpen.__version__": tpen.__version__,
        "uv.lock tpen entry": _uv_lock_version(),
        "README.md current release": _readme_version(),
    }

    assert len(set(sources.values())) == 1, f"version sources disagree: {sources}"
