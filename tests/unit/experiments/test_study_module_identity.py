"""Study modules must resolve their siblings to their OWN study directory.

Study directories under ``experiments/`` are not packages, so each one lands on
``sys.path`` -- explicitly, and implicitly whenever pytest collects a test from
it.  A sibling reached with a bare ``import plan`` is therefore cached under the
bare key ``"plan"``, and with three ``plan.py``, three ``collect.py``, four
``launch.py`` and two ``utils/`` packages in the tree, the first study loaded
owns those names for every study after it.

Three properties are pinned here, deliberately different in kind:

``test_no_bare_import_of_a_colliding_sibling``
    A STRUCTURAL rule over the source: no study module may reach a sibling by
    bare import when that name is supplied by more than one study.  This is the
    rule that converts a naming convention into something enforced -- notably
    for ``he-cutover``, which is safe today only because it happened to prefix
    its own modules ``cutover_plan``/``cutover_strata``.

``test_composed_session_resolves_each_study_to_itself``
    A BEHAVIOURAL measurement: load the studies together in one interpreter and
    check the file each module actually came from.  Static reasoning about an
    import graph is exactly what missed this defect the first time, so the
    structural rule is not trusted on its own.

``test_he_v1_boundary_has_one_canonical_module_identity``
    A DUPLICATE-IDENTITY check, which the other two cannot see.  Zero ambiguous
    bare keys and correct module identity are different properties: a study
    loaded partly scoped and partly bare has two copies of its own modules, two
    distinct exception classes, and an ``except`` clause that silently misses.
    An earlier revision of this fix satisfied both rules above while breaking
    this one.

WHAT THE RULE DOES NOT SAY, which matters as much as what it does.  It does NOT
forbid two studies from having same-named modules.  A study is free to own a
``plan.py`` like every other study; the rule constrains only the MECHANISM used
to reach it.  A guard phrased as "no two directories may share a basename" would
pass every assertion in this file and would fail the first time somebody tried
to add an ordinary study -- so both directions are mutation-tested below.
"""

from __future__ import annotations

import ast
import collections
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO_ROOT / "experiments"


# --------------------------------------------------------------------------
# The structural predicate, parameterized by root so it can be aimed at
# synthetic trees as well as at the real one.
# --------------------------------------------------------------------------
def _sys_path_eligible_dirs(root: Path) -> list[Path]:
    """Return directories under ``root`` that can land on ``sys.path``.

    A directory can land on ``sys.path`` exactly when it is NOT itself a package:
    that is the condition under which ``sys.path.insert(0, STUDY_DIR)`` and
    pytest's prepend import mode both make its contents top-level modules.
    """

    dirs = {p.parent for p in root.rglob("*.py") if "__pycache__" not in p.parts}
    return sorted(d for d in dirs if not (d / "__init__.py").exists())


def _provided_top_level_names(study_dir: Path) -> set[str]:
    """Return the top-level importable names ``study_dir`` supplies."""

    names = {p.stem for p in study_dir.glob("*.py")}
    names |= {
        s.name
        for s in study_dir.iterdir()
        if s.is_dir() and (s / "__init__.py").exists()
    }
    return names


def find_bare_colliding_sibling_imports(root: Path) -> list[str]:
    """Return one message per bare import of an ambiguous sibling under ``root``.

    Parameters
    ----------
    root : Path
        Tree to scan (``experiments/`` in the real check).

    Returns
    -------
    list of str
        ``"<path>:<line>: <name>"`` for every violation; empty when clean.
    """

    study_dirs = _sys_path_eligible_dirs(root)
    provides = {d: _provided_top_level_names(d) for d in study_dirs}

    owners: dict[str, set[Path]] = collections.defaultdict(set)
    for study_dir, names in provides.items():
        for name in names:
            owners[name].add(study_dir)
    colliding = {name for name, dirs in owners.items() if len(dirs) > 1}

    violations: list[str] = []
    for study_dir in study_dirs:
        for path in sorted(study_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # not ours to police
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                else:
                    continue
                for name in names:
                    if name in provides[study_dir] and name in colliding:
                        violations.append(f"{path}:{node.lineno}: {name}")
    return violations


def _sys_module_names(tree: ast.AST) -> set[str]:
    """Return names bound to the ``sys`` module and never rebound in this file.

    WHAT THIS RESOLVES: ``import sys`` / ``import sys as X`` followed by
    attribute access, where ``X`` is never reassigned or shadowed.

    KNOWN-UNCAUGHT, measured by independent review of PR #484 round 2 against
    the sibling ``sys.path`` rule and applying equally here: ``from sys import
    modules``, ``importlib.import_module("sys")``, an assignment alias
    (``s = sys``), and aliasing the mapping itself (``m = sys.modules``).
    Closing those needs scope-aware binding analysis, which is a static-analysis
    project rather than a guard; a guard that claims more than it delivers is
    worse than an honest narrow one. Tracked separately.

    Names that are REBOUND anywhere are dropped, because once ``s = Registry()``
    appears a later ``s.modules[...]`` cannot be attributed to ``sys`` by
    parsing -- and guessing yields a false positive, the failure mode that only
    shows up when it refuses somebody's legitimate code.
    """

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    aliases.add(alias.asname or "sys")

    rebound = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    rebound |= {
        arg.arg
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for arg in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        ]
    }
    return aliases - rebound


def find_bare_sys_modules_registrations(root: Path) -> list[str]:
    """Return one message per ``sys.modules[<plain name>] = ...`` under ``root``.

    An import scan cannot see this: publishing a module into the shared slot is
    an ASSIGNMENT, not an import.  Several study tests did exactly that --

        sys.modules[spec.name] = module   # unique key, fine
        sys.modules[name] = module        # BARE key, re-creates the collision

    -- behind a ``bind_direct`` flag whose only purpose was to make a loaded
    module's bare sibling imports resolve.  With siblings loaded study-scoped
    the flag is unnecessary, and leaving it in would have handed the shared
    slot back after the imports were fixed.

    The rule resolves the MAPPING'S OWNER before rejecting.  ``.modules`` is not
    a reserved word: a perfectly ordinary ``Registry`` or plugin object can have
    a ``modules`` dict, and ``self.modules[name] = module`` is unremarkable code
    with nothing to do with the import system.  Flagging by attribute name alone
    would refuse a legitimate study -- an over-restriction that passes every
    test in this file and only surfaces when a colleague's study will not load.
    So only a subscript of ``<name bound to the sys module>.modules`` counts,
    where the binding is established from this file's own imports.

    Limitation, stated rather than hidden: a subscript that is a plain name is
    indistinguishable at parse time from one holding an already-unique key, so
    this rule asks for ``spec.name`` or an explicitly-built unique key. That is
    a test-authoring convention, not a constraint on study layout.
    """

    violations: list[str] = []
    for study_dir in _sys_path_eligible_dirs(root):
        for path in sorted(study_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue

            sys_aliases = _sys_module_names(tree)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "modules"
                        and isinstance(target.slice, ast.Name)
                    ):
                        continue
                    owner = target.value.value
                    if isinstance(owner, ast.Name) and owner.id in sys_aliases:
                        violations.append(
                            f"{path}:{node.lineno}: {owner.id}.modules[{target.slice.id}]"
                        )
    return violations


# --------------------------------------------------------------------------
# The rule, against the real tree.
# --------------------------------------------------------------------------
def test_no_module_is_published_under_a_bare_sys_modules_key() -> None:
    """No study file hands a module back to the shared bare slot."""

    violations = find_bare_sys_modules_registrations(EXPERIMENTS)
    assert violations == [], (
        "These register a module under a bare key, re-creating the shared slot "
        "that makes resolution order-dependent:\n  "
        + "\n  ".join(violations)
        + "\n\nRegister under spec.name (or another explicitly unique key) only."
    )


def test_sys_modules_rule_fires_on_a_bare_key(tmp_path: Path) -> None:
    """OVER-PERMISSIVE mutant: a bare-key registration must be caught."""

    _write_study(tmp_path / "study_a", "import sys\nsys.modules[name] = module\n")
    _write_study(tmp_path / "study_b", "VALUE = 1\n")

    violations = find_bare_sys_modules_registrations(tmp_path)

    assert len(violations) == 1, violations
    assert violations[0].endswith("sys.modules[name]")


def test_sys_modules_rule_allows_a_unique_key(tmp_path: Path) -> None:
    """OVER-RESTRICTIVE mutant: registering under ``spec.name`` must stay green."""

    _write_study(tmp_path / "study_a", "import sys\nsys.modules[spec.name] = module\n")
    _write_study(tmp_path / "study_b", "VALUE = 1\n")

    assert find_bare_sys_modules_registrations(tmp_path) == []



def test_no_bare_import_of_a_colliding_sibling() -> None:
    """No study reaches an ambiguous sibling by bare import."""

    violations = find_bare_colliding_sibling_imports(EXPERIMENTS)
    assert violations == [], (
        "These modules reach a sibling whose name is supplied by more than one "
        "study, so which module they get depends on collection order:\n  "
        + "\n  ".join(violations)
        + "\n\nUse experiments/toolkit/study_imports.sibling(__file__, ...) instead."
    )


# --------------------------------------------------------------------------
# Mutation in BOTH directions.  The rule must fire on a real ambiguity and must
# stay silent on a legitimate study layout.
# --------------------------------------------------------------------------
def _write_study(study_dir: Path, body: str, *, extra: dict[str, str] | None = None) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "plan.py").write_text("VALUE = 1\n", encoding="utf-8")
    (study_dir / "collect.py").write_text(body, encoding="utf-8")
    for name, text in (extra or {}).items():
        (study_dir / name).write_text(text, encoding="utf-8")


def test_rule_fires_when_a_shared_name_is_bare_imported(tmp_path: Path) -> None:
    """OVER-PERMISSIVE mutant: two studies share ``plan`` and import it bare."""

    _write_study(tmp_path / "study_a", "import plan\n")
    _write_study(tmp_path / "study_b", "import plan\n")

    violations = find_bare_colliding_sibling_imports(tmp_path)

    assert len(violations) == 2, violations
    assert all(v.endswith(": plan") for v in violations)


def test_rule_allows_two_studies_to_own_the_same_module_name(tmp_path: Path) -> None:
    """OVER-RESTRICTIVE mutant: same names, reached correctly -- must stay green.

    This is the direction a naive guard gets wrong.  A rule that forbade
    duplicate basenames would flag this layout, and nobody could add an ordinary
    study.  Sharing the name is fine; reaching it by bare import is not.
    """

    loader = "from experiments.toolkit.study_imports import sibling\nplan = sibling(__file__, 'plan')\n"
    _write_study(tmp_path / "study_a", loader)
    _write_study(tmp_path / "study_b", loader)

    assert find_bare_colliding_sibling_imports(tmp_path) == []


def test_rule_allows_a_bare_import_of_an_unambiguous_sibling(tmp_path: Path) -> None:
    """A name only one study supplies is not ambiguous, so bare import is fine."""

    _write_study(
        tmp_path / "study_a",
        "import only_here\n",
        extra={"only_here.py": "VALUE = 2\n"},
    )
    _write_study(tmp_path / "study_b", "VALUE = 3\n")

    assert find_bare_colliding_sibling_imports(tmp_path) == []


# --------------------------------------------------------------------------
# Behavioural: a COMPOSED session, measured rather than reasoned about.
#
# Run in a subprocess so the composition is built here explicitly and does not
# depend on -- or leak into -- whatever else the surrounding suite imported.
# Running these modules in-process would also be the one condition under which
# the defect cannot appear, which is how it stayed hidden before.
# --------------------------------------------------------------------------
_COMPOSED_PROBE = """
import json, pathlib, sys, types
sys.path.insert(0, {repo!r})
from experiments.toolkit.study_imports import load_study_module

REPO = pathlib.Path({repo!r})
HOOKE = REPO / "experiments" / "hooke"
PAIRS = [HOOKE / "pair_stability_v3", HOOKE / "tpen-pair-scan-v1"]

# Reproduce the condition a composed pytest session creates: every study
# directory on sys.path at once.  Without this the probe would exercise the one
# situation in which the defect cannot occur (a single study in isolation),
# which is exactly how it stayed hidden through two reviews.
for _study in PAIRS:
    sys.path.insert(0, str(_study))

out = {{}}
# Load the SAME sibling names from BOTH studies, in this order and reversed.
for order in ("forward", "reverse"):
    studies = PAIRS if order == "forward" else list(reversed(PAIRS))
    seen = {{}}
    for study in studies:
        for name in ("utils.layout", "stats", "launch", "plot"):
            mod = load_study_module(study, name)
            seen[f"{{study.name}}::{{name}}"] = mod.__file__
    out[order] = seen

# THE SILENT CASE.  Resolving the top-level load correctly is not enough: a
# study module binds names from its OWN siblings while executing, and it is
# those bindings that a shared bare key corrupts.  So inspect what each loaded
# module actually holds and ask which study each object was defined in.
STUDY_DIRS = {{p.name for p in PAIRS}}


def _defining_file(value):
    if isinstance(value, types.ModuleType):
        return getattr(value, "__file__", None)
    if isinstance(value, type):
        owner = sys.modules.get(getattr(value, "__module__", ""), None)
        return getattr(owner, "__file__", None)
    globals_ = getattr(value, "__globals__", None)
    if isinstance(globals_, dict):
        return globals_.get("__file__")
    return None


def _study_of(path_str):
    for part in pathlib.Path(path_str).parts:
        if part in STUDY_DIRS:
            return part
    return None


foreign = []
for study in PAIRS:
    for name in ("launch", "plot", "collect"):
        mod = load_study_module(study, name)
        for attr, value in vars(mod).items():
            if attr.startswith("__"):
                continue
            defined_in = _defining_file(value)
            if not defined_in:
                continue
            other = _study_of(defined_in)
            if other is not None and other != study.name:
                foreign.append(f"{{study.name}}.{{name}}.{{attr}} defined in {{other}}")
out["foreign_bindings"] = sorted(set(foreign))

# The cross-study boundary: he-cutover reaching into he-v1.
sys.path.insert(0, str(REPO / "experiments" / "atomistic" / "he-cutover"))
import pipeline  # noqa: F401  -- he-cutover's real entry module
import hev1
out["cutover_plan_file"] = hev1.plan_stage.__file__
out["cutover_own_plan_exists"] = (REPO / "experiments/atomistic/he-cutover/plan.py").exists()

# Bare top-level keys that a colliding study name should never occupy.
out["bare_keys"] = sorted(
    k for k in sys.modules
    if k in {{"plan", "collect", "launch", "train", "utils", "stats", "plot"}}
)
print("JSON_START" + json.dumps(out) + "JSON_END")
"""


@pytest.fixture(scope="module")
def composed_probe() -> dict:
    """Run the composed-session probe once and return its measurements."""

    script = _COMPOSED_PROBE.format(repo=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    raw = result.stdout.split("JSON_START")[1].split("JSON_END")[0]
    import json

    return json.loads(raw)


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_composed_session_resolves_each_study_to_itself(
    composed_probe: dict, order: str
) -> None:
    """Each study's sibling comes from that study, in either load order."""

    for key, resolved in composed_probe[order].items():
        study_name, _, module_name = key.partition("::")
        assert f"/{study_name}/" in resolved, (
            f"in {order} order, {study_name} asked for {module_name} and received "
            f"{resolved} -- a different study's module, with no exception raised"
        )


def test_load_order_does_not_change_resolution(composed_probe: dict) -> None:
    """Reversing the load order changes nothing -- the defect's signature."""

    assert composed_probe["forward"] == composed_probe["reverse"]


def test_cross_study_boundary_reaches_the_intended_study(composed_probe: dict) -> None:
    """he-cutover's boundary yields He-v1's ``plan``, not its own or another's."""

    assert "/he-v1/" in composed_probe["cutover_plan_file"]


def test_no_colliding_study_name_occupies_a_bare_module_key(
    composed_probe: dict,
) -> None:
    """No ambiguous study module is cached under its bare top-level name.

    A bare key is the shared slot the whole defect runs through: whoever fills
    it first supplies every study afterwards.
    """

    assert composed_probe["bare_keys"] == []


def test_no_module_holds_an_object_defined_in_another_study(
    composed_probe: dict,
) -> None:
    """No study module binds a name that another study defined.

    This is the silent case, and the one the other assertions cannot see.
    Loading a study module by path under a unique key resolves the TOP-LEVEL
    import correctly even when the defect is present -- what it cannot fix is
    the bare imports the loaded module performs while executing.  Those
    bindings are where the wrong study's constants, paths and helpers actually
    arrive, and they arrive without raising anything.

    Concretely: the two hooke studies' ``launch.py`` differ only in an embedded
    config path, so a swap here means a study reading another study's
    ``configs/smoke.yaml`` with nothing to indicate it.
    """

    assert composed_probe["foreign_bindings"] == [], (
        "these names came from a different study than the module holding them:\n  "
        + "\n  ".join(composed_probe["foreign_bindings"])
    )


def test_sys_modules_rule_ignores_an_unrelated_modules_mapping(tmp_path: Path) -> None:
    """OVER-RESTRICTIVE mutant: ``self.modules[name] = module`` must stay green.

    ``.modules`` is not reserved.  A registry object with a ``modules`` dict is
    ordinary code and has nothing to do with the import system.  A rule that
    matched on the attribute name alone would refuse a legitimate study, and
    would do so invisibly -- passing every other assertion here and failing
    only when somebody's study could not load.
    """

    registry = (
        "class Registry:\n"
        "    def __init__(self):\n"
        "        self.modules = {}\n"
        "    def add(self, name, module):\n"
        "        self.modules[name] = module\n"
    )
    _write_study(tmp_path / "study_a", registry)
    _write_study(tmp_path / "study_b", "VALUE = 1\n")

    assert find_bare_sys_modules_registrations(tmp_path) == []


def test_study_slug_is_injective_across_hyphen_and_underscore() -> None:
    """Sanitizing is not enough: the slug must not merge two real studies.

    ``new-study`` and ``new_study`` both sanitize to ``new_study``.  Under a
    merged key the second study's ``plan.py`` silently returns the FIRST
    study's cached module -- the wrong-module-without-an-exception failure this
    whole module exists to remove, reintroduced by the fix for it.
    """

    from experiments.toolkit.study_imports import study_slug

    assert study_slug(EXPERIMENTS / "new-study") != study_slug(EXPERIMENTS / "new_study")
    assert study_slug(EXPERIMENTS / "a" / "study") != study_slug(EXPERIMENTS / "b" / "study")


def test_distinct_studies_whose_names_differ_only_by_separator(tmp_path: Path) -> None:
    """End-to-end form of the same property, measured rather than reasoned.

    Two studies whose directory names differ only by ``-`` versus ``_`` must
    load their own ``plan.py``, not one another's.
    """

    for name, value in (("new-study", "HYPHEN"), ("new_study", "UNDERSCORE")):
        study = tmp_path / name
        study.mkdir()
        (study / "plan.py").write_text(f"WHICH = {value!r}\n", encoding="utf-8")

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from experiments.toolkit.study_imports import load_study_module
        import pathlib
        root = pathlib.Path({str(tmp_path)!r})
        a = load_study_module(root / "new-study", "plan")
        b = load_study_module(root / "new_study", "plan")
        print(a.WHICH, b.WHICH)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["HYPHEN", "UNDERSCORE"], result.stdout


# --------------------------------------------------------------------------
# The He-v1 boundary.  Adopted from the independent reviewer's boundary_probe.py
# rather than rewritten, because it caught a regression this lane's own
# instrument did not: the first fix converted only COLLIDING sibling names, so
# `hev1.py` received study-scoped modules while He-v1's own bare imports built a
# SECOND copy of the same files.  Zero ambiguous bare keys and correct module
# identity are different properties, and only the first was being measured.
# --------------------------------------------------------------------------
_BOUNDARY_PROBE = """
import sys
sys.path.insert(0, {repo!r})
sys.path.insert(0, {cutover!r})
import hev1
print('IDENTITIES', hev1.canary is hev1.eval_stage.canary,
      hev1.strata is hev1.plan_stage.strata,
      hev1.layout is hev1.plan_stage.layout)
print('STRATA_STATE_IS_SHARED', hev1.strata.STRATA is hev1.plan_stage.strata.STRATA)
try:
    try:
        hev1.plan_stage.strata.stratum('review-not-a-stratum')
    except hev1.strata.StratumError:
        print('EXPECTED_ERROR_CAUGHT')
except ValueError as exc:
    print('EXPECTED_ERROR_MISSED', type(exc).__module__, type(exc).__name__)
"""


def test_he_v1_boundary_has_one_canonical_module_identity() -> None:
    """He-cutover's view of He-v1 is the SAME module object He-v1 uses itself.

    Duplicate identity is a wrong-module defect just as a shared bare key is.
    Its signature is worse in one way: two copies of ``strata`` mean two
    distinct ``StratumError`` classes, so ``except hev1.strata.StratumError``
    silently fails to catch an error raised through ``hev1.plan_stage.strata``
    and the exception escapes as an unrelated type.
    """

    script = _BOUNDARY_PROBE.format(
        repo=str(REPO_ROOT),
        cutover=str(EXPERIMENTS / "atomistic" / "he-cutover"),
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    assert "IDENTITIES True True True" in result.stdout, result.stdout
    assert "STRATA_STATE_IS_SHARED True" in result.stdout, result.stdout
    assert "EXPECTED_ERROR_CAUGHT" in result.stdout, (
        "a StratumError raised through one copy of he-v1's strata was not caught "
        "by `except` against the other copy:\n" + result.stdout
    )


def test_sys_modules_rule_does_not_accuse_a_rebound_alias(tmp_path: Path) -> None:
    """FALSE POSITIVE, measured by round-2 review: rebound name is not ``sys``."""

    _write_study(
        tmp_path / "study_a",
        "import sys as s\n"
        "class Reg:\n    def __init__(self):\n        self.modules = {}\n"
        "s = Reg()\n"
        "name = 'x'\nmodule = None\n"
        "s.modules[name] = module\n",
    )
    _write_study(tmp_path / "study_b", "VALUE = 1\n")

    assert find_bare_sys_modules_registrations(tmp_path) == []


# --------------------------------------------------------------------------
# Cross-checkout collision.  Adopted from the round-2 reviewer's
# cross_checkout_probe.py rather than rewritten -- it found a regression this
# lane introduced while fixing the previous one.
#
# The cache key is the study's path RELATIVE to experiments/, so two checkouts
# of this repository produce the same key for the same study.  Before the cache
# was validated against its requested source, a caller in checkout B silently
# received checkout A's module: the defect this file exists to prevent, one
# level up from studies to checkouts.
#
# The remedy here is deliberately the cheap half.  Full canonical identity
# across checkouts means putting provenance in the key, which is a design
# change and is filed separately.  Validation converts a SILENT wrong module
# into a LOUD error, which is the transformation this slice is for.
# --------------------------------------------------------------------------
_CROSS_CHECKOUT_PROBE = """
import importlib.util, pathlib, sys
A = pathlib.Path({repo!r})
B = pathlib.Path({other!r})

def direct(root, name):
    path = root / "experiments" / "atomistic" / "he-v1" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_probe_" + str(len(sys.modules)), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, str(A))
from experiments.toolkit import study_imports

# Checkout A's strata occupies the shared, checkout-relative key.
study_imports.load_study_module(A / "experiments" / "atomistic" / "he-v1", "strata")
try:
    plan_b = direct(B, "plan")
except ImportError as exc:
    print("RAISED_IMPORTERROR")
    print("MENTIONS_BOTH", "strata.py" in str(exc) and str(B) in str(exc))
else:
    print("RETURNED_SILENTLY", plan_b.strata.__file__)
"""


def test_a_foreign_checkout_raises_instead_of_returning_the_wrong_module(
    tmp_path: Path,
) -> None:
    """A second checkout must get a loud error, never another checkout's module."""

    other = tmp_path / "checkout-b"
    for sub in (Path("experiments") / "toolkit", Path("experiments") / "atomistic" / "he-v1"):
        shutil.copytree(
            REPO_ROOT / sub,
            other / sub,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    script = _CROSS_CHECKOUT_PROBE.format(repo=str(REPO_ROOT), other=str(other))
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "RAISED_IMPORTERROR" in result.stdout, (
        "checkout B silently received a module from checkout A:\n" + result.stdout
    )
    assert "MENTIONS_BOTH True" in result.stdout, (
        "the error must name the bound file and the requested one, or it cannot "
        "be diagnosed:\n" + result.stdout
    )
