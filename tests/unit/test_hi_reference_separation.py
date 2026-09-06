"""Train and evaluation hold the reference apart, and the boundary is checked.

The consolidated authority requires separate train and evaluation
manifests/processes, with only the evaluation side resolving the literature
value, and names "import tests" as one of the mechanisms that must enforce it.
This module is that import test, plus the two ends it separates.

A separation that is only documented is a comment. What makes it real is that
no module on the training path can reach ``tpen.hi_manifest`` -- so a reference
is not merely unused by a training process, it is unreachable from one.

The corpus is a SWEEP, not a list. It began as three named entry points, which
answered "do the modules we thought of import the reference"; the question that
matters is "can any module on the training path reach it", and those differ by
exactly the module nobody remembered to add. Every module under
``tpen/training/`` and ``tpen/callback/`` is therefore included, and
``TestTheSweptCorpusIsReal`` guards the sweep itself -- an empty or shrunken
glob would otherwise make this file weaker than the hardcoded list it replaced
while still reporting green.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.hi_manifest import (
    HI_EVALUATION_SCHEMA,
    load_evaluation_manifest,
    reference_energy,
)
from tpen.hi_schema import HI_TRAIN_SCHEMA, validate_hi_train_config

# Anchored to this file rather than to the process working directory. A bare
# relative path silently pins the whole module to being run from the repository
# root: from anywhere else it reads a DIFFERENT tree, or no tree at all, and a
# sweep that finds nothing passes. That matters most where it is least visible
# -- a cluster job whose working directory is the submission directory rather
# than the checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]

CONTROL_CONFIG = _REPO_ROOT / "experiments/atomistic/he-importance/configs/train.yaml"
EVALUATION_MANIFEST = _REPO_ROOT / "experiments/atomistic/he-importance/manifests/evaluation.yaml"

# Individually named modules on the training path. These predate the package
# sweep below and are kept explicit: they are the entry points the reference
# would most plausibly be reached through, and naming them means the guard
# still covers them if a package is ever renamed out from under the sweep.
NAMED_TRAIN_PATH_MODULES = ("tpen/hi_schema.py", "tpen/run.py", "tpen/config_schema.py")

# Whole packages on the training path. Every module under these is swept,
# because the hazard is a module nobody thought to list -- the same reasoning
# that makes the admitted-callback set in ``tpen.hi_schema`` an allowlist.
TRAIN_PATH_PACKAGES = ("tpen/training", "tpen/callback")


def _swept_modules() -> tuple[str, ...]:
    """Return every repo-relative training-path module, named plus swept."""

    paths = {_REPO_ROOT / relative for relative in NAMED_TRAIN_PATH_MODULES}
    for package in TRAIN_PATH_PACKAGES:
        paths.update((_REPO_ROOT / package).rglob("*.py"))
    return tuple(sorted(str(path.relative_to(_REPO_ROOT)) for path in paths))


# Modules that are on the training path and must not reach the reference.
TRAIN_PATH_MODULES = _swept_modules()

REFERENCE_MODULE = "tpen.hi_manifest"


def _package_of(path: Path) -> str:
    """Return the dotted package a source file lives in, for relative imports.

    ``tpen/training/trainer.py`` and ``tpen/training/__init__.py`` both resolve
    to ``tpen.training``: a level-1 relative import means the containing
    package in both cases.
    """

    relative = path.resolve().relative_to(_REPO_ROOT)
    return ".".join(relative.parts[:-1])


def _resolve_import_from(node: ast.ImportFrom, package: str) -> str | None:
    """Return the absolute module a ``from ... import`` names, or None.

    ``node.level`` is the number of leading dots. Level 1 is the containing
    package, level 2 its parent, and so on.
    """

    if node.level == 0:
        return node.module
    parts = package.split(".") if package else []
    if node.level - 1 > len(parts):
        return None
    base = ".".join(parts[: len(parts) - (node.level - 1)])
    if node.module:
        return f"{base}.{node.module}" if base else node.module
    return base or None


def _imported_modules(path: Path) -> set[str]:
    """Return every module name a source file imports, statically.

    Parsed with ``ast`` rather than by importing, so the answer describes the
    file rather than whatever happens to be in ``sys.modules`` from an earlier
    test.

    **This helper previously recognised two import forms out of seven**, and
    the sweep built on it therefore certified a reachability claim it could not
    support. It matched ``import a.b`` and ``from a.b import c``, and MISSED
    ``from a import b``, its aliased spelling, and every relative form. The
    round-1 lane review measured it at 2/7 on isolated per-form fixtures.

    Two things are handled now that were not:

    - ``from PKG import NAME`` may import a MODULE rather than an attribute, so
      ``PKG.NAME`` is recorded alongside ``PKG``. Recording only ``node.module``
      is what lost ``from tpen import hi_manifest``.
    - RELATIVE imports are resolved against the file's own package, so
      ``from . import x`` and ``from ..y import z`` produce absolute names.
      Previously they produced ``None`` or a bare leaf and vanished.

    Recording ``PKG.NAME`` for an attribute import adds names that are not
    modules, e.g. ``tpen.hi_manifest.reference_energy``. That is harmless for a
    membership test and cannot manufacture a false positive for
    `REFERENCE_MODULE`: the only way to synthesise ``tpen.hi_manifest`` is
    ``from tpen import hi_manifest``, which is a genuine import of it.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _package_of(path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from(node, package)
            if base is None:
                continue
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


class TestTheSweptCorpusIsReal:
    """A sweep that matched nothing would pass every test below it.

    This is the failure the three hardcoded module names could not have: a
    glob is only as strong as what it finds, and ``rglob`` over a mistyped,
    moved or renamed package returns an empty set silently. Replacing an
    explicit list with a sweep is a REGRESSION unless the sweep is shown to
    have found more than the list did.

    Reported as a census rather than a bare assertion: the counts are printed
    into the failure message so a shrinking corpus is diagnosable without
    re-running under a debugger.
    """

    def test_every_named_module_survived_the_sweep(self) -> None:
        missing = sorted(set(NAMED_TRAIN_PATH_MODULES) - set(TRAIN_PATH_MODULES))
        assert not missing, f"named modules dropped out of the corpus: {missing}"

    @pytest.mark.parametrize("package", TRAIN_PATH_PACKAGES)
    def test_each_package_contributed_modules(self, package: str) -> None:
        """Per-package, so one empty package cannot hide behind a full one."""

        assert (_REPO_ROOT / package).is_dir(), f"{package} is not a directory under {_REPO_ROOT}"
        contributed = [path for path in TRAIN_PATH_MODULES if path.startswith(f"{package}/")]
        assert contributed, (
            f"{package} contributed no modules to the sweep; the corpus is "
            f"{len(TRAIN_PATH_MODULES)} modules and the guard is weaker than "
            "the hardcoded list it replaced"
        )

    def test_the_sweep_is_strictly_larger_than_the_named_list(self) -> None:
        assert len(TRAIN_PATH_MODULES) > len(NAMED_TRAIN_PATH_MODULES), (
            f"sweep found {len(TRAIN_PATH_MODULES)} modules against "
            f"{len(NAMED_TRAIN_PATH_MODULES)} named ones; the package sweep "
            "added nothing"
        )

    def test_known_members_are_present(self) -> None:
        """Anchor on files that exist today, so a silent relocation is caught.

        Named individually rather than counted: a count floor is satisfied by
        any collection of the right size, including one assembled from the
        wrong directory.
        """

        for expected in ("tpen/training/trainer.py", "tpen/callback/base.py"):
            assert expected in TRAIN_PATH_MODULES, (
                f"{expected} is missing from the swept corpus; either it moved "
                "and TRAIN_PATH_PACKAGES is now wrong, or the sweep is reading "
                "the wrong tree"
            )


class TestImportSeparation:
    @pytest.mark.parametrize("module_path", TRAIN_PATH_MODULES)
    def test_no_training_module_imports_the_reference_holder(self, module_path: str) -> None:
        imported = _imported_modules(_REPO_ROOT / module_path)
        assert REFERENCE_MODULE not in imported, (
            f"{module_path} imports {REFERENCE_MODULE}; the reference must be unreachable "
            "from the training path, not merely unused by it"
        )

    def test_the_parser_would_notice_an_import(self) -> None:
        """Positive control: prove the instrument can see what it looks for.

        Without this, "no training module imports it" would pass equally well
        for a parser that returned an empty set -- which is exactly what a
        typo in the module name would produce.

        NOT SUFFICIENT ON ITS OWN, and the reason is the lesson of this file.
        This control uses the ONE spelling this test module happens to use, and
        that spelling was one the parser already handled. So it passed for
        years while certifying an instrument blind to five of seven forms.
        `TestTheParserSeesEveryImportForm` is the control that can actually
        fail; this one is kept because it exercises the real file rather than a
        synthetic fixture.
        """

        assert REFERENCE_MODULE in _imported_modules(Path(__file__))


# Every syntactic way to import `tpen.hi_manifest`, each with the package the
# importing file would live in. Relative forms are meaningless without that
# context, which is why the fixture carries it rather than assuming a default.
#
# THE SET IS DRAWN FROM WHAT THE INSTRUMENT MIGHT MISS, NOT FROM WHAT IT
# HANDLES. That distinction is the whole finding: the previous control used a
# spelling the parser already understood, so it could not have failed. A
# control drawn from the handled set cannot detect an unhandled set.
_IMPORT_FORMS: tuple[tuple[str, str, str], ...] = (
    ("absolute-module", "tpen/training", "import tpen.hi_manifest\n"),
    ("absolute-from", "tpen/training", "from tpen.hi_manifest import reference_energy\n"),
    ("absolute-module-alias", "tpen/training", "import tpen.hi_manifest as manifest\n"),
    ("package-from", "tpen/training", "from tpen import hi_manifest\n"),
    ("package-from-alias", "tpen/training", "from tpen import hi_manifest as manifest\n"),
    ("relative-parent-package", "tpen/training", "from .. import hi_manifest\n"),
    ("relative-parent-module", "tpen/training", "from ..hi_manifest import reference_energy\n"),
    ("relative-same-package", "tpen", "from . import hi_manifest\n"),
    ("relative-same-module", "tpen", "from .hi_manifest import reference_energy\n"),
    ("package-init-relative", "tpen/callback", "from ..hi_manifest import reference_energy\n"),
)


class TestTheParserSeesEveryImportForm:
    """The control that can fail, drawn from the space the instrument may miss.

    Measured on the previous parser: 2 of 7 forms detected. `import a.b` and
    `from a.b import c` were caught; `from a import b`, its aliased spelling,
    and every relative spelling were missed -- a relative import produced no
    usable name at all. The sweep over 38 modules was therefore reporting on a
    fraction of the ways a module can be reached.

    That is worth stating as a general shape rather than as one bug: L1f
    widened the CORPUS from 3 modules to 38 and added a guard for the widening,
    while the parser's FORM coverage went unguarded and unmeasured. A
    completeness claim with two dimensions needs a control on each.
    """

    @pytest.mark.parametrize(
        ("label", "package", "source"),
        _IMPORT_FORMS,
        ids=[form[0] for form in _IMPORT_FORMS],
    )
    def test_each_form_is_detected(
        self, label: str, package: str, source: str, tmp_path
    ) -> None:
        """Each form in isolation, so a failure names one spelling.

        One fixture per form on purpose. Putting several forms in one file and
        reading the union cannot attribute which spelling produced which name,
        so a parser handling only one of them would still look complete.
        """

        module_dir = _REPO_ROOT / package
        assert module_dir.is_dir(), f"fixture package {package} does not exist"
        target = module_dir / f"_import_form_fixture_{label.replace('-', '_')}.py"
        target.write_text(source, encoding="utf-8")
        try:
            detected = _imported_modules(target)
        finally:
            target.unlink()
        assert REFERENCE_MODULE in detected, (
            f"{label}: {source.strip()!r} in package {package} was not detected; "
            f"parser saw {sorted(detected)}"
        )

    def test_an_unrelated_import_is_not_detected(self, tmp_path) -> None:
        """Negative control, so 'detected' is not simply 'always true'.

        A sibling module with a similar name must NOT satisfy the membership
        test. Without this, a parser that returned the reference for every file
        would pass every case above.
        """

        target = _REPO_ROOT / "tpen/training/_import_form_fixture_negative.py"
        target.write_text("from tpen import hi_schema\nfrom . import optim\n", encoding="utf-8")
        try:
            detected = _imported_modules(target)
        finally:
            target.unlink()
        assert REFERENCE_MODULE not in detected
        assert "tpen.hi_schema" in detected, "the parser should still see the real import"


class TestEveryHIConfigDeclaresTheSchema:
    """Second, independent net against a config that forgets the marker.

    The run-time family rule in ``tpen.hi_schema`` catches a config that says
    it is helium-importance but omits ``schema:``. It is the net that covers
    configs L2 will GENERATE, which never live in the repository and which no
    directory scan can see.

    This is the other net: a static scan of the HI config directory, which
    catches a repo config that omitted the marker AND the experiment name and
    would therefore slip past the run-time rule. Neither net covers the other's
    population, which is why both exist.
    """

    HI_CONFIG_DIR = _REPO_ROOT / "experiments/atomistic/he-importance/configs"

    def test_the_directory_is_not_empty(self) -> None:
        """A scan over zero files passes vacuously and protects nothing."""

        assert list(self.HI_CONFIG_DIR.glob("*.yaml"))

    def test_every_config_declares_the_hi_train_schema(self) -> None:
        for path in sorted(self.HI_CONFIG_DIR.glob("*.yaml")):
            cfg = OmegaConf.load(path)
            declared = OmegaConf.select(cfg, "schema", default=None)
            assert declared == HI_TRAIN_SCHEMA, (
                f"{path} declares schema {declared!r}; every config in the "
                "helium-importance train family must declare "
                f"{HI_TRAIN_SCHEMA!r}, or it silently receives no enforcement"
            )

    def test_every_config_actually_passes_the_firewall(self) -> None:
        """Declaring the schema is not the same as satisfying it."""

        for path in sorted(self.HI_CONFIG_DIR.glob("*.yaml")):
            validate_hi_train_config(OmegaConf.load(path), env={})


class TestTheTrainConfigHoldsNoReference:
    def test_the_control_config_passes_the_firewall(self) -> None:
        cfg = OmegaConf.load(CONTROL_CONFIG)
        validate_hi_train_config(cfg, env={})

    def test_the_control_config_contains_no_reference_value(self) -> None:
        """Read as text, so an interpolation cannot hide the literal."""

        text = CONTROL_CONFIG.read_text(encoding="utf-8")
        assert "-2.903724377034119598" not in text

    def test_adding_a_reference_to_the_control_config_is_refused(self) -> None:
        """Red arm. Without it, the passing test above proves only that the
        firewall is quiet, not that it is watching this file."""

        from tpen.config_schema import ClosedSchemaError

        cfg = OmegaConf.load(CONTROL_CONFIG)
        cfg.system.reference_energy = -2.903724377034119598
        with pytest.raises(ClosedSchemaError):
            validate_hi_train_config(cfg, env={})


class TestTheManifestHoldsTheReference:
    def test_the_manifest_carries_the_qualified_reference(self) -> None:
        reference = reference_energy(load_evaluation_manifest(EVALUATION_MANIFEST))
        assert reference.energy == pytest.approx(-2.903724377034119598)
        assert reference.qualification == "infinite_mass_nonrelativistic"
        assert reference.units == "hartree"
        assert reference.system_id == "he_atom"

    def test_the_manifest_declares_the_evaluation_schema(self) -> None:
        manifest = load_evaluation_manifest(EVALUATION_MANIFEST)
        assert manifest.schema == HI_EVALUATION_SCHEMA

    def test_loading_the_train_config_as_a_manifest_is_refused(self, ) -> None:
        """The mistake this guards is silent, which is why it is guarded.

        A training config read as a manifest would simply have no reference in
        it, and "no reference found" is indistinguishable from success unless
        the loader refuses the file outright.
        """

        with pytest.raises(ValueError, match="expected 'tpen.hi.evaluation.v1'"):
            load_evaluation_manifest(CONTROL_CONFIG)

    def test_an_unqualified_reference_is_refused(self, tmp_path) -> None:
        """A reference with no qualification cannot be compared honestly.

        Nothing would record which physics produced it, so a discrepancy
        against a differently qualified value would read as model error.
        """

        path = tmp_path / "manifest.yaml"
        path.write_text(
            f"schema: {HI_EVALUATION_SCHEMA}\nreference:\n  energy: -2.9\n  units: hartree\n"
            "  system_id: he_atom\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="qualification"):
            reference_energy(load_evaluation_manifest(path))
