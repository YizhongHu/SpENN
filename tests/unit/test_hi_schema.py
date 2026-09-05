"""Tests for the closed helium-importance train schema policy.

The falsifier named in L1a's acceptance contract is "a nested forbidden
reference reaches construction". These tests pin the rejection half of it; the
half that proves nothing was constructed lives with the ``tpen.run`` wiring,
because only there is there anything to construct.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tpen.config_schema import ClosedSchemaError
from tpen.hi_schema import (
    ADMITTED_CALLBACK_TARGETS,
    HI_TRAIN_SCHEMA,
    declared_schema,
    validate_hi_train_config,
)


def _config(**sections: object):
    """Return a schema-declaring HI train config with ``sections`` merged in."""

    base: dict[str, object] = {"schema": HI_TRAIN_SCHEMA}
    base.update(sections)
    return OmegaConf.create(base)


def _rules(error: ClosedSchemaError) -> set[str]:
    return {rejection.rule for rejection in error.rejections}


def _paths(error: ClosedSchemaError) -> set[str]:
    return {rejection.path for rejection in error.rejections}


class TestOptIn:
    """The firewall applies to configurations that ask for it, and only those."""

    def test_a_config_declaring_no_schema_is_not_validated(self) -> None:
        cfg = OmegaConf.create({"system": {"reference_energy": -2.9}})
        validate_hi_train_config(cfg)

    def test_a_config_declaring_another_schema_is_not_validated(self) -> None:
        cfg = OmegaConf.create({"schema": "other.v1", "system": {"reference_energy": -2.9}})
        validate_hi_train_config(cfg)

    def test_declared_schema_reads_the_top_level_key(self) -> None:
        assert declared_schema(_config()) == HI_TRAIN_SCHEMA
        assert declared_schema(OmegaConf.create({})) is None

    def test_the_frozen_he_v1_train_fixture_is_left_alone(self) -> None:
        """The historical fixture is noncompliant BY RECORD and must stay loadable.

        ``experiments/atomistic/he-v1/configs/train.yaml`` carries
        ``system.reference_energy``. The plan of record designates it a
        historical implementation fixture to be preserved, not edited. It
        declares no schema, so the firewall must not touch it -- and it must
        also be genuinely noncompliant, or this test would pass for the wrong
        reason and stop protecting anything.
        """

        cfg = OmegaConf.load("experiments/atomistic/he-v1/configs/train.yaml")
        validate_hi_train_config(cfg)

        # Red arm: opting the same content in must reject it.
        opted_in = OmegaConf.merge(OmegaConf.create({"schema": HI_TRAIN_SCHEMA}), cfg)
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(opted_in)
        assert "system.reference_energy" in _paths(caught.value)


class TestForbiddenSurfaces:
    def test_rejects_a_nested_reference_energy(self) -> None:
        cfg = _config(system={"nuclei": {"reference_energy": -2.903724377034119598}})
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert "forbidden-surface:reference" in _rules(caught.value)
        assert "system.nuclei.reference_energy" in _paths(caught.value)

    def test_rejects_a_reference_reached_only_through_interpolation(self) -> None:
        """The forbidden value is a plain number until resolution moves it.

        ``model.exact`` is not a forbidden key name. What makes this a violation
        is that resolution places the value under ``trainer.baseline_energy``,
        which is. A raw-tree-only sweep would still catch the destination key
        here; the point of the case is that the two trees agree, so a later
        change that drops one sweep does not silently pass this config.
        """

        cfg = _config(
            model={"scalar": -2.9},
            trainer={"baseline_energy": "${model.scalar}"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        trees = {r.tree for r in caught.value.rejections if r.path == "trainer.baseline_energy"}
        assert trees == {"raw", "resolved"}

    @pytest.mark.parametrize(
        ("key", "rule"),
        [
            ("energy_gap", "forbidden-surface:gap"),
            ("accuracy_band", "forbidden-surface:band"),
            ("continuation_from", "forbidden-surface:continuation"),
            ("ground_truth", "forbidden-surface:reference"),
        ],
    )
    def test_rejects_each_forbidden_family(self, key: str, rule: str) -> None:
        cfg = _config(trainer={key: 1})
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert rule in _rules(caught.value)

    def test_checkpoint_resume_is_not_a_continuation_surface(self) -> None:
        """Resume is recovery; continuation is selection. Only one is forbidden.

        ``tpen.checkpoint.TrainResume`` is the standard payload and appears in
        every training configuration. A continuation rule that also swept
        ``resume`` would make the schema unsatisfiable.
        """

        cfg = _config(
            callbacks=[
                {
                    "_target_": "tpen.callback.Checkpoint",
                    "payload": {"_target_": "tpen.checkpoint.TrainResume"},
                    "output_dir": "out",
                }
            ]
        )
        validate_hi_train_config(cfg)

    def test_bandwidth_is_not_a_band_surface(self) -> None:
        """Substring matching would reject this ordinary key."""

        validate_hi_train_config(_config(sampler={"bandwidth": 1.0}))


class TestClosedSections:
    def test_rejects_an_undeclared_top_level_section(self) -> None:
        cfg = _config(diagnostics={"kind": "energy"})
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert "unknown-field" in _rules(caught.value)
        assert "diagnostics" in _paths(caught.value)

    def test_accepts_the_declared_sections(self) -> None:
        validate_hi_train_config(
            _config(
                experiment={"name": "hi"},
                run={"root": "outputs"},
                runtime={"seed": 0},
                system={"n_particles": 2},
                model={"channels": 32},
                trainer={"max_steps": 10},
            )
        )


class TestForbiddenResolvers:
    """Environment and clock interpolation are how two ranks could diverge."""

    def test_rejects_an_environment_interpolation(self) -> None:
        cfg = _config(runtime={"seed": "${oc.env:RANK,0}"})
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert "forbidden-resolver" in _rules(caught.value)

    def test_rejects_a_clock_interpolation(self) -> None:
        cfg = _config(run={"root": "outputs/${now:%Y-%m-%d}"})
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert "forbidden-resolver" in _rules(caught.value)

    def test_accepts_an_ordinary_node_reference(self) -> None:
        validate_hi_train_config(
            _config(system={"spatial_dim": 3}, model={"spatial_dim": "${system.spatial_dim}"})
        )


class TestAdmittedCallbacks:
    def test_rejects_a_callback_outside_the_admitted_set(self) -> None:
        cfg = _config(callbacks=[{"_target_": "tpen.diagnostics.energy.EnergyDiagnostic"}])
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert "unadmitted-callback" in _rules(caught.value)

    def test_accepts_every_admitted_callback(self) -> None:
        cfg = _config(callbacks=[{"_target_": name} for name in sorted(ADMITTED_CALLBACK_TARGETS)])
        validate_hi_train_config(cfg)

    def test_a_nested_target_is_not_judged_as_a_callback(self) -> None:
        """A schedule or payload is a constructor argument, not a callback.

        ``tpen.checkpoint.EveryNUpdates`` is not in the callback allowlist and
        must not be, so treating every nested ``_target_`` as a callback
        identity would reject a standard checkpoint block.
        """

        cfg = _config(
            callbacks=[
                {
                    "_target_": "tpen.callback.Checkpoint",
                    "schedule": {"_target_": "tpen.checkpoint.EveryNUpdates", "every_n": 1000},
                }
            ]
        )
        validate_hi_train_config(cfg)

    def test_no_admitted_callback_carries_a_reference_in_its_name(self) -> None:
        """A cheap standing guard on the allowlist itself.

        The allowlist is hand-maintained, so the failure mode is someone adding
        a reference-bearing callback to it. This cannot catch a reference hidden
        behind an innocent class name, and is not claimed to -- it catches the
        careless case for free.
        """

        from tpen.config_schema import tokens_of

        for target in ADMITTED_CALLBACK_TARGETS:
            assert "reference" not in tokens_of(target.rsplit(".", 1)[-1])


class TestUnresolvableConfig:
    def test_a_broken_interpolation_is_reported_as_a_rejection(self) -> None:
        """Preconstruction failure stays one exception type for the caller."""

        cfg = _config(model={"channels": "${missing.node}"})
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert "unresolvable" in _rules(caught.value)

    def test_a_broken_interpolation_does_not_suppress_the_raw_findings(self) -> None:
        """The reference must survive a config that also fails to resolve.

        This is the case where the finding and the thing that hides it share a
        failure domain. If the raw sweep ran only after a successful
        resolution, one unrelated typo would turn a reference-bearing config
        into a bare "does not resolve" -- and the author would fix the typo,
        rerun, and only then discover the reference. On this project the rerun
        can be a cluster job.
        """

        cfg = _config(
            system={"reference_energy": -2.9},
            model={"channels": "${missing.node}"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert {"unresolvable", "forbidden-surface:reference"} <= _rules(caught.value)
        assert "system.reference_energy" in _paths(caught.value)


class TestEveryFindingIsReported:
    def test_multiple_violations_arrive_together(self) -> None:
        """One cluster cycle per violation is the cost of failing on the first."""

        cfg = _config(
            sneaky={},
            system={"reference_energy": -2.9},
            trainer={"accuracy_band": 1},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            validate_hi_train_config(cfg)
        assert {
            "unknown-field",
            "forbidden-surface:reference",
            "forbidden-surface:band",
        } <= _rules(caught.value)
