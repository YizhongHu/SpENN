"""The A8 Gaussian coordinate gate, and the exposure of the three A8 arms.

The study admits exactly three feature updates: ``x + u``, ``u``, and
``x + g(R) u``. Two of the three already existed as landed primitives; this
slice exposes them and adds the coordinate gate for the third.

Most of what follows tests what the gate is NOT, because the failure mode here
is substitution rather than breakage. A norm gate, an RMS normalization and this
gate all multiply the update by something in ``(0, 1)``, so all three look alike
in a training curve. Only one reads the electron coordinates.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tpen.data.batch import ElectronBatch
from tpen.data.real import Feature, Update
from tpen.nn import GaussianCoordinateGate, ReplaceUpdater, ResidualUpdater
from tpen.nn.context import TPENForwardContext

SIGMA = 1.0


def _batch(positions, nuclear_positions=((0.0, 0.0, 0.0),)):
    """Build a batch from nested position lists, float64 as the study requires."""

    return ElectronBatch(
        positions=torch.tensor(positions, dtype=torch.float64),
        nuclear_positions=torch.tensor(nuclear_positions, dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0] * len(nuclear_positions), dtype=torch.float64),
    )


def _context(positions, **kwargs):
    return TPENForwardContext(batch=_batch(positions, **kwargs))


def _update(batch_size=1, channels=2, n_particles=2, fill=1.0):
    """Build an Update with a zero-order block plus orders 1 and 2."""

    return Update(
        [
            torch.zeros(batch_size, 0, dtype=torch.float64),
            torch.full((batch_size, channels, n_particles), fill, dtype=torch.float64),
            torch.full((batch_size, channels, n_particles, n_particles), fill, dtype=torch.float64),
        ]
    )


class TestGateValue:
    def test_matches_the_closed_form(self) -> None:
        """One helium configuration, computed by hand from the definition."""

        # Electrons at (1,0,0) and (0,2,0): r1^2 + r2^2 = 1 + 4 = 5.
        gate = GaussianCoordinateGate(sigma=SIGMA).gate(
            _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        )
        assert gate.shape == (1,)
        assert gate.item() == pytest.approx(math.exp(-5.0 / 2.0))

    def test_is_one_at_the_nucleus(self) -> None:
        gate = GaussianCoordinateGate().gate(
            _context([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        )
        assert gate.item() == pytest.approx(1.0)

    def test_decays_far_from_the_nucleus(self) -> None:
        """The whole point of the arm: the correction fades when electrons leave."""

        near = GaussianCoordinateGate().gate(_context([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]]))
        far = GaussianCoordinateGate().gate(_context([[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]]))
        assert near.item() > 0.9
        assert far.item() < 1e-20

    def test_sigma_widens_the_gate(self) -> None:
        positions = [[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        narrow = GaussianCoordinateGate(sigma=0.5).gate(_context(positions))
        wide = GaussianCoordinateGate(sigma=4.0).gate(_context(positions))
        assert wide.item() > narrow.item()

    def test_one_gate_per_sample(self) -> None:
        gate = GaussianCoordinateGate().gate(
            _context(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            )
        )
        assert gate.shape == (2,)
        assert gate[0].item() == pytest.approx(1.0)
        assert gate[1].item() == pytest.approx(math.exp(-0.5))

    def test_rejects_a_nonpositive_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            GaussianCoordinateGate(sigma=0.0)


class TestWhatTheGateIsNot:
    """Substitution, not breakage, is the failure mode this slice guards."""

    def test_the_gate_does_not_depend_on_the_update(self) -> None:
        """NOT exp(-u^2), and not any function of the update's magnitude.

        A norm gate would change when u changes. This one must not: scaling u
        by 3 must scale the output by exactly 3.
        """

        gate = GaussianCoordinateGate()
        context = _context([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        small = gate(_update(fill=1.0), context)
        large = gate(_update(fill=3.0), context)
        for lhs, rhs in zip(small.blocks[1:], large.blocks[1:]):
            assert torch.allclose(3.0 * lhs, rhs)

    def test_one_scalar_is_shared_across_orders_channels_and_tuples(self) -> None:
        """Not a per-electron, per-channel or per-tuple gate."""

        context = _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        gated = GaussianCoordinateGate()(_update(fill=1.0), context)
        expected = math.exp(-5.0 / 2.0)
        for block in gated.blocks[1:]:
            assert torch.allclose(block, torch.full_like(block, expected))

    def test_the_zero_order_block_survives(self) -> None:
        gated = GaussianCoordinateGate()(_update(), _context([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))
        assert gated.blocks[0].shape == (1, 0)

    def test_the_norm_gated_updater_is_not_exported(self) -> None:
        """It must not be reachable as an A8 arm from a config's ``tpen.nn.*``."""

        import tpen.nn

        assert not hasattr(tpen.nn, "NormGatedUpdater")
        assert "NormGatedUpdater" not in tpen.nn.__all__


class TestCoordinateSemantics:
    def test_distances_are_measured_from_the_nucleus_not_the_origin(self) -> None:
        """Translate the whole atom; the gate must not notice.

        For the helium control the nucleus sits at the origin, so a gate that
        wrongly measured from the origin would agree with this one on every
        production configuration and the error would never surface. Moving the
        atom is the only thing that separates them.
        """

        at_origin = GaussianCoordinateGate().gate(
            _context([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], nuclear_positions=((0.0, 0.0, 0.0),))
        )
        shifted = GaussianCoordinateGate().gate(
            _context([[[6.0, 0.0, 0.0], [5.0, 1.0, 0.0]]], nuclear_positions=((5.0, 0.0, 0.0),))
        )
        assert shifted.item() == pytest.approx(at_origin.item())

    def test_an_origin_measuring_gate_would_disagree(self) -> None:
        """Positive control for the test above, so it cannot pass vacuously."""

        origin_value = math.exp(-(6.0**2 + 5.0**2 + 1.0**2) / 2.0)
        nucleus_value = math.exp(-2.0 / 2.0)
        assert origin_value != pytest.approx(nucleus_value)

    def test_the_gate_is_permutation_invariant(self) -> None:
        """It is a sum over electrons, so relabelling them changes nothing."""

        first = GaussianCoordinateGate().gate(_context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]))
        swapped = GaussianCoordinateGate().gate(_context([[[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]]))
        assert swapped.item() == pytest.approx(first.item())

    def test_more_than_one_nucleus_is_refused(self) -> None:
        """Refused rather than assigned an unstated convention.

        Nearest nucleus, all nuclei, and a charge-weighted sum are all
        defensible and are different models. The authority defines the gate for
        a one-nucleus atom only.
        """

        context = _context(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            nuclear_positions=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
        )
        with pytest.raises(ValueError, match="single nucleus"):
            GaussianCoordinateGate().gate(context)

    def test_a_missing_nucleus_is_refused(self) -> None:
        batch = ElectronBatch(positions=torch.zeros(1, 2, 3, dtype=torch.float64))
        with pytest.raises(ValueError, match="nuclear_positions"):
            GaussianCoordinateGate().gate(TPENForwardContext(batch=batch))


class TestTheThreeAdmittedArms:
    """Exactly three, spelled the way a config will spell them."""

    def test_residual_is_x_plus_u(self) -> None:
        x = Feature(
            [
                torch.zeros(1, 0, dtype=torch.float64),
                torch.full((1, 2, 2), 5.0, dtype=torch.float64),
                torch.full((1, 2, 2, 2), 5.0, dtype=torch.float64),
            ]
        )
        out = ResidualUpdater()(x, _update(fill=1.0))
        assert torch.allclose(out.blocks[1], torch.full_like(out.blocks[1], 6.0))

    def test_replacement_is_u(self) -> None:
        x = Feature(
            [
                torch.zeros(1, 0, dtype=torch.float64),
                torch.full((1, 2, 2), 5.0, dtype=torch.float64),
                torch.full((1, 2, 2, 2), 5.0, dtype=torch.float64),
            ]
        )
        out = ReplaceUpdater()(x, _update(fill=1.0))
        assert torch.allclose(out.blocks[1], torch.full_like(out.blocks[1], 1.0))

    def test_gaussian_residual_is_x_plus_gate_times_u(self) -> None:
        """The composed arm, which is how a config expresses the third level."""

        x = Feature(
            [
                torch.zeros(1, 0, dtype=torch.float64),
                torch.full((1, 2, 2), 5.0, dtype=torch.float64),
                torch.full((1, 2, 2, 2), 5.0, dtype=torch.float64),
            ]
        )
        context = _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        gated = GaussianCoordinateGate()(_update(fill=1.0), context)
        out = ResidualUpdater()(x, gated)
        expected = 5.0 + math.exp(-5.0 / 2.0)
        assert torch.allclose(out.blocks[1], torch.full_like(out.blocks[1], expected))

    def test_all_three_are_reachable_from_tpen_nn(self) -> None:
        """A config names these by dotted path; unexported means unusable."""

        import tpen.nn

        for name in ("ResidualUpdater", "ReplaceUpdater", "GaussianCoordinateGate"):
            assert hasattr(tpen.nn, name), name
            assert name in tpen.nn.__all__, name

    def test_the_three_arms_are_distinguishable(self) -> None:
        """They must actually differ, or the scan compares one thing three times."""

        x = Feature(
            [
                torch.zeros(1, 0, dtype=torch.float64),
                torch.full((1, 2, 2), 5.0, dtype=torch.float64),
                torch.full((1, 2, 2, 2), 5.0, dtype=torch.float64),
            ]
        )
        u = _update(fill=1.0)
        context = _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])

        residual = ResidualUpdater()(x, u).blocks[1]
        replacement = ReplaceUpdater()(x, u).blocks[1]
        gaussian = ResidualUpdater()(x, GaussianCoordinateGate()(u, context)).blocks[1]

        assert not torch.allclose(residual, replacement)
        assert not torch.allclose(residual, gaussian)
        assert not torch.allclose(replacement, gaussian)


class TestGradients:
    def test_the_gate_passes_gradient_to_the_update(self) -> None:
        context = _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        block = torch.full((1, 2, 2), 1.0, dtype=torch.float64, requires_grad=True)
        update = Update([torch.zeros(1, 0, dtype=torch.float64), block])
        GaussianCoordinateGate()(update, context).blocks[1].sum().backward()
        assert block.grad is not None
        assert torch.allclose(block.grad, torch.full_like(block.grad, math.exp(-5.0 / 2.0)))

    def test_the_gate_carries_gradient_to_the_coordinates(self) -> None:
        """The gate is part of the wavefunction, so the local energy needs this."""

        positions = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float64, requires_grad=True
        )
        batch = ElectronBatch(
            positions=positions,
            nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
        )
        GaussianCoordinateGate().gate(TPENForwardContext(batch=batch)).sum().backward()
        assert positions.grad is not None
        assert not torch.allclose(positions.grad, torch.zeros_like(positions.grad))

    def test_the_gate_has_no_parameters(self) -> None:
        """It is one configuration-level scalar, not a learned gate."""

        assert list(GaussianCoordinateGate().parameters()) == []
        assert GaussianCoordinateGate().state_dict() == {}
