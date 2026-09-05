"""SR consumes what `TPENWaveFunction` actually EMITS, not a synthetic stand-in.

Every other test of this engine builds its `ScoreUpdateInput` from hand-made
score blocks, which is right for pinning the algebra but blind to one thing: a
mismatch between what the engine consumes and what the wavefunction really
produces.  Layout order, sample-shape flattening, dtype, detachment, and the
sign convention are all agreements between two modules, and a synthetic input
satisfies the consumer's half of every one of them by construction.

So these tests drive the real provider,
:meth:`TPENWaveFunction.evaluate_materialized_parameter_score_request`, and feed
its output straight into the update method.  No trainer is involved -- the
trainer's own integration is a later slice -- so this establishes that the seam
is consumable at THIS layer, where a mismatch is cheap to find.

It found one.  See `test_score_seam_refuses_a_two_electron_tpen_model`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.hooke_models import build_tiny_spenn
from tests.helpers.sr_dense_oracle import energy_gradient, sr_direction
from tests.unit.nn.test_tpen_wavefunction_parameter_scores import _build_model
from tpen.data.batch import ElectronBatch, ParameterScoreForwardPacket
from tpen.nn import InteractionMode, MaterializedParameterScoreRequest
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput
from tpen.training.vmc import compute_vmc_objective

LEARNING_RATE = 1.0e-3
SOLVE_TOLERANCE = 1.0e-9


def _connected_model():
    """Build a real TPENWaveFunction whose parameters are all score-reachable.

    Deliberately NOT `build_tiny_spenn()`: at its two-electron count 16 of its
    39 parameters are unreachable and the provider refuses outright, which is
    the blocker pinned at the bottom of this file.
    """

    torch.manual_seed(0)
    return _build_model(InteractionMode.TENSOR_PRODUCT)


def _batch(n_walkers: int = 6, *, seed: int = 5) -> ElectronBatch:
    """Return a flat-sample-shape three-electron batch."""

    generator = torch.Generator().manual_seed(seed)
    return ElectronBatch(
        positions=torch.randn(n_walkers, 3, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0, 1.0]] * n_walkers, dtype=torch.float64),
    )


def _energies(n_walkers: int, *, seed: int = 17) -> torch.Tensor:
    """Deterministic finite local energies.

    These are synthetic on purpose. The seam under test here is the SCORE path;
    where the energies come from is the Hamiltonian's business and would only
    add an unrelated failure source.
    """

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_walkers, generator=generator, dtype=torch.float64)


def _emit(model, batch: ElectronBatch, *, chunk_size: int | None = None):
    """Ask the real provider for raw parameter-score blocks."""

    packet = model.evaluate_materialized_parameter_score_request(
        request=MaterializedParameterScoreRequest(chunk_size=chunk_size),
        batch=batch,
    )
    assert isinstance(packet, ParameterScoreForwardPacket)
    return packet


def _independent_flatten(packet) -> np.ndarray:
    """Flatten emitted blocks to ``[B, P]`` WITHOUT the module under test.

    `flatten_parameter_score_blocks` is the subject here, so using it would
    make the oracle comparison circular. Re-deriving the column order from the
    layout is the point: if the engine's ordering ever diverges from the
    layout's own slot order, this disagrees.
    """

    scores = packet.parameter_scores
    n_samples = int(np.prod(scores.sample_shape)) if scores.sample_shape else 1
    columns = [
        block.detach().numpy().reshape(n_samples, slot.numel)
        for slot, block in zip(scores.layout.slots, scores.blocks, strict=True)
    ]
    return np.hstack(columns)


def _method(model, *, solve_space: str = "parameter", relative: float = 1.0e-2):
    """Build an SR method over the model's live parameters."""

    parameters = tuple(model.parameters())
    policy = SRPolicy(
        solve_space=solve_space,
        damping=DampingPolicy(absolute=0.0, relative=relative, minimum=1.0e-12),
        learning_rate=LEARNING_RATE,
    )
    return StochasticReconfigurationUpdate(
        torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
        policy=policy,
        conventions=ScoreConventions(solve_dtype=torch.float64),
    )


def _score_input(model, batch, packet, energies, *, step: int = 0) -> ScoreUpdateInput:
    """Assemble the typed input from a REAL emitted packet.

    The batch is passed explicitly rather than stashed at module scope: these
    tests must not depend on execution order, and pytest-xdist would make a
    shared global genuinely wrong rather than merely untidy.
    """

    return ScoreUpdateInput(
        batch=batch,
        wavefunction=packet.output,
        local_energy=energies,
        step=step,
        parameter_scores=packet.parameter_scores,
        parameter_binding=model.parameter_binding,
    )


def test_the_emitted_blocks_are_the_uncentered_scores_the_engine_assumes() -> None:
    """The provider emits ``d log|psi| / d theta`` per sample, raw and uncentered.

    The engine's entire geometry rests on that convention. Recomputing the same
    quantity with an independent per-sample autograd loop checks the emitted
    payload against its stated meaning rather than against the engine.
    """

    model = _connected_model()
    batch = _batch()
    packet = _emit(model, batch)
    emitted = _independent_flatten(packet)

    parameters = model.parameter_binding.parameters
    with torch.enable_grad():
        logabs = model(batch).logabs
        rows = []
        for index in range(int(logabs.numel())):
            grads = torch.autograd.grad(
                logabs.reshape(-1)[index],
                parameters,
                retain_graph=index + 1 < int(logabs.numel()),
            )
            rows.append(np.concatenate([g.detach().numpy().reshape(-1) for g in grads]))
    reference = np.vstack(rows)

    assert emitted.shape == reference.shape
    np.testing.assert_allclose(emitted, reference, rtol=1.0e-12, atol=1.0e-12)
    # Raw means UNCENTERED: the column means are generally nonzero, and the
    # engine is the thing that centers them.
    assert np.abs(emitted.mean(axis=0)).max() > 0.0
    # Detached: a live graph here would leak into the optimizer's parameters.
    assert not any(block.requires_grad for block in packet.parameter_scores.blocks)


def test_sr_consumes_a_real_emitted_packet_and_matches_the_oracle() -> None:
    """End to end at the engine layer: real emission in, oracle-checked step out.

    This is the check that a synthetic `ScoreUpdateInput` cannot make. It uses
    the model's own `parameter_binding` and the provider's own blocks, so a
    disagreement in layout order, sample flattening, dtype, or sign convention
    between the two modules shows up here.
    """

    model = _connected_model()
    batch = _batch()
    packet = _emit(model, batch)
    energies = _energies(int(packet.output.logabs.numel()))
    before = [p.detach().clone() for p in model.parameters()]

    method = _method(model)
    result = method.update(_score_input(model, batch, packet, energies))

    assert result.applied is True
    expected = sr_direction(
        _independent_flatten(packet),
        energies.numpy(),
        absolute=0.0,
        relative=1.0e-2,
    )
    displacement = np.concatenate(
        [
            (b - p.detach()).numpy().reshape(-1)
            for b, p in zip(before, model.parameters(), strict=True)
        ]
    )
    np.testing.assert_allclose(
        displacement,
        LEARNING_RATE * expected.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    assert result.grad_norm == pytest.approx(
        float(np.linalg.norm(expected.gradient)), rel=1.0e-9
    )


def test_minsr_agrees_with_dense_sr_on_real_emitted_scores() -> None:
    """The two routes agree on real emissions, not only on synthetic matrices."""

    batch = _batch()
    results = {}
    for space in ("parameter", "sample"):
        model = _connected_model()
        packet = _emit(model, batch)
        energies = _energies(int(packet.output.logabs.numel()))
        method = _method(model, solve_space=space)
        method.update(_score_input(model, batch, packet, energies))
        results[space] = np.concatenate(
            [p.detach().numpy().reshape(-1) for p in model.parameters()]
        )
        assert method.last_telemetry.diagnostics.space == space

    np.testing.assert_allclose(
        results["parameter"], results["sample"], rtol=1.0e-8, atol=1.0e-8
    )


def test_euclidean_limit_on_real_scores_matches_the_real_objective_gradient() -> None:
    """With damping dominant, the step aligns with the model's own VMC gradient.

    Both sides now come from the same real model: the scores from the provider,
    the reference from autograd through `compute_vmc_objective`. That closes the
    convention loop end to end rather than assuming the emitted sign.
    """

    model = _connected_model()
    batch = _batch()
    packet = _emit(model, batch)
    energies = _energies(int(packet.output.logabs.numel()))

    with torch.enable_grad():
        logabs = model(batch).logabs
        compute_vmc_objective(logabs, energies).loss.backward()
    reference = np.concatenate(
        [p.grad.detach().numpy().reshape(-1) for p in model.parameters()]
    )
    for parameter in model.parameters():
        parameter.grad = None

    method = _method(model, relative=1.0e10)
    method.update(_score_input(model, batch, packet, energies))
    direction = np.concatenate(
        [p.grad.detach().numpy().reshape(-1) for p in model.parameters()]
    )

    np.testing.assert_allclose(
        direction / np.linalg.norm(direction),
        reference / np.linalg.norm(reference),
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    # And the oracle agrees with autograd on the emitted scores, so the
    # reference itself is not taken on trust.
    np.testing.assert_allclose(
        energy_gradient(_independent_flatten(packet), energies.numpy()),
        reference,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_chunked_emission_is_consumable_and_gives_the_same_step() -> None:
    """The provider has two implementations; the engine must consume both.

    `chunk_size=None` materializes one gradient per sample, a nonzero
    `chunk_size` uses batched VJPs. They are different code paths and only one
    of them is exercised by default.
    """

    batch = _batch()
    steps = []
    for chunk_size in (None, 2):
        model = _connected_model()
        packet = _emit(model, batch, chunk_size=chunk_size)
        energies = _energies(int(packet.output.logabs.numel()))
        method = _method(model)
        assert method.update(_score_input(model, batch, packet, energies)).applied is True
        steps.append(
            np.concatenate([p.detach().numpy().reshape(-1) for p in model.parameters()])
        )

    np.testing.assert_allclose(steps[0], steps[1], rtol=1.0e-9, atol=1.0e-9)


def test_score_seam_refuses_a_two_electron_tpen_model() -> None:
    """PINS A BLOCKER: SR cannot run on a two-electron TPEN model, so not on helium.

    Measured on Cannon job 44572987: of the 39 trainable parameters in
    `build_tiny_spenn()`, 16 are structurally disconnected from ``logabs`` at
    its two-electron count -- `stack.layers.0.mixing.weights.g0` through `g14`
    and `stack.layers.0.path_aggregation.weights.o1`. The equivariant mixing
    allocates a weight per tensor path, and at two electrons most of those
    paths carry nothing, so autograd never reaches them.

    Both `_slow_parameter_score_blocks` and `_chunked_parameter_score_blocks`
    pass ``allow_unused=False`` and convert the resulting error into
    "materialized parameter scores found an unused or disconnected parameter",
    failing the ENTIRE request.

    The guard is right about the case it was built for -- a parameter no code
    path consumes, its own `_UnusedPfaffianReadout` test -- and cannot tell that
    apart from a tensor path that is simply empty at this particle count.
    Flipping ``allow_unused`` would destroy its real purpose, and `tpen/nn/` is
    outside this lane's write surface, so the fix belongs to the seam's owner.
    The correct score for a structurally inactive parameter is exactly zero, so
    a seam that made that distinction would need no change on the SR side.

    Asserts the CURRENT behaviour, so it fails the moment the seam is fixed --
    at which point `_connected_model` above can become `build_tiny_spenn`.
    """

    model = build_tiny_spenn()
    batch = ElectronBatch(
        positions=torch.zeros((4, 2, 3), dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64),
    )

    for chunk_size in (None, 2):
        with pytest.raises(RuntimeError, match="unused or disconnected"):
            _emit(model, batch, chunk_size=chunk_size)


def test_the_two_electron_refusal_is_caused_by_disconnection_specifically() -> None:
    """The blocker is disconnection, not some other breakage of that model.

    Without this, the test above would pass for any reason the emission failed
    and would keep passing if the cause changed. Counting unreachable
    parameters names the cause directly, and the second half confirms the model
    the rest of this file uses is NOT subject to it -- otherwise those tests
    would prove nothing.
    """

    blocked = build_tiny_spenn()
    blocked_batch = ElectronBatch(
        positions=torch.zeros((4, 2, 3), dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64),
    )
    parameters = blocked.parameter_binding.parameters
    with torch.enable_grad():
        grads = torch.autograd.grad(
            blocked(blocked_batch).logabs.sum(), parameters, allow_unused=True
        )
    unreachable = [
        slot.ordinal
        for slot, grad in zip(
            blocked.parameter_binding.layout.slots, grads, strict=True
        )
        if grad is None
    ]

    assert unreachable, "the blocker is disconnection; finding none means it is fixed"
    assert len(unreachable) < len(parameters), "some parameters must remain reachable"

    connected = _connected_model()
    connected_batch = _batch()
    with torch.enable_grad():
        connected_grads = torch.autograd.grad(
            connected(connected_batch).logabs.sum(),
            connected.parameter_binding.parameters,
            allow_unused=True,
        )
    assert all(grad is not None for grad in connected_grads)
