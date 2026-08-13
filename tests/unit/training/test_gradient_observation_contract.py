"""Regression test for the undeclared ``optimizer.zero_grad`` ordering contract.

`tpen.callback.GradientStats` observes gradients AFTER ``optimizer.step()`` has
already consumed them. That works only because the trainer clears gradients
part-way through the *following* iteration -- ``optimizer.zero_grad`` sits
before `Backward`, not after the update -- so the gradients written by
``loss.backward()`` are still live when the iteration completes.

Nothing declared that. A probe run on FASRC Cannon at SHA ``65ac4d9`` moved that
single statement to just after ``optimizer.step()`` and measured the result on
an A100 over six steps:

===== ================================== =================================
step   ``checks/gradient/global_grad_norm``  ``train/grad_norm`` (control)
===== ================================== =================================
0      1.88629012 -> **0.0**                1.88629012, bit-identical
...    ...                                  ...
5      4.55848862 -> **0.0**                4.55848862, bit-identical
===== ================================== =================================

``n_grad_tensors`` went 23 -> 0 and ``n_grad_elements`` 1312 -> 0: the callback
saw no gradient tensors at all, yet still logged as if the numbers were valid.
The whole suite reported ``859 passed, 3 skipped`` in BOTH arms. Not one test
noticed.

The *second* half of that finding -- that the callback reported ``passed`` while
observing nothing -- was filed as defect ``933b5f78`` and recorded here rather
than fixed here. It is fixed now: `tpen.callback.GradientStats` fails when it
sees no gradients on an iteration whose ``optimizer_step`` is ``True``. The two
arm-B tests below therefore assert ``passed is False``, and those assertions do
double duty: they are the only end-to-end check that the trainer still hands the
callback its ``optimizer_step`` discriminator at all. Drop that one assignment in
`tpen.training.trainer` and every gradient observation silently becomes
unfalsifiable again -- these two tests are what notices.

The Event Clock migration preserves that boundary rather than moving it (posture
A), so the contract survives and is DECLARED here instead of being fixed by
relocating the observer. Three tests do that job together:

`test_gradient_stats_observes_the_gradients_the_update_consumed` is the contract
itself, and fails the moment ``zero_grad`` moves after the update.

`test_clearing_gradients_after_the_update_empties_every_gradient_metric` gives
that first test teeth, by reproducing the probe's arm B and pinning that the
move really does zero the metric while leaving training untouched. Without it
the contract test could pass vacuously.

`test_the_perturbation_moves_only_what_the_observer_sees` runs the two arms
against each other. That pairing is what makes the teeth evidence rather than a
tautology: a perturbation that emptied the metric by wrecking training would
satisfy the test above on its own, and only a term-by-term comparison against
the control rules it out.

The perturbation is applied through the optimizer rather than by editing
`tpen.training.trainer`, so the trainer under test is the shipped one. A second
Cannon probe (job ``38268849``) drove this module's own ``_fit`` and
`_ZeroGradAfterStep` and confirmed the substitution reproduces the source-level
move exactly: ``n_grad_tensors`` 23 -> 0, ``global_grad_norm`` -> 0.0, and
``train/grad_norm``/``train/loss`` bit-identical between the two arms.

Getting that far took two rounds, and round one is the reason `_fit` seeds. The
tiny model's weights are drawn from the GLOBAL torch RNG -- `build_tiny_spenn`
calls ``instantiate(cfg.model)``, which seeds nothing -- so back-to-back arms in
one process start from different initialisations and their trajectories diverge
for reasons that have nothing to do with gradients. Job ``38267439`` read that
drift as the perturbation changing training. It was not; it was the measurement.
The sampler was never the problem (``pair_train.yaml`` sets ``seed:
${runtime.seed}`` = 0 and `MetropolisSampler` owns its generator), but a
``None`` seed there would draw from OS entropy just as silently, so `_fit`
asserts the seed is set rather than trusting the fixture to keep supplying it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tpen.callback import GradientStats
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn
from tests.helpers.run_context import RecordingLogger, make_run_context

MAX_STEPS = 2
SEED = 0


class _ZeroGradAfterStep(torch.optim.Adam):
    """Adam that behaves as if ``zero_grad`` had been moved after ``step``.

    The trainer's pre-`Backward` ``optimizer.zero_grad(set_to_none=True)``
    becomes a no-op, and the real clear happens immediately after
    ``optimizer.step()`` returns. That is exactly the one-statement move the
    Cannon probe made, expressed without touching the trainer.

    Subclassing the real optimizer is deliberate: ``VMCTrainer.fit`` annotates
    ``optimizer`` as ``torch.optim.Optimizer`` and the suite runs typeguard over
    ``tpen``, so a duck-typed wrapper would be rejected at call time.
    """

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Swallow the trainer's pre-backward clear."""

        del set_to_none

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        """Apply the update, then clear the gradients it just consumed."""

        result = super().step(closure)
        super().zero_grad(set_to_none=True)
        return result


def _fit(
    tmp_path: Path,
    *,
    optimizer_class: type[torch.optim.Optimizer],
    run_id: str = "helper-run",
) -> RecordingLogger:
    """Run the real tiny-TPEN loop with `GradientStats` attached.

    Both sources of randomness are pinned so two calls differing only in
    ``optimizer_class`` are comparable term by term:

    * the global torch RNG, reset immediately before `build_tiny_spenn` because
      that is what the model's initial weights are drawn from;
    * the sampler's own generator, which `MetropolisSampler` seeds itself -- but
      only when it was given a seed, so that is asserted rather than assumed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Root for the run directory, normally pytest's ``tmp_path`` fixture.
    optimizer_class : type
        Optimizer to construct over the model's parameters.
    run_id : str, optional
        Run identifier, and hence the run subdirectory name. Two arms sharing a
        ``tmp_path`` need distinct ids so their artifacts do not collide.
    """

    logger = RecordingLogger()
    callback = GradientStats(fail_fast=False)
    context = make_run_context(tmp_path, callbacks=[callback], loggers=[logger], run_id=run_id)
    # Nothing between this and the model construction may consume the global
    # RNG, or the two arms start from different weights.
    torch.manual_seed(SEED)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    assert sampler.seed is not None, (
        "the smoke fixture stopped supplying a sampler seed: MetropolisSampler "
        "then draws from OS entropy and two arms of this test cannot be compared."
    )
    VMCTrainer(max_steps=MAX_STEPS, log_every_n_steps=1).fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()],
        optimizer=optimizer_class(model.parameters(), lr=0.01),
        context=context,
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )
    return logger


def test_gradient_stats_observes_the_gradients_the_update_consumed(tmp_path: Path) -> None:
    """THE contract. Breaks if ``optimizer.zero_grad`` ever moves after ``step``."""

    logger = _fit(tmp_path, optimizer_class=torch.optim.Adam)

    gradient_records = logger.by_namespace("checks/gradient")
    assert len(gradient_records) == MAX_STEPS
    for record in gradient_records:
        metrics = record.metrics
        assert metrics["n_grad_tensors"] != 0, (
            "GradientStats saw no gradient tensors: something cleared them before "
            "the iteration completed. Check the position of optimizer.zero_grad."
        )
        assert metrics["n_grad_elements"] != 0
        assert metrics["global_grad_norm"] != 0.0

    # The trainer's own pre-update norm is the control: it is computed inside
    # the loop body and is unaffected by when gradients are cleared afterwards.
    for record in logger.by_namespace("train"):
        assert record.metrics["grad_norm"] != 0.0


def test_clearing_gradients_after_the_update_empties_every_gradient_metric(
    tmp_path: Path,
) -> None:
    """Teeth for the contract above: the move really does zero the metric.

    Asserting the defective numbers looks strange in isolation. It is here so
    that the test above cannot pass vacuously -- if some future change made
    `GradientStats` robust to gradient clearing, this test fails and says so,
    and the contract test stops being the thing protecting the metric.
    """

    logger = _fit(tmp_path, optimizer_class=_ZeroGradAfterStep)

    gradient_records = logger.by_namespace("checks/gradient")
    assert len(gradient_records) == MAX_STEPS
    for record in gradient_records:
        metrics = record.metrics
        assert metrics["n_grad_tensors"] == 0
        assert metrics["n_grad_elements"] == 0
        assert metrics["global_grad_norm"] == 0.0
        # What used to make this invisible in production: a check observing
        # nothing still reported success (defect `933b5f78`). It now fails,
        # because this arm's iterations DID apply an optimizer update -- so an
        # empty gradient set here is a broken observation, not an idle step.
        assert metrics["passed"] is False

    # The control the Cannon probe used. Training itself is untouched by the
    # move -- only what the observer can see changes -- so this must stay
    # non-zero while every metric above is empty.
    train_records = logger.by_namespace("train")
    assert len(train_records) == MAX_STEPS
    for record in train_records:
        assert record.metrics["grad_norm"] != 0.0


def test_the_perturbation_moves_only_what_the_observer_sees(tmp_path: Path) -> None:
    """The two arms side by side: the observation window moves, training does not.

    The Cannon probe's decisive control was that ``train/grad_norm`` came out
    BIT-IDENTICAL between arms, which is what isolates the effect to the
    observation mechanism. The two tests above check each arm in isolation and
    so cannot see that; a perturbation that emptied ``checks/gradient`` by
    breaking the update would pass both. This compares them term by term.

    Equality, not closeness: both arms run in one process on one device from the
    same seed, and the claim under test is that the arithmetic is the same
    arithmetic, not merely similar. Job ``38268849`` measured exactly this
    comparison on the shipped `_fit` and got equality on both series.
    """

    control = _fit(tmp_path, optimizer_class=torch.optim.Adam, run_id="control")
    perturbed = _fit(tmp_path, optimizer_class=_ZeroGradAfterStep, run_id="perturbed")

    def train_series(logger: RecordingLogger, key: str) -> list[float]:
        return [record.metrics[key] for record in logger.by_namespace("train")]

    def check_series(logger: RecordingLogger, key: str) -> list[Any]:
        return [record.metrics[key] for record in logger.by_namespace("checks/gradient")]

    assert train_series(control, "grad_norm") == train_series(perturbed, "grad_norm")
    assert train_series(control, "loss") == train_series(perturbed, "loss")

    # ... while the observer went blind. Stated here as the other half of the
    # pairing: the equality above is only interesting because this is true.
    control_tensors = check_series(control, "n_grad_tensors")
    assert all(count != 0 for count in control_tensors), control_tensors
    assert check_series(perturbed, "n_grad_tensors") == [0] * MAX_STEPS

    # ... and now says so. Defect `933b5f78` was that an observer which saw
    # nothing still reported success (reproduced a third time by job
    # `38268849`). Asserted at the point where the emptiness is proven, and
    # paired with the control arm below so the verdict is shown to track the
    # observation rather than being uniformly negative.
    assert check_series(perturbed, "passed") == [False] * MAX_STEPS
    assert check_series(control, "passed") == [True] * MAX_STEPS
