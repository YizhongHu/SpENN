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

The Event Clock migration preserves that boundary rather than moving it (posture
A), so the contract survives and is DECLARED here instead of being fixed by
relocating the observer. Two tests do that job together:

`test_gradient_stats_observes_the_gradients_the_update_consumed` is the contract
itself, and fails the moment ``zero_grad`` moves after the update.

`test_clearing_gradients_after_the_update_empties_every_gradient_metric` gives
that first test teeth, by reproducing the probe's arm B and pinning that the
move really does zero the metric while leaving training untouched. Without it
the contract test could pass vacuously.

The perturbation is applied through the optimizer rather than by editing
`tpen.training.trainer`, so the trainer under test is the shipped one.
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


def _fit(tmp_path: Path, *, optimizer_class: type[torch.optim.Optimizer]) -> RecordingLogger:
    """Run the real tiny-TPEN loop with `GradientStats` attached."""

    logger = RecordingLogger()
    callback = GradientStats(fail_fast=False)
    context = make_run_context(tmp_path, callbacks=[callback], loggers=[logger])
    model = build_tiny_spenn()
    VMCTrainer(max_steps=MAX_STEPS, log_every_n_steps=1).fit(
        model=model,
        sampler=build_tiny_sampler(),
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
        # The defect that makes this invisible in production: a check observing
        # nothing still reports success. Filed separately as `933b5f78`; pinned
        # here, not fixed here.
        assert metrics["passed"] is True

    # The control the Cannon probe used. Training itself is untouched by the
    # move -- only what the observer can see changes -- so this must stay
    # non-zero while every metric above is empty.
    train_records = logger.by_namespace("train")
    assert len(train_records) == MAX_STEPS
    for record in train_records:
        assert record.metrics["grad_norm"] != 0.0
