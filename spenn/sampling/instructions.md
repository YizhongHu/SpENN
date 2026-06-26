# Sampling Subdirectory Instructions

This file describes the responsibilities and expected interfaces for the `sampling/` package.

The sampling package owns **Monte Carlo walkers and proposal/acceptance logic**. It should not know about Specht irreps, Hamiltonian internals, local-energy formulas, Hydra, or WandB. It should treat the wavefunction model as a callable black box that returns `WavefunctionOutput`.

Recommended structure:

```text
sampling/
  __init__.py
  walkers.py
  metropolis.py
  moves.py
  equilibrate.py
```

---

## 1. Core mathematical target

VMC samples configurations from

\[
\pi_\theta(X)\propto |\psi_\theta(X)|^2.
\]

If the model returns

\[
f(X)=\log |\psi_\theta(X)|,
\]

then the log target density is

\[
\log \pi_\theta(X)=2f(X)+\text{constant}.
\]

A Metropolis-Hastings proposal

\[
X\to X'
\]

is accepted with probability

\[
\alpha(X\to X')=
\min\left(1,
\exp\left[2f(X')-2f(X)+\log q(X|X')-\log q(X'|X)\right]
\right).
\]

For symmetric random-walk proposals,

\[
q(X'|X)=q(X|X'),
\]

so

\[
\log \alpha = 2(f(X')-f(X)).
\]

---

## 2. Expected wavefunction interface

The sampler should call:

```python
out = model(batch)
```

where `out` contains:

```python
out.logabs  # Tensor shape [batch]
out.sign    # Tensor shape [batch], optional for sampler
out.phase   # Optional
```

The sampler generally only needs `logabs` for acceptance decisions.

It should not inspect internal model features.

---

## 3. `sampling/walkers.py`

Responsibilities:

- Store current walker configurations.
- Store cached model evaluations used by samplers.
- Provide utilities for moving between walker objects and model batch objects.

Suggested dataclass:

```python
from dataclasses import dataclass, field
import torch

@dataclass
class Walkers:
    positions: torch.Tensor  # [n_walkers, n_electrons, dim]
    logabs: torch.Tensor | None = None  # [n_walkers]
    sign: torch.Tensor | None = None    # [n_walkers]
    aux: dict = field(default_factory=dict)

    def to(self, device=None, dtype=None): ...
    def clone(self): ...
    def detach(self): ...
```

Recommended methods:

```python
def make_batch(self):
    """Return an ElectronBatch/model input object."""
    ...

def update_cache(self, model):
    """Evaluate model and store logabs/sign/phase."""
    ...
```

Important:

- Sampler steps should keep `walkers.logabs` up to date.
- If positions are changed, cached logabs/sign must be recomputed or invalidated.
- Use detached tensors for persistent walker state; do not accidentally retain computation graphs across MC steps.

---

## 4. `sampling/moves.py`

Responsibilities:

- Define proposal move kernels.
- Keep proposal generation separate from acceptance logic.

Suggested base interface:

```python
class Move:
    def propose(self, walkers, model=None):
        """
        Return proposed_positions and log proposal ratio.

        log_q_reverse_minus_forward = log q(X | X') - log q(X' | X)
        shape: [n_walkers]
        """
        ...
```

### Random-walk Gaussian move

Proposal:

\[
X'=X+\sigma \xi,
\qquad \xi\sim\mathcal N(0,I).
\]

This proposal is symmetric, so

\[
\log q(X|X')-\log q(X'|X)=0.
\]

Suggested class:

```python
class GaussianMove(Move):
    def __init__(self, step_size: float, move_all: bool = True):
        self.step_size = step_size
        self.move_all = move_all

    def propose(self, walkers, model=None):
        ...
```

Options:

- `move_all=True`: move all electrons at once.
- `move_all=False`: randomly select one electron per walker.

For early implementation, moving all electrons is simpler. Single-electron moves may improve acceptance later.

---

## 5. `sampling/metropolis.py`

Responsibilities:

- Implement symmetric Metropolis-Hastings using a `Move` object.
- Own acceptance/rejection logic.
- Return updated walkers and metrics.

Suggested class:

```python
class MetropolisSampler:
    def __init__(self, move, n_steps: int = 10):
        self.move = move
        self.n_steps = n_steps

    @torch.no_grad()
    def step(self, model, walkers):
        ...

    @torch.no_grad()
    def sample(self, model, walkers, n_steps: int | None = None):
        ...
```

Acceptance logic:

```python
proposed_positions, log_q_ratio = self.move.propose(walkers, model=model)
proposed_batch = walkers.with_positions(proposed_positions).make_batch()
proposed_out = model(proposed_batch)

log_accept = 2 * (proposed_out.logabs - walkers.logabs) + log_q_ratio
accept = torch.log(torch.rand_like(log_accept)) < torch.clamp(log_accept, max=0.0)
```

Then update positions and cache only for accepted walkers.

Important implementation notes:

- Use `torch.no_grad()` for ordinary Metropolis sampling.
- Make sure cached `walkers.logabs` exists before stepping.
- Avoid building computation graphs during sampling unless the sampler explicitly needs gradients, as MALA does.
- Return acceptance rate.

Suggested metrics:

```python
{
    "acceptance_rate": accept.float().mean(),
    "mean_logabs": walkers.logabs.mean(),
    "step_size": self.move.step_size,
}
```

---

## 6. `sampling/equilibrate.py`

Responsibilities:

- Provide burn-in/equilibration utilities.
- Optionally adapt proposal step size during burn-in.
- Do not perform gradient descent or model optimization.

Suggested functions:

```python
def equilibrate(model, sampler, walkers, n_steps: int, target_acceptance: float | None = None):
    ...
```

Step-size adaptation, if implemented, should be simple:

```python
if acceptance_rate < target_low:
    step_size *= 0.9
elif acceptance_rate > target_high:
    step_size *= 1.1
```

Only adapt during burn-in, not during production sampling unless explicitly configured.

---

## 7. `sampling/__init__.py`

Export stable public objects:

```python
from .walkers import Walkers
from .moves import GaussianMove
from .metropolis import MetropolisSampler
from .equilibrate import equilibrate
```

---

## 8. Interaction with training loop

The trainer should own the high-level loop:

```python
walkers = sampler.sample(model, walkers, n_steps=n_mc_steps)
local_e, aux = hamiltonian.local_energy(model, walkers.make_batch())
loss, metrics = loss_fn(model, local_e, walkers)
loss.backward()
optimizer.step()
```

The sampler should not compute local energy.

The Hamiltonian should not move walkers.

The model should not know about the sampler.

---

## 9. Distributed and parallel sampling

The sampling code should support batched walkers from day one:

```python
positions.shape == [n_walkers, n_electrons, dim]
```

For multi-GPU DDP:

- Each rank owns its own walkers.
- Each rank samples independently.
- Gradients are synchronized by DDP during training.
- Metrics such as acceptance rate and energy should be averaged across ranks by training utilities, not inside sampler core logic.

Do not add distributed dependencies inside basic sampler code. Instead expose metrics and let training orchestration reduce them across ranks.

---

## 10. Required tests for `sampling/`

### Walker tests

- `Walkers.clone()` creates independent storage.
- `Walkers.detach()` removes graphs.
- `Walkers.to(device, dtype)` moves all tensors.
- Cache update stores `logabs` with shape `[n_walkers]`.

### Move tests

- Gaussian move preserves shape.
- Gaussian move returns zero proposal ratio.
- Single-electron mode changes only one electron per walker, if implemented.

### Metropolis tests

Use a simple fake model:

\[
\log |\psi(X)|=-\alpha\|X\|^2.
\]

Test:

- acceptance mask has shape `[n_walkers]`,
- positions update only when accepted,
- cached logabs matches new positions after step,
- acceptance rate is between 0 and 1.

### Equilibration tests

- Burn-in returns walkers with valid cache.
- Step-size adaptation changes step size in expected direction.

---

## 11. Practical implementation order

Implement in this order:

1. `Walkers`
2. `GaussianMove`
3. `MetropolisSampler`
4. sampler tests with fake Gaussian model
5. `equilibrate`
6. distributed-friendly metrics

Do not start with MALA. Add gradient-informed proposal kernels only after the random-walk Metropolis sampler is robust.
