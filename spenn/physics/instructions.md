# Physics Subdirectory Instructions

This file describes the responsibilities and expected interfaces for the `physics/` package.

The physics package should contain **Hamiltonian and local-energy logic only**. It should not know about Specht irreps, SpechtMP internals, Monte Carlo proposal rules, Hydra, or WandB. The physics code should treat the wavefunction model as a callable black box that returns a wavefunction value in a stable representation.

Recommended structure:

```text
physics/
  __init__.py
  systems.py
  hamiltonian.py
  local_energy.py
  kinetic.py
  potential.py
  cusp.py
```

---

## 1. Core mathematical target

For a wavefunction \(\psi_\theta(X)\), the variational Monte Carlo objective estimates

\[
E(\theta)=\frac{\langle \psi_\theta|H|\psi_\theta\rangle}{\langle \psi_\theta|\psi_\theta\rangle}
=\mathbb E_{X\sim |\psi_\theta|^2}\left[E_L(X)\right],
\]

where the local energy is

\[
E_L(X)=\frac{H\psi_\theta(X)}{\psi_\theta(X)}.
\]

For a nonrelativistic electronic Hamiltonian in atomic units,

\[
H=-\frac12\sum_{i=1}^{N_e}\nabla_i^2+V(X).
\]

Thus

\[
E_L(X)=-\frac12\sum_i \frac{\nabla_i^2\psi(X)}{\psi(X)}+V(X).
\]

If the model returns

\[
f(X)=\log |\psi(X)|,
\]

then inside a fixed nodal pocket,

\[
\frac{\nabla_i^2\psi}{\psi}
=
\nabla_i^2 f + \|\nabla_i f\|^2.
\]

So the kinetic contribution is

\[
T_L(X)
=
-\frac12\sum_i
\left(
\nabla_i^2 f(X)+\|\nabla_i f(X)\|^2
\right).
\]

The sign/phase matters for nodes and acceptance ratios, but the kinetic formula can be implemented from derivatives of `logabs` away from nodes.

---

## 2. Expected wavefunction interface

The physics package should assume the model has an interface like:

```python
out = model(batch)
```

where `out` is a dataclass-like object with at least:

```python
out.logabs  # Tensor shape [batch]
out.sign    # Tensor shape [batch], for real wavefunctions
out.phase   # Optional Tensor shape [batch], for complex wavefunctions later
out.aux     # Optional dict
```

The physics code should not depend on how `logabs` and `sign` were produced.

The input `batch` should contain electron positions:

```python
batch.positions  # Tensor shape [batch, n_electrons, dim]
```

Potentially later:

```python
batch.spins
batch.nuclei
batch.charges
```

---

## 3. `physics/systems.py`

Responsibilities:

- Define static physical systems.
- Store nuclear positions, charges, electron count, spin partition, and dimension.
- Provide constructors for simple test systems.
- Avoid model-specific or sampler-specific logic.

Suggested dataclasses:

```python
from dataclasses import dataclass
import torch

@dataclass
class MolecularSystem:
    nuclear_positions: torch.Tensor  # [n_nuclei, dim]
    nuclear_charges: torch.Tensor    # [n_nuclei]
    n_electrons: int
    n_up: int | None = None
    n_down: int | None = None
    dim: int = 3

    @property
    def device(self):
        return self.nuclear_positions.device

    @property
    def dtype(self):
        return self.nuclear_positions.dtype
```

Suggested helpers:

```python
def make_hydrogen_atom(dtype=torch.float64, device="cpu") -> MolecularSystem: ...
def make_helium_atom(dtype=torch.float64, device="cpu") -> MolecularSystem: ...
def make_h2(distance: float, dtype=torch.float64, device="cpu") -> MolecularSystem: ...
```

Keep all units in atomic units unless explicitly stated otherwise.

---

## 4. `physics/potential.py`

Responsibilities:

- Compute potential energy for batches of electron configurations.
- Implement electron-electron, electron-nucleus, and nucleus-nucleus terms.
- Avoid kinetic/autograd logic.

For positions

```python
x = batch.positions  # [B, N, D]
R = system.nuclear_positions  # [A, D]
Z = system.nuclear_charges    # [A]
```

potential terms are:

### Electron-electron

\[
V_{ee}(X)=\sum_{i<j}\frac{1}{\|r_i-r_j\|}.
\]

### Electron-nucleus

\[
V_{en}(X)=-\sum_i\sum_A\frac{Z_A}{\|r_i-R_A\|}.
\]

### Nucleus-nucleus

\[
V_{nn}=\sum_{A<B}\frac{Z_AZ_B}{\|R_A-R_B\|}.
\]

Total:

\[
V=V_{ee}+V_{en}+V_{nn}.
\]

Suggested class:

```python
class CoulombPotential(torch.nn.Module):
    def __init__(self, system: MolecularSystem, eps: float = 1e-12):
        super().__init__()
        self.system = system
        self.eps = eps

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Return potential energy with shape [batch]."""
        ...
```

Implementation notes:

- Add small `eps` inside distances only for numerical safety.
- Use masking or upper-triangular indexing for electron-electron and nucleus-nucleus terms.
- The returned tensor must have shape `[batch]`.
- Keep dtype/device consistent with input positions.

---

## 5. `physics/kinetic.py`

Responsibilities:

- Compute kinetic local energy using PyTorch autograd.
- Provide a clean interface so optimized implementations can replace the naive version later.

For

\[
f(X)=\log |\psi(X)|,
\]

compute

\[
T_L(X)
=-\frac12\sum_{i,d}
\left(
\partial_{x_{id}}^2 f(X)
+
(\partial_{x_{id}}f(X))^2
\right).
\]

Suggested interface:

```python
class KineticEnergy(torch.nn.Module):
    def forward(self, model, positions: torch.Tensor, batch=None) -> torch.Tensor:
        """Return kinetic local energy with shape [batch]."""
        ...
```

Naive autograd algorithm:

1. Clone positions and set `requires_grad_(True)`.
2. Build/update a batch object with these positions.
3. Evaluate `out = model(batch)`.
4. Let `f = out.logabs`, shape `[B]`.
5. Compute gradient:

```python
grad = torch.autograd.grad(f.sum(), positions, create_graph=True)[0]
```

6. Compute Laplacian by looping over electron-coordinate dimensions:

```python
lap = torch.zeros(B, dtype=positions.dtype, device=positions.device)
flat_grad = grad.reshape(B, -1)
flat_pos = positions.reshape(B, -1)
for a in range(flat_pos.shape[1]):
    second = torch.autograd.grad(
        flat_grad[:, a].sum(),
        positions,
        create_graph=True,
        retain_graph=True,
    )[0].reshape(B, -1)[:, a]
    lap = lap + second
```

7. Compute:

```python
grad_sq = (grad ** 2).sum(dim=(-1, -2))
kinetic = -0.5 * (lap + grad_sq)
```

Important notes:

- This naive method is expensive but correct for early development.
- It should be easy to replace later with vectorized Hessian, functorch/vmap, forward-mode, finite differences for debugging, or custom formulas.
- Use `torch.float64` for early physics tests.
- Do not detach local energy inside the Hamiltonian. Loss code decides how to handle gradients.

---

## 6. `physics/local_energy.py`

Responsibilities:

- Combine kinetic and potential terms.
- Provide a single entry point for VMC loss.
- Return diagnostics if useful.

Suggested class:

```python
class LocalEnergy(torch.nn.Module):
    def __init__(self, kinetic, potential):
        super().__init__()
        self.kinetic = kinetic
        self.potential = potential

    def forward(self, model, batch):
        positions = batch.positions
        kinetic = self.kinetic(model, positions, batch=batch)
        potential = self.potential(positions)
        local_e = kinetic + potential
        return local_e, {"kinetic": kinetic, "potential": potential}
```

Shape convention:

```python
local_e.shape == [batch]
kinetic.shape == [batch]
potential.shape == [batch]
```

---

## 7. `physics/hamiltonian.py`

Responsibilities:

- Provide high-level Hamiltonian objects.
- Own system, potential, kinetic, and local-energy modules.
- Expose `local_energy(model, batch)`.

Suggested class:

```python
class ElectronicHamiltonian(torch.nn.Module):
    def __init__(self, system, kinetic=None, potential=None):
        super().__init__()
        self.system = system
        self.kinetic = kinetic or KineticEnergy()
        self.potential = potential or CoulombPotential(system)
        self.local_energy_module = LocalEnergy(self.kinetic, self.potential)

    def local_energy(self, model, batch):
        return self.local_energy_module(model, batch)
```

The training loop should call:

```python
local_e, aux = hamiltonian.local_energy(model, walkers.as_batch())
```

The Hamiltonian must not mutate walkers or run MC steps.

---

## 8. `physics/cusp.py`

Responsibilities:

- Implement cusp factors or cusp-related helper functions.
- For now, prioritize electron-electron cusp.
- Keep cusp logic modular so it can be attached to the wavefunction or used as a feature.

A simple electron-electron Jastrow/cusp factor can be added to the log amplitude:

\[
\log |\psi(X)| = \log |\psi_{NN}(X)| + J_{ee}(X).
\]

A common schematic form is

\[
J_{ee}(X)=\sum_{i<j} u(r_{ij}).
\]

The exact cusp coefficient depends on spin and convention. For early code, implement the class but allow coefficients to be configured explicitly.

Suggested class:

```python
class ElectronElectronCusp(torch.nn.Module):
    def __init__(self, same_spin_coeff: float, opposite_spin_coeff: float | None = None):
        super().__init__()
        self.same_spin_coeff = same_spin_coeff
        self.opposite_spin_coeff = opposite_spin_coeff

    def forward(self, positions: torch.Tensor, spins=None) -> torch.Tensor:
        """Return cusp/Jastrow log factor with shape [batch]."""
        ...
```

The cusp module may also live under `nn/encoding/cusp.py` if it is used directly in the model. The `physics/cusp.py` version should contain reusable formulas and tests.

---

## 9. `physics/__init__.py`

Export stable public objects:

```python
from .systems import MolecularSystem
from .hamiltonian import ElectronicHamiltonian
from .local_energy import LocalEnergy
from .kinetic import KineticEnergy
from .potential import CoulombPotential
from .cusp import ElectronElectronCusp
```

---

## 10. Required tests for `physics/`

### Potential tests

- Two electrons at known separation should produce known `V_ee`.
- One electron near one nucleus should produce known `V_en`.
- Two nuclei should produce known `V_nn`.
- Batch shape should be preserved.

### Kinetic tests

Use analytic wavefunctions with known Laplacians.

Example:

\[
f(X)=-\alpha\sum_i\|r_i\|^2.
\]

Then

\[
\nabla_i f=-2\alpha r_i,
\]

\[
\nabla_i^2 f=-2\alpha D.
\]

So

\[
T_L=-\frac12\sum_i\left(-2\alpha D+4\alpha^2\|r_i\|^2\right).
\]

Compare autograd result against this formula.

### Local energy tests

- Check shape `[batch]`.
- Check dtype/device propagation.
- Check no unexpected detach.
- Check gradients can flow to model parameters.

---

## 11. Performance notes

- Autograd Laplacian is the likely bottleneck.
- Keep the kinetic implementation isolated so it can be replaced.
- Use `torch.float64` for correctness tests.
- Add profiling hooks later, but keep this package mathematically simple first.
