# Hooke Multibody Report

## Purpose

Test whether the SpENN-QMC stack can run a genuinely multielectron Hooke trap
with VMC energy minimization only. The initial scaffold targets `N=3`.

## Hamiltonian

The system is an `N`-electron harmonic Coulomb trap:

```text
H = sum_i (-1/2 nabla_i^2 + 1/2 omega^2 |r_i|^2) + sum_{i<j} 1 / |r_i-r_j|
```

## Model

The trial form is `psi_theta(R) = exp(J_ee(R)) psi_SpENN(R)`. The configured
cusp slopes are `1/4` for same-spin pairs and `1/2` for opposite-spin pairs.
The initial readout is Pfaffian-based and uses particle-token antisymmetry.

## Reference

No numeric multibody reference is configured yet. SpENN is not trained against
any exact or reference wavefunction. Energy tables should be interpreted as VMC
estimates until an independent reference pipeline is added.

## Reproduction

See `README.md` for CPU, GPU, plotting, and spin-scan commands.

## Known Limitations

The scaffold is intentionally small. Pfaffian evaluation, local-energy second
derivatives, and Specht tuple tensors become expensive quickly as `N`, channel
counts, or Specht order increase.
