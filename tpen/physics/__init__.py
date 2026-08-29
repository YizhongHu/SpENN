"""QMC physics namespace.

Import symbols from their defining submodules, for example::

    from tpen.physics.hamiltonian import HamiltonianTerm, LocalEnergyResult, local_energy
    from tpen.physics.kinetic import KineticEnergy
    from tpen.physics.potential import HarmonicTrap, ElectronElectronInteraction
    from tpen.physics.hooke import HookeSingletExact, HookeTripletExact
"""

from tpen.physics.terms import PHYSICAL_TERM_NAMES, summarize_physical_terms

__all__ = ["PHYSICAL_TERM_NAMES", "summarize_physical_terms"]
