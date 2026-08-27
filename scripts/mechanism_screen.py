"""Milestone-0 structural map for candidate chiral-bias mechanisms.

This script intentionally contains *no quantitative CISS/CPL ranking*.
Its purpose is only to demonstrate how a microscopic mechanism would have to
be mapped onto the derivative coefficients

    r_j^+ = d ln k_j^+ / dh,    r_j^- = d ln k_j^- / dh,
    A_j = r_j^+ - r_j^-,       F_j = (r_j^+ + r_j^-)/2,

before the general response identity can be applied.  Calling an entire
physical mechanism "thermodynamic" or "kinetic" is generally too coarse: a
specific microscopic implementation may populate the affinity sector A, the
barrier sector F, or both.

CISS
----
A CISS-mediated surface process cannot be assigned a universal A/F pair from
'CISS' alone.  Depending on the microscopic mechanism it may involve an
adsorption free-energy difference, a transient spin-dependent barrier, a
spin-selective electron-transfer rate, or combinations thereof.  The toy
stoichiometric map below makes only one robust point: selective adsorption
that transfers L/D between solution and a surface has zero projection on the
*total* L-D inventory.  A downstream inventory-changing or symmetry-amplifying
step is required to convert biased occupancy into bulk handedness.  That step
need not be autocatalytic.

CPL
---
Circularly polarized photochemistry can generate differential hazards without
an equilibrium ground-state splitting.  A strictly irreversible photolysis
channel, however, cannot be inserted naively into a reversible-pair entropy-
production bound because ln(J+/J-) diverges as J- -> 0.  A theorem-grade map
must either include the photon field/reservoir and a microscopic reverse
channel, or formulate the irreversible hazard process separately.
"""
from __future__ import annotations

import numpy as np

from pvchiral.bias import decompose, eps_channels


print("=" * 76)
print("Milestone 0: structural mapping only (no mechanism magnitude claims)")
print("=" * 76)

# Toy surface bookkeeping.  chi_surface marks where a hypothetical surface
# perturbation acts; total_excess counts molecular handedness irrespective of
# phase.  This does NOT claim that CISS is a conservative species-energy bias.
labels = ["L_sol", "D_sol", "L_ads", "D_ads"]
chi_surface = np.array([0.0, 0.0, 1.0, -1.0])
total_excess = np.array([1.0, -1.0, 1.0, -1.0])
steps = {
    "adsorb L: L_sol -> L_ads": np.array([-1.0, 0.0, 1.0, 0.0]),
    "adsorb D: D_sol -> D_ads": np.array([0.0, -1.0, 0.0, 1.0]),
    "create L_ads from achiral feed": np.array([0.0, 0.0, 1.0, 0.0]),
    "create D_ads from achiral feed": np.array([0.0, 0.0, 0.0, 1.0]),
}

print("\nToy surface stoichiometry (illustrative bookkeeping, not a CISS model):")
print(f"  species: {', '.join(labels)}")
print(f"  {'step':39} {'surface projection':>19} {'total L-D projection':>22}")
for name, nu in steps.items():
    q_surface = float(nu @ chi_surface)
    q_total = float(nu @ total_excess)
    print(f"  {name:39} {q_surface:+19.1f} {q_total:+22.1f}")

print("\nInterpretation:")
print("  * adsorption can be surface-selective while leaving total L-D inventory unchanged;")
print("  * a downstream reaction, trapping, destruction, precipitation, crystallization,")
print("    or amplification step must project on total excess to alter bulk handedness;")
print("  * its microscopic rate response must be computed before assigning A and F.")

print("\nGeneric derivative-coefficient example:")
r_plus = np.array([0.8, -0.1])
r_minus = np.array([-0.2, -0.1])
A, F = decompose(r_plus, r_minus)
Jp, Jm = np.array([2.0, 1.5]), np.array([1.0, 1.0])
P = np.array([1.0, -1.0])
parts = eps_channels(P, A, F, Jp, Jm, N=1.0, h=1e-3)
print(f"  A = {A}")
print(f"  F = {F}")
print(f"  affinity contribution = {parts['affinity']:+.4e}")
print(f"  barrier contribution  = {parts['barrier']:+.4e}")
print("\nNo CISS energy, CPL anisotropy, alignment factor, or density derating is")
print("used here.  Those belong to a later mechanism-specific mapping milestone.")
