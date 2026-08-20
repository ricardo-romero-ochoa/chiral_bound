"""Conditional kinetic ceilings for the chirality-weighted activity."""
from __future__ import annotations


def lambda_kinetic_ceiling(k_max: float, z: float, B_max: float = 1.0) -> float:
    """Conditional bound on Lambda.

    If total chirality-changing one-way activity satisfies
    ``sum_j a_j <= z k_max s`` and ``|B_j| <= B_max``, then

        Lambda <= (z/2) B_max^2 k_max.

    This is not a thermodynamic or topology-independent theorem; ``z`` and
    ``k_max`` must be supplied as physical kinetic constraints.
    """
    if k_max < 0 or z < 0 or B_max < 0:
        raise ValueError("k_max, z, and B_max must be nonnegative")
    return float(0.5 * z * B_max**2 * k_max)
