"""Ginzburg criterion for the driven chiral bifurcation."""
from __future__ import annotations

import numpy as np


def a_ginzburg(theta: float, b: float, D: float, d: int = 3) -> float:
    """Return the Ginzburg scale.

    For Model-A dynamics with free-energy density
    ``-a eta^2/2 + b eta^4/4 + D |grad eta|^2/2`` and white-noise strength
    ``theta``, the Gaussian approximation is self-consistent for
    ``a >> a_Gi``, where

        a_Gi = (theta b / D^(d/2))^(2/(4-d)),  d < 4.
    """
    if not 0 < d < 4:
        raise ValueError("the closed Ginzburg estimate requires 0 < d < 4")
    if theta < 0 or b <= 0 or D <= 0:
        raise ValueError("theta >= 0, b > 0, and D > 0 are required")
    return float((theta * b / D ** (d / 2.0)) ** (2.0 / (4.0 - d)))


def gi_closed_form(kappa_m1: float, N_D: float) -> float:
    """Return ``a_Gi/b`` for the Frank scaling in d=3.

    With ``theta=X/rho``, ``b=(kappa-1)X`` and
    ``N_D=rho(D/b)^(3/2)``, direct substitution gives

        a_Gi / b = [1 / ((kappa-1) N_D)]^2.

    The factor of two present in package v1.1.0 was erroneous.
    """
    if kappa_m1 <= 0 or N_D <= 0:
        raise ValueError("kappa_m1 and N_D must be positive")
    return float((1.0 / (kappa_m1 * N_D)) ** 2)
