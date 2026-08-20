"""General weak-bias response channels for reversible reaction networks.

This module uses *response coefficients* rather than finite log-rate shifts.
For a dimensionless weak perturbation ``h`` define, for each reversible
reaction pair j,

    r_j^+ = d ln k_j^+ / dh,
    r_j^- = d ln k_j^- / dh.

The coefficients decompose into an affinity/entropic sector and a
kinetic/barrier sector,

    A_j = r_j^+ - r_j^-,
    F_j = (r_j^+ + r_j^-)/2.

To first order in h, the projected drift shift is exactly

    delta eps = (h/N) sum_j P_j [ A_j a_j/2 + F_j j_j ],

where ``a_j = J_j^+ + J_j^-`` is the one-way activity and
``j_j = J_j^+ - J_j^-`` is the net current.  The affinity sector survives at
equilibrium.  The barrier sector is proportional to current and therefore
vanishes at detailed balance.

For the species-free-energy/PVED parameterization with ``h = g``, local
detailed balance gives ``A_j = B_j`` while a Brønsted partition ``alpha_j``
gives ``F_j = B_j (alpha_j - 1/2)``.  The perturbation amplitude ``g`` is kept
explicit and is never absorbed into A or F.

The bounds implemented here are channel-wise Cauchy--Schwarz bounds.  The
barrier-sector bound involves entropy production; this means that a specified
dissipation budget constrains barrier response.  Entropy production itself has
no universal finite ceiling without additional physical/resource constraints.
"""
from __future__ import annotations

import numpy as np


def decompose(r_plus, r_minus):
    """Return derivative coefficients ``(A, F)`` from ``(r_plus, r_minus)``.

    ``A = r_plus - r_minus`` is the affinity/entropic response coefficient and
    ``F = (r_plus + r_minus)/2`` is the kinetic/barrier response coefficient.
    """
    r_plus, r_minus = np.asarray(r_plus, float), np.asarray(r_minus, float)
    return r_plus - r_minus, 0.5 * (r_plus + r_minus)


def ldb_response_coefficients(B, alpha):
    """PVED/species-energy response coefficients, without the amplitude ``g``.

    For ``h=g`` and ``d ln(k+/k-)/dg = B``, the Brønsted parameterization gives
    ``A=B`` and ``F=B(alpha-1/2)``.
    """
    B, alpha = np.asarray(B, float), np.asarray(alpha, float)
    return B, B * (alpha - 0.5)


def responses_from_ldb(B, alpha):
    """Backward-compatible alias for :func:`ldb_response_coefficients`.

    Version 1.3-dev deliberately removes the old ``g`` argument so response
    derivatives cannot be confused with finite first-order log-rate shifts.
    """
    return ldb_response_coefficients(B, alpha)


def eps_from_responses(P, r_plus, r_minus, Jp, Jm, N, h=1.0):
    """Direct first-order projected drift shift for perturbation amplitude ``h``."""
    P = np.asarray(P, float)
    r_plus, r_minus = np.asarray(r_plus, float), np.asarray(r_minus, float)
    Jp, Jm = np.asarray(Jp, float), np.asarray(Jm, float)
    return float(h) * float(np.sum(P * (r_plus * Jp - r_minus * Jm))) / N


def eps_channels(P, A, F, Jp, Jm, N, h=1.0):
    """Exact first-order identity resolved into affinity and barrier sectors."""
    P, A, F = (np.asarray(x, float) for x in (P, A, F))
    Jp, Jm = np.asarray(Jp, float), np.asarray(Jm, float)
    activity, current = Jp + Jm, Jp - Jm
    affinity_term = 0.5 * float(h) * float(np.sum(P * A * activity)) / N
    barrier_term = float(h) * float(np.sum(P * F * current)) / N
    return {
        "affinity": affinity_term,
        "barrier": barrier_term,
        "total": affinity_term + barrier_term,
    }


def Lambda_A(A, Jp, Jm, s):
    """Affinity-response-weighted activity per chiral unit."""
    A = np.asarray(A, float)
    activity = np.asarray(Jp, float) + np.asarray(Jm, float)
    return float(np.sum(A**2 * activity)) / (2 * s)


def Lambda_S(S, Jp, Jm, s):
    """Deprecated compatibility alias for :func:`Lambda_A`."""
    return Lambda_A(S, Jp, Jm, s)


def affinity(Jp, Jm):
    """Edge affinity ``ln(J+/J-)``; infinite if one one-way flux is zero."""
    Jp, Jm = np.asarray(Jp, float), np.asarray(Jm, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(Jp / Jm)


def entropy_production(Jp, Jm):
    """Edge entropy production ``sigma_j = j_j ln(J+/J-) >= 0``.

    The reversible-pair expression diverges in a strictly irreversible limit.
    A finite dissipation bound therefore requires positive forward and reverse
    one-way fluxes, or an explicit microscopic reservoir model that restores a
    reversible description.
    """
    Jp, Jm = np.asarray(Jp, float), np.asarray(Jm, float)
    current, edge_affinity = Jp - Jm, affinity(Jp, Jm)
    out = current * edge_affinity
    return np.where(np.isfinite(out), out, np.inf)


def bound_affinity(Theta, s, A, Jp, Jm, h=1.0):
    """``|delta eps_affinity| <= |h| sqrt(s Theta Lambda_A)``."""
    return abs(float(h)) * np.sqrt(s * Theta * Lambda_A(A, Jp, Jm, s))


def bound_thermodynamic(Theta, s, S, Jp, Jm, h=1.0):
    """Compatibility alias for :func:`bound_affinity`."""
    return bound_affinity(Theta, s, S, Jp, Jm, h=h)


def bound_barrier(Theta, F, Jp, Jm, h=1.0):
    """Dissipation-conditioned barrier-sector response bound.

    ``|delta eps_barrier| <= |h| sqrt(Theta sum_j F_j^2 sigma_j)``.

    This is a bound in terms of the *actual* entropy production.  It does not
    imply a universal ceiling on response because entropy production itself is
    unbounded absent an independently imposed power/resource budget.
    """
    F = np.asarray(F, float)
    return abs(float(h)) * np.sqrt(
        Theta * float(np.sum(F**2 * entropy_production(Jp, Jm)))
    )


def bound_kinetic(Theta, K, Jp, Jm, h=1.0):
    """Compatibility alias for :func:`bound_barrier`."""
    return bound_barrier(Theta, K, Jp, Jm, h=h)


def current_dissipation_slack(Jp, Jm):
    """Return ``a*sigma/2 - j^2`` for finite reversible channels.

    Non-negativity follows from ``A >= 2 tanh(A/2)`` for ``A>=0`` (and its
    odd extension), with ``j=a tanh(A/2)``.
    """
    Jp, Jm = np.asarray(Jp, float), np.asarray(Jm, float)
    activity, current = Jp + Jm, Jp - Jm
    return activity * entropy_production(Jp, Jm) / 2 - current**2


def current_fraction(Jp, Jm):
    """Return edge current fractions ``rho_j=(Jp-Jm)/(Jp+Jm)``.

    Edges with zero total activity are assigned ``rho_j=0`` because they do
    not contribute to any response or noise sum.
    """
    Jp, Jm = np.asarray(Jp, float), np.asarray(Jm, float)
    a = Jp + Jm
    return np.divide(Jp - Jm, a, out=np.zeros_like(a), where=a > 0)


def effective_response_coefficient(A, F, Jp, Jm):
    """Reaction-pair response coefficient ``Q_j=A_j+2 F_j rho_j``.

    With ``rho_j=(Jp-Jm)/(Jp+Jm)``, the exact first-order drift contribution
    of edge ``j`` can be written as ``h P_j a_j Q_j/(2N)``.  Pairing the
    forward and reverse channels before applying Cauchy--Schwarz retains the
    cancellation of barrier response near detailed balance.
    """
    A, F = np.asarray(A, float), np.asarray(F, float)
    return A + 2.0 * F * current_fraction(Jp, Jm)


def Lambda_response(A, F, Jp, Jm, s):
    r"""Response-weighted paired activity ``Lambda_resp``.

    .. math::

        \Lambda_{\rm resp}=\frac{1}{2s}\sum_j a_j
        \left(A_j+2F_j\rho_j\right)^2.

    It has units of inverse time when one-way fluxes are densities per time.
    Unlike the affinity-only ``Lambda_A``, it contains the *actual operating
    point* through the current fractions ``rho_j`` and is valid for arbitrary
    finite derivative coefficients ``A_j`` and ``F_j``.
    """
    if s <= 0:
        raise ValueError("s must be positive")
    a = np.asarray(Jp, float) + np.asarray(Jm, float)
    Q = effective_response_coefficient(A, F, Jp, Jm)
    return float(np.sum(a * Q**2)) / (2.0 * float(s))


def bound_total_response(Theta, s, A, F, Jp, Jm, h=1.0):
    """Sharp reaction-pair Cauchy bound on the complete drift response.

    For the projected chemical-Langevin noise strength ``Theta``, the exact
    first-order drift shift obeys

    ``|delta eps| <= |h| sqrt(s Theta Lambda_response)``.

    The bound requires no convex-barrier assumption.  Equality in the
    Cauchy--Schwarz step is possible when the projected stoichiometric weights
    are proportional to ``Q_j=A_j+2F_j rho_j`` on active reactions.
    """
    if Theta < 0:
        raise ValueError("Theta must be nonnegative")
    return abs(float(h)) * np.sqrt(
        float(s) * float(Theta) * Lambda_response(A, F, Jp, Jm, s)
    )


def directed_fisher_information_rate(r_plus, r_minus, Jp, Jm):
    """Poisson path Fisher-information rate for directed reaction counts.

    ``I_h = sum_j [(r_j+)^2 J_j+ + (r_j-)^2 J_j-]``.

    This is the information rate entering a generic pathwise Cramer--Rao
    response bound.  The paired chemical bound is never weaker because
    forward/reverse cancellation gives ``s Lambda_resp <= 2 I_h``.
    """
    rp, rm = np.asarray(r_plus, float), np.asarray(r_minus, float)
    Jp, Jm = np.asarray(Jp, float), np.asarray(Jm, float)
    return float(np.sum(rp**2 * Jp + rm**2 * Jm))


def paired_fisher_slack(r_plus, r_minus, Jp, Jm, s=1.0):
    """Return ``2 I_h - s Lambda_resp >= 0``.

    The inequality is an edgewise Cauchy contraction that quantifies how much
    response information is lost when opposite directed reaction events have
    the same stoichiometric projection and therefore partially cancel in the
    deterministic drift.
    """
    A, F = decompose(r_plus, r_minus)
    info = directed_fisher_information_rate(r_plus, r_minus, Jp, Jm)
    return 2.0 * info - float(s) * Lambda_response(A, F, Jp, Jm, s)


def pved_response_activity(B, alpha, Jp, Jm, s):
    """State-dependent response activity for the PVED specialization.

    Here ``A=B`` and ``F=B(alpha-1/2)``.  On the convex transition-state
    domain ``0<=alpha<=1`` this obeys ``Lambda_resp <= 4 Lambda`` where
    ``Lambda=sum B^2 a/(2s)`` is the original chirality-weighted activity.
    At detailed balance ``Lambda_resp=Lambda`` independently of ``alpha``.
    """
    B, alpha = np.asarray(B, float), np.asarray(alpha, float)
    return Lambda_response(B, B * (alpha - 0.5), Jp, Jm, s)
