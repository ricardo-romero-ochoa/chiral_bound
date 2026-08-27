"""Path-space information bounds for stochastic symmetry selection.

The module separates two levels of description.

1. Generic exponential rate perturbations
       w_rho^h = w_rho^0 exp(h r_rho)
   admit finite-field path-relative-entropy ceilings controlled by the
   response-weighted integrated activity, sum r_rho^2 w_rho.

2. A symmetrically split parity-violating energy difference (PVED) is the
   chemically structured specialization r_{j,+}=B_j/2, r_{j,-}=-B_j/2.
   The response-weighted activity then becomes one quarter of the integrated
   chirality-weighted activity sum B_j^2 (w_{j,+}+w_{j,-}).

Data processing converts any path-KL ceiling into a rigorous bound on a binary
final event, such as the final sign of a chiral order parameter.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.optimize import brentq


def bernoulli_kl(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)) in nats."""
    p = float(p)
    q = float(q)
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("p and q must lie in [0,1]")
    if p == q:
        return 0.0
    if q == 0.0:
        return 0.0 if p == 0.0 else math.inf
    if q == 1.0:
        return 0.0 if p == 1.0 else math.inf
    out = 0.0
    if p > 0.0:
        out += p * math.log(p / q)
    if p < 1.0:
        out += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return float(out)


def bernoulli_probability_interval(q: float, kl_budget: float) -> Tuple[float, float]:
    """Exact data-processing interval {p: D_Ber(p||q) <= kl_budget}."""
    q = float(q)
    D = float(kl_budget)
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie strictly between 0 and 1")
    if D < 0.0:
        raise ValueError("kl_budget must be nonnegative")
    if D == 0.0:
        return q, q

    d0 = bernoulli_kl(0.0, q)
    d1 = bernoulli_kl(1.0, q)
    lo = 0.0 if D >= d0 else brentq(lambda p: bernoulli_kl(p, q) - D, 0.0, q)
    hi = 1.0 if D >= d1 else brentq(lambda p: bernoulli_kl(p, q) - D, q, 1.0)
    return float(lo), float(hi)


def favored_probability_upper_from_symmetric_baseline(kl_budget: float) -> float:
    """Upper bound on a favored binary event when the reference probability is 1/2."""
    return bernoulli_probability_interval(0.5, kl_budget)[1]


def wrong_probability_lower_from_reverse_kl(kl_budget: float) -> float:
    """Lower bound on wrong-sign probability from D(P_0 || P_h) <= C.

    The zero-field binary sign event is assumed exactly symmetric, p_0=1/2,
    while the field-favored probability under P_h is 1-q. Data processing gives

        D_Ber(1/2 || 1-q) <= C,

    which can be inverted in closed form.
    """
    C = float(kl_budget)
    if C < 0.0:
        raise ValueError("kl_budget must be nonnegative")
    x = math.exp(-2.0 * C)
    return float(0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - x))))


def required_kl_for_wrong_probability(q_star: float) -> float:
    """Minimum reverse-KL budget compatible with wrong probability <= q_star.

    Assumes the reference final sign is exactly symmetric (probability 1/2).
    """
    q = float(q_star)
    if not (0.0 < q <= 0.5):
        raise ValueError("q_star must lie in (0, 1/2]")
    return float(bernoulli_kl(0.5, 1.0 - q))


def _phi_forward(u):
    """Per-rate factor for D(P_h || P_0) when w_h/w_0=exp(u)."""
    u = np.asarray(u, dtype=float)
    return u + np.expm1(-u)


def _phi_reverse(u):
    """Per-rate factor for D(P_0 || P_h) when w_h/w_0=exp(u)."""
    u = np.asarray(u, dtype=float)
    return np.expm1(u) - u


def exponential_tilt_kl_rate(
    rates,
    response_coefficients,
    h: float,
    *,
    direction: str = "h||0",
) -> float:
    """Exact instantaneous CTMC KL rate for an exponential propensity tilt.

    Parameters
    ----------
    rates
        Propensities under the measure used in the expectation. For
        ``direction='h||0'`` these are biased rates; for ``'0||h'`` they are
        reference rates.
    response_coefficients
        Channel coefficients r_rho in w_rho^h / w_rho^0 = exp(h r_rho).
    h
        Dimensionless perturbation amplitude.
    direction
        Either ``'h||0'`` or ``'0||h'``.
    """
    w = np.asarray(rates, dtype=float)
    r = np.asarray(response_coefficients, dtype=float)
    if w.shape != r.shape:
        raise ValueError("rates and response_coefficients must have matching shapes")
    if np.any(w < 0):
        raise ValueError("rates must be nonnegative")
    u = float(h) * r
    if direction == "h||0":
        val = w * _phi_forward(u)
    elif direction == "0||h":
        val = w * _phi_reverse(u)
    else:
        raise ValueError("direction must be 'h||0' or '0||h'")
    return float(np.sum(val))


def response_activity_kl_ceiling(
    h: float,
    integrated_r2_activity: float,
    rmax: float,
) -> float:
    """Finite-field path-KL ceiling for an exponential rate perturbation.

    If A_r = E int dt sum_rho r_rho^2 w_rho is evaluated under the measure in
    the corresponding KL expectation, then

        D <= (h^2/2) exp(|h| rmax) A_r.

    This follows from exp(u)-1-u <= u^2 exp(|u|)/2 and its reverse analogue.
    """
    A = float(integrated_r2_activity)
    rmax = float(rmax)
    if A < 0.0 or rmax < 0.0:
        raise ValueError("activity and rmax must be nonnegative")
    hh = abs(float(h))
    return float(0.5 * hh * hh * math.exp(hh * rmax) * A)


def response_activity_weak_kl(h: float, integrated_r2_activity: float) -> float:
    """Leading O(h^2) path KL for an exponential rate perturbation."""
    A = float(integrated_r2_activity)
    if A < 0.0:
        raise ValueError("activity must be nonnegative")
    return float(0.5 * float(h) ** 2 * A)


def pved_directed_response_coefficients(B, alpha=0.5):
    """Directed log-rate response coefficients for PVED barrier partition alpha.

    r_{j,+}=alpha_j B_j and r_{j,-}=-(1-alpha_j) B_j.
    """
    B = np.asarray(B, dtype=float)
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0:
        a = np.full_like(B, float(a))
    if a.shape != B.shape:
        raise ValueError("alpha must be scalar or have the same shape as B")
    return a * B, -(1.0 - a) * B


def symmetric_pved_kl_rate(
    forward_rates,
    reverse_rates,
    B,
    g: float,
    *,
    direction: str = "h||0",
) -> float:
    """Exact instantaneous CTMC KL rate for symmetric PVED rate splitting."""
    fp = np.asarray(forward_rates, dtype=float)
    fm = np.asarray(reverse_rates, dtype=float)
    B = np.asarray(B, dtype=float)
    if fp.shape != fm.shape or fp.shape != B.shape:
        raise ValueError("forward_rates, reverse_rates and B must have matching shapes")
    rp, rm = pved_directed_response_coefficients(B, 0.5)
    rates = np.concatenate([fp, fm])
    coeff = np.concatenate([rp, rm])
    return exponential_tilt_kl_rate(rates, coeff, g, direction=direction)


def symmetric_pved_activity_kl_ceiling(
    g: float,
    integrated_b2_activity: float,
    Bmax: float,
) -> float:
    """Finite-h KL ceiling from integrated chirality-weighted activity.

    For alpha=1/2, A_r = (1/4) int dt sum_j B_j^2(a_j), hence

        D <= (g^2/8) exp(|g| Bmax/2) * A_chi.
    """
    A = float(integrated_b2_activity)
    if A < 0.0 or Bmax < 0.0:
        raise ValueError("activity and Bmax must be nonnegative")
    return response_activity_kl_ceiling(g, 0.25 * A, 0.5 * float(Bmax))


def convex_pved_activity_kl_ceiling(
    g: float,
    integrated_b2_activity: float,
    Bmax: float,
) -> float:
    """Alpha-independent PVED KL ceiling for the convex domain 0<=alpha<=1.

    Since |r_{j,+}|, |r_{j,-}| <= |B_j|, the response-weighted activity is no
    larger than the chirality-weighted activity A_chi, giving

        D <= (g^2/2) exp(|g| Bmax) A_chi.

    This is four times looser in the weak-field coefficient than the symmetric
    split but requires no knowledge of alpha within the convex domain.
    """
    A = float(integrated_b2_activity)
    if A < 0.0 or Bmax < 0.0:
        raise ValueError("activity and Bmax must be nonnegative")
    return response_activity_kl_ceiling(g, A, float(Bmax))


def symmetric_pved_weak_kl(g: float, integrated_b2_activity: float) -> float:
    """Leading O(g^2) path KL for symmetric rate splitting."""
    A = float(integrated_b2_activity)
    if A < 0.0:
        raise ValueError("activity must be nonnegative")
    return response_activity_weak_kl(g, 0.25 * A)


def required_integrated_nlambda_for_fidelity(
    g: float,
    q_star: float,
    *,
    Bmax: float = 1.0,
    symmetric_split: bool = True,
) -> float:
    """Necessary integrated N_chi*Lambda budget for target wrong probability.

    Uses A_chi = 2 int N_chi Lambda dt. The returned quantity has dimensions of
    a weighted number of chirality-changing events.
    """
    gg = abs(float(g))
    if gg == 0.0:
        return math.inf
    d = required_kl_for_wrong_probability(q_star)
    if symmetric_split:
        # C <= (g^2/8) exp(g Bmax/2) A_chi
        # A_chi = 2 I, hence C <= (g^2/4) exp(...) I.
        return float(4.0 * math.exp(-0.5 * gg * Bmax) * d / (gg * gg))
    # Convex alpha-independent ceiling:
    # C <= (g^2/2) exp(g Bmax) A_chi = g^2 exp(...) I.
    return float(math.exp(-gg * Bmax) * d / (gg * gg))


def kz_cell_path_information(
    g: float,
    N_xi: float,
    integrated_lambda: float,
) -> float:
    """Weak-field path KL of one KZ cell for symmetric PVED splitting.

    Because sum B^2 a_total = 2 N_xi Lambda, the leading path KL is

        D = g^2 N_xi / 4 * integral Lambda dt.
    """
    if N_xi < 0.0 or integrated_lambda < 0.0:
        raise ValueError("N_xi and integrated_lambda must be nonnegative")
    return float((float(g) ** 2) * float(N_xi) * float(integrated_lambda) / 4.0)


def kz_impulse_constant_activity(g: float, N_xi: float, Lambda: float, a_hat: float) -> float:
    """Weak-field KL over [-t_hat,+t_hat] with t_hat=1/a_hat and constant Lambda."""
    if a_hat <= 0.0:
        raise ValueError("a_hat must be positive")
    return kz_cell_path_information(g, N_xi, 2.0 * float(Lambda) / float(a_hat))


def gaussian_action_from_R(R: float) -> float:
    """Gaussian sign-selection action R^2/2."""
    return 0.5 * float(R) ** 2
