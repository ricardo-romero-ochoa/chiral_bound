"""Kibble-Zurek freeze-out, local selection, and global-domain statistics."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def a_hat(b: float, tau_c=np.inf, delta: float = 0.0, k_rac: float = 0.0) -> float:
    """Single-timescale freeze-out floor.

    This helper implements the original order-of-magnitude model

        a_hat = max[sqrt(2 b delta / tau_c), 2 k_rac].

    For colored forcing use :func:`pvchiral.environment.a_hat_spectral`.
    """
    if b <= 0 or delta < 0 or k_rac < 0:
        raise ValueError("b > 0, delta >= 0, and k_rac >= 0 are required")
    fl = np.sqrt(2.0 * b * delta / tau_c) if (delta > 0 and np.isfinite(tau_c)) else 0.0
    return float(max(fl, 2.0 * k_rac))


def tauQ_eff(b: float, tau_c=np.inf, delta: float = 0.0, k_rac: float = 0.0) -> float:
    """Effective dimensionless quench time, ``tau_Q~=(b/a_hat)^2``."""
    return float((b / a_hat(b, tau_c, delta, k_rac)) ** 2)


def selection_ratio(g, Lam, b, a_hat_, rho, D, d=3):
    """Upper bound on the Gaussian local selection signal-to-noise ratio."""
    del b  # retained for API compatibility and dimensional bookkeeping
    if min(g, Lam, a_hat_, rho, D) < 0 or a_hat_ == 0 or D == 0:
        raise ValueError("parameters must be nonnegative, with a_hat and D positive")
    xi = np.sqrt(D / a_hat_)
    return float(2.0 * g * np.sqrt(Lam / a_hat_) * np.sqrt(rho * xi**d))


def selection_ratio_tauQ(g, Lam, b, a_hat_, rho, D, d=3):
    """Equivalent quench-time representation; the dependence on ``b`` cancels."""
    if min(g, Lam, b, a_hat_, rho, D) < 0 or min(b, a_hat_, D) == 0:
        raise ValueError("parameters must be nonnegative, with b, a_hat, D positive")
    ell = np.sqrt(D / b)
    N_D = rho * ell**d
    tq = (b / a_hat_) ** 2
    return float(2.0 * g * np.sqrt(Lam / b) * np.sqrt(N_D) * tq ** ((2 + d) / 8))


def local_sign_probability(R: float) -> float:
    """Gaussian probability that one correlation domain chooses the favored sign."""
    return float(norm.cdf(R))


def wrong_sign_probability(R: float) -> float:
    """Gaussian probability that one correlation domain chooses the disfavored sign."""
    return float(norm.sf(R))


def all_domains_favored_probability(R: float, n_domains: int) -> float:
    """Independent-domain upper-level estimate ``Phi(R)**n_domains``."""
    if n_domains < 1:
        raise ValueError("n_domains must be positive")
    logp = n_domains * norm.logcdf(R)
    return float(np.exp(logp))


def R_for_global_confidence(n_domains: int, confidence: float = 0.95) -> float:
    """R required for all independent domains to be favored with given confidence."""
    if n_domains < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n_domains >= 1 and 0 < confidence < 1 are required")
    p_local = np.exp(np.log(confidence) / n_domains)
    return float(norm.ppf(p_local))


def rescue_factors(R_current: float, target: float = 1.0, d: int = 3) -> dict:
    """Single-parameter multiplicative rescue factors at fixed other parameters.

    Since ``R proportional sqrt(rho) D^(d/4) g sqrt(Lambda)``, the returned
    factors apply to density, diffusivity, bias magnitude, and activity.
    """
    if R_current <= 0 or target <= 0 or d <= 0:
        raise ValueError("R_current, target, and d must be positive")
    q = target / R_current
    return {
        "rho": float(q**2),
        "D": float(q ** (4.0 / d)),
        "g": float(q),
        "Lambda": float(q**2),
    }


def general_selection_ratio_bound(h: float, Lambda_resp: float, a_hat_: float, N_xi: float) -> float:
    """Critical-domain selection bound for a generic weak rate perturbation.

    Combining the reaction-pair response theorem with Gaussian Model-A
    freeze-out gives

    ``|R| <= |h| sqrt(Lambda_resp * N_xi / a_hat)``.
    """
    if Lambda_resp < 0 or a_hat_ <= 0 or N_xi < 0:
        raise ValueError("Lambda_resp >= 0, a_hat > 0, and N_xi >= 0 are required")
    return abs(float(h)) * float(np.sqrt(Lambda_resp * N_xi / a_hat_))


def wrong_sign_probability_lower_bound(h: float, Lambda_resp: float, a_hat_: float, N_xi: float) -> float:
    """Incoming-freeze Gaussian proxy for the disfavored-sign probability.

    Milestone 2 showed that this is *not* a hard late-time bound: a sustained
    field acts throughout the impulse/postcritical interval and can drive the
    eventual wrong-domain fraction well below this instantaneous proxy.
    """
    Rmax = general_selection_ratio_bound(h, Lambda_resp, a_hat_, N_xi)
    return float(norm.sf(Rmax))


def all_domains_favored_probability_upper_bound(
    h: float,
    Lambda_resp: float,
    a_hat_: float,
    N_xi: float,
    n_domains: int,
) -> float:
    """Independent-domain incoming-freeze Gaussian fidelity proxy.

    This should not be interpreted as an upper bound on late-time global
    homochirality; see the Milestone-2 RDME audit.
    """
    if n_domains < 1:
        raise ValueError("n_domains must be positive")
    Rmax = general_selection_ratio_bound(h, Lambda_resp, a_hat_, N_xi)
    return float(np.exp(n_domains * norm.logcdf(Rmax)))


def response_activity_required_for_global_confidence(
    h: float,
    a_hat_: float,
    N_xi: float,
    n_domains: int,
    confidence: float = 0.95,
) -> float:
    """Activity required by the incoming-freeze Gaussian proxy.

    Milestone 2 invalidated interpreting this inversion as a necessary
    condition on *late-time* global fidelity. It remains useful only inside the
    instantaneous Gaussian freeze-out approximation.
    """
    if h == 0 or a_hat_ <= 0 or N_xi <= 0:
        raise ValueError("h must be nonzero and a_hat, N_xi must be positive")
    Rreq = R_for_global_confidence(n_domains, confidence)
    return float(a_hat_ * Rreq**2 / (float(h)**2 * N_xi))


def pved_activity_required_for_global_confidence(
    g: float,
    a_hat_: float,
    N_xi: float,
    n_domains: int,
    confidence: float = 0.95,
) -> float:
    """Original-PVED activity scale from the incoming-freeze Gaussian proxy.

    The algebraic relation follows from ``Lambda_resp <= 4 Lambda`` on the
    convex domain, but Milestone 2 showed that it is not a necessary condition
    for late-time selection because the bias keeps acting after incoming
    freeze-out.
    """
    if g == 0 or a_hat_ <= 0 or N_xi <= 0:
        raise ValueError("g must be nonzero and a_hat, N_xi must be positive")
    Rreq = R_for_global_confidence(n_domains, confidence)
    return float(a_hat_ * Rreq**2 / (4.0 * float(g)**2 * N_xi))
