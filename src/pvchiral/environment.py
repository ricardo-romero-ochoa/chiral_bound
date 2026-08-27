"""Spectral environmental forcing and chemical low-pass filtering."""
from __future__ import annotations

import numpy as np


def normalized_power_law_psd(
    omega: np.ndarray,
    beta: float,
    variance: float,
) -> np.ndarray:
    """One-sided finite-band PSD proportional to ``omega**(-beta)``.

    The returned PSD is normalized so that its numerical integral over the
    supplied positive-frequency grid equals ``variance``.
    """
    w = np.asarray(omega, dtype=float)
    if w.ndim != 1 or len(w) < 2 or np.any(w <= 0) or np.any(np.diff(w) <= 0):
        raise ValueError("omega must be a strictly increasing positive grid")
    if variance < 0:
        raise ValueError("variance must be nonnegative")
    shape = w ** (-float(beta))
    norm = float(np.trapezoid(shape, w))
    return variance * shape / norm


def low_pass_transfer_sq(omega: np.ndarray, tau_filter: float = 0.0) -> np.ndarray:
    """Squared gain of a first-order chemical/compartment low-pass filter."""
    if tau_filter < 0:
        raise ValueError("tau_filter must be nonnegative")
    w = np.asarray(omega, dtype=float)
    return 1.0 / (1.0 + (w * tau_filter) ** 2)


def spectral_slew(
    sigma_a: float,
    beta: float,
    omega_min: float,
    omega_max: float,
    *,
    tau_filter: float = 0.0,
    n_grid: int = 8192,
) -> dict:
    """Return RMS slew and its spectral diagnostics.

    ``sigma_a`` is the RMS amplitude of the control parameter ``a``.  The RMS
    derivative is

        v_rms = [int omega^2 S_a(omega) |H(omega)|^2 d omega]^(1/2),

    with dimensions of ``a/time``.  For mean-field Model-A freeze-out, the
    environmental distance from threshold is ``a_hat_env=sqrt(v_rms)``.
    """
    if sigma_a < 0 or omega_min <= 0 or omega_max <= omega_min:
        raise ValueError("require sigma_a >= 0 and 0 < omega_min < omega_max")
    if n_grid < 64:
        raise ValueError("n_grid must be at least 64")
    w = np.geomspace(omega_min, omega_max, int(n_grid))
    psd = normalized_power_law_psd(w, beta, sigma_a**2)
    gain2 = low_pass_transfer_sq(w, tau_filter)
    derivative_variance = float(np.trapezoid(w**2 * psd * gain2, w))
    v_rms = float(np.sqrt(derivative_variance))
    per_log = w**3 * psd * gain2
    return {
        "omega": w,
        "psd": psd,
        "gain2": gain2,
        "slew_per_log_frequency": per_log,
        "v_rms": v_rms,
        "a_hat_env": float(np.sqrt(v_rms)),
    }


def a_hat_spectral(
    sigma_a: float,
    beta: float,
    omega_min: float,
    omega_max: float,
    *,
    tau_filter: float = 0.0,
    k_rac: float = 0.0,
    n_grid: int = 8192,
) -> float:
    """Freeze-out floor from finite-band colored forcing and racemization."""
    env = spectral_slew(
        sigma_a,
        beta,
        omega_min,
        omega_max,
        tau_filter=tau_filter,
        n_grid=n_grid,
    )["a_hat_env"]
    return float(max(env, 2.0 * k_rac))


def fastest_decade_dominates(beta: float) -> bool:
    """Unfiltered power-law criterion based on omega^3 S(omega)."""
    return bool(beta < 3.0)
