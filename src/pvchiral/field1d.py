"""Stochastic quenches of the biased scalar Model-A equation in one dimension.

The dimensionless equation is

    d eta/dt = D nabla^2 eta + a(t) eta - eta^3 + eps + sqrt(2 theta) zeta,
    a(t) = t/tau_Q.

The routines return realization-level observables so uncertainty is estimated
across independent fields rather than by treating correlated lattice sites as
independent samples.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def quench_observables(
    eps: float,
    tau_Q: float,
    theta: float,
    D: float = 1.0,
    dx: float = 2.0,
    dt: float = 0.25,
    nx: int = 256,
    nreal: int = 32,
    a_i: float = -0.20,
    a_meas: float = 0.03,
    seed: int = 0,
) -> dict:
    """Simulate a quench and return realization-level sign statistics."""
    if min(tau_Q, D, dx, dt, nx, nreal) <= 0 or theta < 0:
        raise ValueError("positive dynamical parameters and theta >= 0 are required")
    if D * dt / dx**2 > 0.5:
        raise ValueError("explicit diffusion step is unstable: require D dt/dx^2 <= 1/2")
    rng = np.random.default_rng(seed)
    eta = np.zeros((int(nreal), int(nx)), dtype=float)
    time = a_i * tau_Q
    t_end = a_meas * tau_Q
    amp = np.sqrt(2.0 * theta * dt / dx)
    pre = D * dt / dx**2
    n_steps = int(np.ceil((t_end - time) / dt))
    for _ in range(n_steps):
        lap = np.roll(eta, 1, 1) + np.roll(eta, -1, 1) - 2.0 * eta
        eta += pre * lap + dt * ((time / tau_Q) * eta - eta**3 + eps)
        if theta > 0:
            eta += amp * rng.standard_normal(eta.shape)
        time += dt
    positive = eta > 0
    per_realization = positive.mean(axis=1)
    walls = (positive != np.roll(positive, -1, axis=1)).mean(axis=1)
    return {
        "f_L": float(per_realization.mean()),
        "f_L_sem": float(per_realization.std(ddof=1) / np.sqrt(nreal)) if nreal > 1 else 0.0,
        "wall_density": float(walls.mean()),
        "wall_density_sem": float(walls.std(ddof=1) / np.sqrt(nreal)) if nreal > 1 else 0.0,
        "per_realization_f_L": per_realization,
        "per_realization_wall_density": walls,
        "nreal": int(nreal),
        "nx": int(nx),
        "n_steps": n_steps,
    }


def quench_fL(*args, **kwargs) -> float:
    """Backward-compatible wrapper returning only the favored-site fraction."""
    return quench_observables(*args, **kwargs)["f_L"]


def xi_hat_over_dx(tau_Q: float, D: float = 1.0, dx: float = 2.0) -> float:
    """Lattice resolution of the mean-field freeze-out correlation length."""
    return float(np.sqrt(D * np.sqrt(tau_Q)) / dx)


def _thresholds(tau_Q, eps, f_L, level: float):
    tau_Q, eps, f_L = map(np.asarray, (tau_Q, eps, f_L))
    xs, ys, raw_tau = [], [], []
    for tq in sorted(set(tau_Q.tolist())):
        idx = np.where(tau_Q == tq)[0]
        order = np.argsort(eps[idx])
        ee = eps[idx][order]
        ff = f_L[idx][order]
        # Sampling noise may weakly violate monotonicity.  The cumulative
        # maximum is a transparent monotone envelope, not a fitted model.
        ff_mono = np.maximum.accumulate(ff)
        if ff_mono[0] <= level <= ff_mono[-1] and np.unique(ff_mono).size > 1:
            loge = np.interp(level, ff_mono, np.log10(ee))
            raw_tau.append(float(tq))
            xs.append(np.log10(tq))
            ys.append(loge)
    return np.asarray(raw_tau), np.asarray(xs), np.asarray(ys)


def exponent_from_scan(
    tau_Q,
    eps,
    f_L,
    level: float = 0.85,
    min_xi_over_dx: float = 3.5,
    D: float = 1.0,
    dx: float = 2.0,
) -> dict:
    """Fit ``d log(eps*)/d log(tau_Q)`` at fixed local selection probability."""
    raw_tau, xs, ys = _thresholds(tau_Q, eps, f_L, level)
    resolved = np.array([xi_hat_over_dx(tq, D, dx) >= min_xi_over_dx for tq in raw_tau])
    if resolved.sum() < 3:
        raise ValueError("at least three resolved threshold crossings are required")
    xr, yr = xs[resolved], ys[resolved]
    slope, intercept = np.polyfit(xr, yr, 1)
    residuals = yr - (slope * xr + intercept)
    n = len(xr)
    dof = n - 2
    sse = float(np.sum(residuals**2))
    slope_se = float(np.sqrt((sse / dof) / np.sum((xr - xr.mean()) ** 2))) if dof > 0 else np.inf
    tcrit = float(student_t.ppf(0.975, dof)) if dof > 0 else np.inf
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": slope_se,
        "slope_ci95": (float(slope - tcrit * slope_se), float(slope + tcrit * slope_se)),
        "rms_resid_dex": float(np.sqrt(np.mean(residuals**2))),
        "n_used": int(n),
        "tau_Q_threshold": raw_tau,
        "log_tauQ": xs,
        "log_eps_star": ys,
        "resolved": resolved,
    }


def bootstrap_exponent_from_scan(
    tau_Q,
    eps,
    f_L,
    f_L_sem,
    *,
    level: float = 0.85,
    min_xi_over_dx: float = 3.5,
    D: float = 1.0,
    dx: float = 2.0,
    n_boot: int = 2000,
    seed: int = 20260731,
) -> dict:
    """Parametric bootstrap of the exponent using realization-level SEMs.

    This propagates the measured uncertainty of each scan point.  It is not a
    substitute for lattice-size or time-step convergence, which are reported
    separately.
    """
    rng = np.random.default_rng(seed)
    tau_Q = np.asarray(tau_Q, float)
    eps = np.asarray(eps, float)
    mean = np.asarray(f_L, float)
    sem = np.asarray(f_L_sem, float)
    slopes = []
    for _ in range(int(n_boot)):
        draw = np.clip(rng.normal(mean, sem), 0.0, 1.0)
        try:
            fit = exponent_from_scan(
                tau_Q,
                eps,
                draw,
                level=level,
                min_xi_over_dx=min_xi_over_dx,
                D=D,
                dx=dx,
            )
        except ValueError:
            continue
        slopes.append(fit["slope"])
    if len(slopes) < max(100, n_boot // 5):
        raise RuntimeError("too few bootstrap fits retained")
    slopes = np.asarray(slopes)
    return {
        "slope_median": float(np.median(slopes)),
        "slope_ci95": tuple(np.quantile(slopes, [0.025, 0.975]).tolist()),
        "n_retained": int(len(slopes)),
        "slopes": slopes,
    }
