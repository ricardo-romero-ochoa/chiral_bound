"""Small two-dimensional validation of biased Model-A quenches."""
from __future__ import annotations

import numpy as np


def quench_observables_2d(
    eps: float,
    tau_Q: float,
    theta: float,
    D: float = 1.0,
    dx: float = 2.0,
    dt: float = 0.25,
    nx: int = 48,
    nreal: int = 12,
    a_i: float = -0.20,
    a_meas: float = 0.03,
    seed: int = 0,
) -> dict:
    """Simulate periodic 2D scalar Model-A fields and return sign statistics."""
    if min(tau_Q, D, dx, dt, nx, nreal) <= 0 or theta < 0:
        raise ValueError("positive dynamical parameters and theta >= 0 are required")
    if D * dt / dx**2 > 0.25:
        raise ValueError("explicit 2D diffusion step requires D dt/dx^2 <= 1/4")
    rng = np.random.default_rng(seed)
    eta = np.zeros((int(nreal), int(nx), int(nx)), dtype=float)
    time = a_i * tau_Q
    t_end = a_meas * tau_Q
    amp = np.sqrt(2.0 * theta * dt / dx**2)
    pre = D * dt / dx**2
    n_steps = int(np.ceil((t_end - time) / dt))
    for _ in range(n_steps):
        lap = (
            np.roll(eta, 1, 1)
            + np.roll(eta, -1, 1)
            + np.roll(eta, 1, 2)
            + np.roll(eta, -1, 2)
            - 4.0 * eta
        )
        eta += pre * lap + dt * ((time / tau_Q) * eta - eta**3 + eps)
        if theta > 0:
            eta += amp * rng.standard_normal(eta.shape)
        time += dt
    positive = eta > 0
    per_realization = positive.mean(axis=(1, 2))
    walls_x = positive != np.roll(positive, -1, axis=1)
    walls_y = positive != np.roll(positive, -1, axis=2)
    walls = 0.5 * (walls_x.mean(axis=(1, 2)) + walls_y.mean(axis=(1, 2)))
    return {
        "f_L": float(per_realization.mean()),
        "f_L_sem": float(per_realization.std(ddof=1) / np.sqrt(nreal)) if nreal > 1 else 0.0,
        "wall_density": float(walls.mean()),
        "wall_density_sem": float(walls.std(ddof=1) / np.sqrt(nreal)) if nreal > 1 else 0.0,
        "nreal": int(nreal),
        "nx": int(nx),
        "n_steps": n_steps,
    }
