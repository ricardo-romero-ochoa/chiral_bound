"""Critical-mode projection for a simple pitchfork zero mode.

The chemical flux bound applies directly to the field projected on the left
critical eigenvector.  Stable-mode displacements induced at O(g) do not alter
that field because the left zero mode annihilates the Jacobian.
"""
from __future__ import annotations

import numpy as np


def critical_modes(jacobian: np.ndarray, atol: float = 1e-9):
    """Return normalized right/left eigenvectors of the eigenvalue nearest zero.

    The vectors ``v`` and ``u`` are normalized so that ``u @ v = 1``.  A
    ``ValueError`` is raised unless the selected eigenvalue is isolated and
    sufficiently close to zero.
    """
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("jacobian must be square")

    vals_r, vecs_r = np.linalg.eig(J)
    idx = int(np.argmin(np.abs(vals_r)))
    lam = vals_r[idx]
    if abs(lam) > atol:
        raise ValueError(f"no zero mode found: closest eigenvalue is {lam}")
    gaps = np.sort(np.abs(vals_r))
    if len(gaps) > 1 and gaps[1] <= 10 * atol:
        raise ValueError("zero mode is not simple")

    v = np.real_if_close(vecs_r[:, idx]).astype(float)
    vals_l, vecs_l = np.linalg.eig(J.T)
    idx_l = int(np.argmin(np.abs(vals_l - lam)))
    u = np.real_if_close(vecs_l[:, idx_l]).astype(float)
    overlap = float(u @ v)
    if abs(overlap) < 1e-12:
        raise ValueError("left and right critical modes have zero overlap")
    u = u / overlap
    v = v / np.linalg.norm(v)
    # Renormalize once more because v was rescaled.
    u = u / float(u @ v)
    return v, u


def critical_field(jacobian: np.ndarray, dF_dg: np.ndarray, *, left_mode=None) -> float:
    """Return the O(g) field coefficient in the scalar critical normal form."""
    J = np.asarray(jacobian, dtype=float)
    q = np.asarray(dF_dg, dtype=float)
    if left_mode is None:
        _, u = critical_modes(J)
    else:
        u = np.asarray(left_mode, dtype=float)
    return float(u @ q)


def projected_field_with_stable_shift(
    jacobian: np.ndarray,
    dF_dg: np.ndarray,
    stable_shift: np.ndarray,
    *,
    left_mode=None,
) -> float:
    """Project ``dF_dg + J h_g`` on the critical left mode.

    On a center manifold, ``stable_shift`` represents ``partial_g h``.  The
    result must equal :func:`critical_field` because ``u^T J = 0``.
    """
    J = np.asarray(jacobian, dtype=float)
    q = np.asarray(dF_dg, dtype=float)
    h = np.asarray(stable_shift, dtype=float)
    if left_mode is None:
        _, u = critical_modes(J)
    else:
        u = np.asarray(left_mode, dtype=float)
    return float(u @ (q + J @ h))
