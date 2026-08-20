"""Flux-sum identity, activity/frenetic decomposition, and the bounds.

CONVENTION.  Parity violation enters as mu_i -> mu_i - g kT chi_i with
chi_L = +1, chi_D = -1, so the FULL enantiomer splitting is

    E_D - E_L = 2 g kT      i.e.    g = Delta_E_PV / (2 kT).

`g` is therefore the HALF-splitting in units of kT.  Use g_from_splitting()
to convert.  (An earlier version of this work identified g with the full
splitting; every selection ratio computed that way is a factor of two high.)

THEOREM STRUCTURE.  Local detailed balance fixes only
d ln(k+/k-)/dg = B_j; the Bronsted parameter alpha_j is free.  Splitting the
weight about 1/2 gives an IDENTITY valid for arbitrary alpha_j,

    eps_eff = (g/N) [ (1/2) sum_j P_j B_j a_j
                      + sum_j P_j B_j (alpha_j - 1/2) j_j ]         (identity)

with activity a_j = J+ + J- and current j_j = J+ - J-.  The first term is
bounded by g sqrt(s Theta Lambda); the second is a frenetic term that
VANISHES IDENTICALLY AT EQUILIBRIUM for every alpha.  Hence

    |eps_eff| <= g sqrt(s Theta Lambda)
                 + (g/N) max_j|alpha_j - 1/2| sum_j |P_j B_j| |j_j|  (T2u)

which is unconditional.  Restricting to the convex-barrier domain
alpha_j in [0,1] makes the second term no larger than the first and recovers

    |eps_eff| <= 2 g sqrt(s Theta Lambda)                            (T2)

T2 and the positivity statement eps_eff >= 0 (T1) are FALSE outside
alpha_j in [0,1]; this is asserted as a characterised boundary in the tests,
not assumed away.
"""
import numpy as np


def g_from_splitting(delta_E_over_kT):
    """Convert a full enantiomer splitting Delta_E_PV/kT to the bias g."""
    return 0.5*delta_E_over_kT


def _proj(net, e_k, N_k):
    e_k = net.chi if e_k is None else np.asarray(e_k, float)
    N_k = net.s if N_k is None else float(N_k)
    P = np.array([float(r.nu @ e_k) for r in net.rxns])
    B = np.array([r.B for r in net.rxns])
    al = np.array([r.alpha for r in net.rxns])
    Jp = np.array([r.Jp for r in net.rxns])
    Jm = np.array([r.Jm for r in net.rxns])
    return P, B, al, Jp, Jm, N_k


def activity(net):
    """Dynamical activity a_j = J+ + J- of each reaction."""
    return np.array([r.Jp + r.Jm for r in net.rxns])


def current(net):
    """Net current j_j = J+ - J- of each reaction."""
    return np.array([r.Jp - r.Jm for r in net.rxns])


def eps_eff(net, e_k=None, N_k=None, g=1e-6):
    """O(g) bias-induced drift projected on ``eta_k``.

    For the critical normal form, ``e_k`` must be the left critical mode; see
    :mod:`pvchiral.center_manifold`.  The default aligned coordinate uses
    ``e_k=chi`` and ``N_k=s``.
    """
    P, B, al, Jp, Jm, N = _proj(net, e_k, N_k)
    return g*float(np.sum(P*B*(al*Jp + (1.0 - al)*Jm)))/N


def eps_decomposition(net, e_k=None, N_k=None, g=1e-6):
    """Activity/frenetic split of eps_eff.  Exact for arbitrary alpha_j."""
    P, B, al, Jp, Jm, N = _proj(net, e_k, N_k)
    a, j = Jp + Jm, Jp - Jm
    act = g*0.5*float(np.sum(P*B*a))/N
    fre = g*float(np.sum(P*B*(al - 0.5)*j))/N
    return dict(activity_term=act, frenetic_term=fre, total=act + fre)


def Theta_k(net, e_k=None, N_k=None):
    """Chemical-Langevin noise strength on eta_k."""
    P, B, al, Jp, Jm, N = _proj(net, e_k, N_k)
    return float(np.sum(P**2*(Jp + Jm)))/(2*N**2)


def Lambda(net):
    """Chirality-weighted one-way flux per chiral unit."""
    return sum(r.B**2*(r.Jp + r.Jm) for r in net.rxns)/(2*net.s)


def frenetic_norm(net, e_k=None, N_k=None):
    """(1/N) sum_j |P_j B_j| |j_j| -- the current-weighted response scale."""
    P, B, al, Jp, Jm, N = _proj(net, e_k, N_k)
    return float(np.sum(np.abs(P*B)*np.abs(Jp - Jm)))/N


def bound_T2(net, e_k=None, N_k=None, g=1e-6):
    """2 g sqrt(s Theta_k Lambda).  Valid only for alpha_j in [0,1]."""
    return 2*g*np.sqrt(net.s*Theta_k(net, e_k, N_k)*Lambda(net))


def bound_unconditional(net, e_k=None, N_k=None, g=1e-6):
    """T2u: holds for arbitrary alpha_j, thermodynamically admissible or not."""
    P, B, al, Jp, Jm, N = _proj(net, e_k, N_k)
    act = g*np.sqrt(net.s*Theta_k(net, e_k, N_k)*Lambda(net))
    fre = g*float(np.max(np.abs(al - 0.5)))*frenetic_norm(net, e_k, N_k)
    return act + fre


def convex_barrier(net, tol=1e-12):
    """True iff every alpha_j lies in [0,1] (the convex transition-state domain)."""
    return all(-tol <= r.alpha <= 1.0 + tol for r in net.rxns)


def frank_reduction(net, k3p, km=1.0, kr=1e-4, Q=None):
    """Closed-form coefficients of the reduced Frank dynamics.

    NOTE: lam_eq = X + Z is the relaxation rate of the chiral mode FOR THIS
    NETWORK ONLY.  Lambda == lam_eq is not a general identity -- see
    tests/test_reduction.py::test_Lambda_is_not_the_relaxation_rate_in_general.
    """
    s = net.s
    X, Z = km*s/2, 2*kr
    Y = 2*Q/s if Q is not None else 2*net.rxns[2].Jm/s
    b = (k3p - km)*s/2
    return dict(s=s, X=X, Y=Y, Z=Z, a_eff=b - Z - Y, b=b, lam_eq=X + Z)
