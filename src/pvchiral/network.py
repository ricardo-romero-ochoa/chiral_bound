"""Thermodynamically consistent chiral reaction networks.

Parity violation enters as a species free-energy shift mu_i -> mu_i - g kT chi_i.
Local detailed balance then FIXES d ln(k+/k-)/dg = B_j = sum_i nu_ij chi_i but
leaves the forward/reverse split free (Bronsted parameter alpha_j).  The
convex-barrier domain alpha_j in [0,1] is an additional kinetic assumption:

    k_j+ = kbar_j+ exp(+alpha_j B_j g),   k_j- = kbar_j- exp(-(1-alpha_j) B_j g)

Rate constants are therefore constructed from (kbar, alpha, B) rather than
assigned independently, so no test can pass by violating thermodynamics.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Rxn:
    """One elementary reversible reaction at a given state."""
    nu: np.ndarray      # stoichiometry over dynamical species
    B: float            # chirality change = nu . chi  (bias coupling)
    alpha: float        # Bronsted split of the bias
    Jp: float           # forward one-way flux
    Jm: float           # reverse one-way flux


@dataclass
class Network:
    rxns: list
    chi: np.ndarray     # chirality number per species
    units: np.ndarray   # chiral units per species (for the unit density s)
    conc: np.ndarray    # steady-state concentrations
    metadata: dict = field(default_factory=dict)

    @property
    def s(self):
        return float(self.units @ self.conc)


def _split(kbar_p, kbar_m, B, alpha, g):
    """Rate constants obeying local detailed balance for chirality change B."""
    return kbar_p*np.exp(alpha*B*g), kbar_m*np.exp(-(1.0 - alpha)*B*g)


class SteadyStateError(RuntimeError):
    """Raised when a numerical network does not reach a valid steady state."""


def _integrate_to_steady(rhs, y0, tmax, *, rtol=1e-11, atol=1e-18, drift_tol=1e-8):
    sol = solve_ivp(rhs, (0.0, tmax), y0, method="LSODA", rtol=rtol, atol=atol)
    if not sol.success:
        raise SteadyStateError(sol.message)
    y_raw = np.asarray(sol.y[:, -1], dtype=float)
    if not np.all(np.isfinite(y_raw)):
        raise SteadyStateError("non-finite steady-state concentrations")
    scale = max(1.0, float(np.max(np.abs(y_raw))))
    if float(np.min(y_raw)) < -1e-10 * scale:
        raise SteadyStateError(f"negative steady-state concentration: {y_raw}")
    y = np.maximum(y_raw, 0.0)
    drift = np.asarray(rhs(sol.t[-1], y), dtype=float)
    drift_norm = float(np.linalg.norm(drift, ord=np.inf))
    if not np.isfinite(drift_norm) or drift_norm > drift_tol * scale:
        raise SteadyStateError(
            f"steady-state drift {drift_norm:.3e} exceeds tolerance {drift_tol*scale:.3e}"
        )
    meta = {
        "solver_success": True,
        "solver_message": str(sol.message),
        "t_final": float(sol.t[-1]),
        "nfev": int(sol.nfev),
        "drift_inf_norm": drift_norm,
    }
    return y, meta


# --------------------------------------------------------------- Frank network
# species [L, D];  A and P chemostatted
#   R1  A + L <-> 2L        R3  L + D <-> P
#   R2  A + D <-> 2D        R4  L <-> D
FRANK_CHI = np.array([1.0, -1.0])
FRANK_NU = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            np.array([-1.0, -1.0]), np.array([-1.0, 1.0])]


def _frank_fluxes(y, a, Q, k3p, alpha, g, kp, km, kr):
    l, d = y
    B = [float(nu @ FRANK_CHI) for nu in FRANK_NU]
    k1p, k1m = _split(kp, km, B[0], alpha, g)
    k2p, k2m = _split(kp, km, B[1], alpha, g)
    k4p, k4m = _split(kr, kr, B[3], 0.5, g)
    return [Rxn(FRANK_NU[0], B[0], alpha, k1p*a*l, k1m*l*l),
            Rxn(FRANK_NU[1], B[1], alpha, k2p*a*d, k2m*d*d),
            Rxn(FRANK_NU[2], B[2], 0.5, k3p*l*d, Q),
            Rxn(FRANK_NU[3], B[3], 0.5, k4p*l, k4m*d)]


def frank_network(a, Q, k3p, alpha, g=1e-6, kp=1.0, km=1.0, kr=1e-4,
                  y0=(0.05, 0.05), tmax=1e9):
    """Integrate the Frank network to steady state and return it."""
    def rhs(t, y):
        out = np.zeros(2)
        for r in _frank_fluxes(np.maximum(y, 0.0), a, Q, k3p, alpha, g, kp, km, kr):
            out += r.nu*(r.Jp - r.Jm)
        return out

    y, meta = _integrate_to_steady(rhs, y0, tmax, rtol=1e-12, atol=1e-20)
    meta.update({"model": "Frank", "a": a, "Q": Q, "k3p": k3p,
                 "alpha": alpha, "g": g})
    return Network(_frank_fluxes(y, a, Q, k3p, alpha, g, kp, km, kr),
                   FRANK_CHI, np.array([1.0, 1.0]), y, meta)


def frank_equilibrium(alpha, g=1e-6, k3p=2.0, l_eq=0.05, kp=1.0, km=1.0, kr=1e-4):
    """Chemostat values placing the Frank network at genuine equilibrium."""
    k4p, k4m = _split(kr, kr, -2.0, 0.5, g)
    d_eq = l_eq*k4p/k4m
    k1p, k1m = _split(kp, km, 1.0, alpha, g)
    return (k1m/k1p)*l_eq, k3p*l_eq*d_eq          # (a_eq, Q_eq)


def frank_chemostat_a(s, Q, k3p, kp=1.0, km=1.0):
    """Chemostat A giving total density s at the racemic steady state."""
    return (km*s/2 + k3p*s/2 - 2*Q/s)/kp


# ----------------------------------------------------- monomer + dimer network
# species [L, D, LL, LD, DD]
OLIG_CHI = np.array([1.0, -1.0, 2.0, 0.0, -2.0])
OLIG_UNITS = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
_ON = [np.array(v, float) for v in
       ([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [-2, 0, 1, 0, 0], [-1, -1, 0, 1, 0],
        [0, -2, 0, 0, 1], [-1, 1, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1],
        [0, 0, 0, -1, 0])]


def _olig_fluxes(y, alpha, g, A, W, kp, km, kd, kdm, kr, ke, kw, kwm):
    L, D, LL, LD, DD = y
    B = [float(nu @ OLIG_CHI) for nu in _ON]
    k1p, k1m = _split(kp, km, B[0], alpha, g)
    k2p, k2m = _split(kp, km, B[1], alpha, g)
    r5p, r5m = _split(kr, kr, B[5], 0.5, g)
    e6p, e6m = _split(ke, ke, B[6], 0.5, g)
    e7p, e7m = _split(ke, ke, B[7], 0.5, g)
    J = [(k1p*A*L, k1m*L*L), (k2p*A*D, k2m*D*D), (kd*L*L, kdm*LL),
         (2*kd*L*D, kdm*LD), (kd*D*D, kdm*DD), (r5p*L, r5m*D),
         (e6p*LL, e6m*LD), (e7p*LD, e7m*DD), (kw*LD, kwm*W)]
    al = [alpha, alpha, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    return [Rxn(_ON[i], B[i], al[i], J[i][0], J[i][1]) for i in range(9)]


def oligomer_network(alpha=0.7, g=1e-6, A=0.05, W=1e-4, kp=1.0, km=1.0,
                     kd=5.0, kdm=0.5, kr=1e-3, ke=1e-3, kw=0.4, kwm=1e-3,
                     tmax=1e8):
    def rhs(t, y):
        out = np.zeros(5)
        for r in _olig_fluxes(np.maximum(y, 0.0), alpha, g, A, W,
                              kp, km, kd, kdm, kr, ke, kw, kwm):
            out += r.nu*(r.Jp - r.Jm)
        return out

    y, meta = _integrate_to_steady(rhs, [0.02]*5, tmax, rtol=1e-12, atol=1e-20)
    meta.update({"model": "monomer-dimer", "alpha": alpha, "g": g,
                 "kd": kd, "kdm": kdm})
    return Network(_olig_fluxes(y, alpha, g, A, W, kp, km, kd, kdm, kr, ke, kw, kwm),
                   OLIG_CHI, OLIG_UNITS, y, meta)


# ------------------------------------------------- generic / test-only builders
def custom_network(chi, units, nu, Jp, Jm, alpha, conc=None):
    """Build a Network directly from arrays.

    For theorem tests the fluxes need not come from integrating a model; what
    matters is that B_j = nu_j . chi.  This lets the suite probe alpha_j
    outside [0,1], which no physically integrated network will produce on its
    own and which the equilibrium tests are structurally blind to.
    """
    chi = np.asarray(chi, float)
    nu = [np.asarray(v, float) for v in nu]
    alpha = np.broadcast_to(np.asarray(alpha, float), (len(nu),))
    rxns = [Rxn(nu[i], float(nu[i] @ chi), float(alpha[i]),
                float(Jp[i]), float(Jm[i])) for i in range(len(nu))]
    conc = np.ones_like(chi) if conc is None else np.asarray(conc, float)
    return Network(rxns, chi, np.asarray(units, float), conc, {"source": "algebraic"})


TWO_STATE_NU = [np.array([-1.0, 1.0])]


def two_state_network(n=1, g=1e-6, k=1.0, alpha=0.5):
    """L_n <-> D_n : two enantiomeric species of n chiral units each.

    Equilibrium here gives eta_eq = tanh(n g), NOT tanh(g): the normalisation
    of the order parameter is network-specific.  Included as a regression test
    against over-generalising the Frank result.
    """
    chi = np.array([float(n), -float(n)])
    B = float(TWO_STATE_NU[0] @ chi)          # = -2n
    kp, km = _split(k, k, B, alpha, g)
    x = 1.0/(1.0 + np.exp(B*g))               # [L_n] at equilibrium, total 1
    conc = np.array([x, 1.0 - x])
    rxns = [Rxn(TWO_STATE_NU[0], B, alpha, kp*conc[0], km*conc[1])]
    return Network(rxns, chi, np.array([float(n), float(n)]), conc,
                   {"source": "analytic-equilibrium", "n": int(n)})
