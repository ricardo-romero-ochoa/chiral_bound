"""Spatial stochastic Frank-network quenches (RDME tau-leap validation).

This module is intentionally separate from the phenomenological Model-A
solvers.  It simulates the underlying reversible chemical reaction network on
periodic lattice voxels with integer molecule counts and stochastic diffusion.
The implementation uses operator-split Poisson/binomial tau leaping; convergence
with the time step is tested in the Milestone-2 audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class FrankRDMEParams:
    omega: float = 100.0       # system-size / voxel volume in concentration units
    s0: float = 1.0            # racemic total chiral concentration
    kp: float = 1.0
    km: float = 1.0
    kr: float = 1.0e-2
    Q: float = 5.0e-3          # reverse L+D <- P flux density
    D: float = 1.0             # lattice diffusion coefficient (dx=1)
    alpha: float = 0.5


def critical_k3p(p: FrankRDMEParams) -> float:
    """k3p at which the racemic Frank state has zero chiral eigenvalue."""
    # a_eff = (k3p-km)s/2 - 2 kr - 2Q/s
    return float(p.km + 2.0 * (2.0 * p.kr + 2.0 * p.Q / p.s0) / p.s0)


def k3p_from_a(a_eff: float, p: FrankRDMEParams) -> float:
    return float(p.km + 2.0 * (a_eff + 2.0 * p.kr + 2.0 * p.Q / p.s0) / p.s0)


def chemostat_A(k3p: float, p: FrankRDMEParams) -> float:
    """A chemostat that keeps s=s0 on the g=0 racemic deterministic branch."""
    return float((p.km*p.s0/2.0 + k3p*p.s0/2.0 - 2.0*p.Q/p.s0) / p.kp)


def lambda_resp_racemic(a_eff: float, p: FrankRDMEParams) -> float:
    """Paired response activity at the g=0 racemic state for alpha=1/2.

    For symmetric splitting Q_j=B_j, so Lambda_resp=Lambda exactly.  Fluxes
    are density fluxes and Lambda_resp has units 1/time.
    """
    if abs(p.alpha - 0.5) > 1e-12:
        raise ValueError("closed-form lambda_resp_racemic is for alpha=1/2")
    k3 = k3p_from_a(a_eff, p)
    A = chemostat_A(k3, p)
    l = d = p.s0/2.0
    # R1/R2: B=+/-1
    a1 = p.kp*A*l + p.km*l*l
    a2 = a1
    # R4 L<->D: B=-2
    a4 = p.kr*l + p.kr*d
    # R3 has B=0 and drops out.
    return float((a1 + a2 + 4.0*a4) / (2.0*p.s0))


def theta_racemic(a_eff: float, p: FrankRDMEParams) -> float:
    """Continuum noise strength for eta=(l-d)/s0 at the racemic branch.

    This is the density-field theta (no voxel-volume factor).  Averaging over a
    domain of volume V introduces the expected 1/V variance reduction.
    """
    Lam = lambda_resp_racemic(a_eff, p)
    # aligned coordinate at alpha=1/2: Theta = Lambda/s0
    return float(Lam / p.s0)


def kz_scales(tau_Q: float, p: FrankRDMEParams) -> dict:
    if tau_Q <= 0:
        raise ValueError("tau_Q must be positive")
    ah = tau_Q ** -0.5
    xi = np.sqrt(p.D / ah)
    that = np.sqrt(tau_Q)
    return {"a_hat": float(ah), "xi_hat": float(xi), "t_hat": float(that)}


def predicted_Rmax(g: float, tau_Q: float, p: FrankRDMEParams, domain_length: float | None = None) -> float:
    """Milestone-1 critical-selection envelope for a 1D RDME domain."""
    sc = kz_scales(tau_Q, p)
    V = sc["xi_hat"] if domain_length is None else float(domain_length)
    Lam = lambda_resp_racemic(-sc["a_hat"], p)
    Nxi = p.omega * p.s0 * V
    return abs(float(g)) * float(np.sqrt(Lam * Nxi / sc["a_hat"]))


def _poisson_capped(rng, mean, cap):
    """Poisson draws capped by available reactants; returns draw and clipped count."""
    x = rng.poisson(np.maximum(mean, 0.0))
    y = np.minimum(x, cap)
    return y, int(np.count_nonzero(x > cap))


def _diffuse_periodic(n: np.ndarray, rate: float, dt: float, rng) -> np.ndarray:
    """Exact first-hop split over one operator-split diffusion substep."""
    if rate <= 0 or dt <= 0:
        return n
    p_leave = 1.0 - np.exp(-2.0 * rate * dt)
    movers = rng.binomial(n, p_leave)
    left = rng.binomial(movers, 0.5)
    right = movers - left
    return n - movers + np.roll(left, -1, axis=1) + np.roll(right, 1, axis=1)


def _reaction_step(L, Dn, *, a_eff, g, dt, p: FrankRDMEParams, rng):
    """One local reaction tau-leap; returns updated counts and cap diagnostics."""
    Om = p.omega
    k3 = k3p_from_a(a_eff, p)
    A = chemostat_A(k3, p)
    alpha = p.alpha

    # Bias-coupled rates.  B1=+1, B2=-1, B4=-2.
    k1p = p.kp*np.exp(alpha*g)
    k1m = p.km*np.exp(-(1.0-alpha)*g)
    k2p = p.kp*np.exp(-alpha*g)
    k2m = p.km*np.exp((1.0-alpha)*g)
    k4p = p.kr*np.exp(-g)   # L -> D
    k4m = p.kr*np.exp(+g)   # D -> L

    clips = 0
    # Birth channels do not consume dynamical reactants beyond catalysts.
    bL = rng.poisson(np.maximum(k1p*A*L*dt, 0.0))
    bD = rng.poisson(np.maximum(k2p*A*Dn*dt, 0.0))
    revP = rng.poisson(p.Q*Om*dt, size=L.shape)

    # Pair annihilation first; its expected leap is small at the chosen dt.
    ann_mean = k3 * L * Dn / Om * dt
    ann, c = _poisson_capped(rng, ann_mean, np.minimum(L, Dn)); clips += c
    L1 = L - ann
    D1 = Dn - ann

    # Reverse-autocatalytic loss, using n(n-1)/Omega.
    dL_mean = k1m * L1 * np.maximum(L1-1, 0) / Om * dt
    dL, c = _poisson_capped(rng, dL_mean, L1); clips += c
    L2 = L1 - dL
    dD_mean = k2m * D1 * np.maximum(D1-1, 0) / Om * dt
    dD, c = _poisson_capped(rng, dD_mean, D1); clips += c
    D2 = D1 - dD

    # Racemization is sampled as binomial first-order conversion and therefore
    # cannot overdraw the remaining population.
    pLD = 1.0 - np.exp(-k4p*dt)
    pDL = 1.0 - np.exp(-k4m*dt)
    rLD = rng.binomial(L2, pLD)
    rDL = rng.binomial(D2, pDL)

    Lnew = L2 - rLD + rDL + bL + revP
    Dnew = D2 - rDL + rLD + bD + revP
    return Lnew.astype(np.int64), Dnew.astype(np.int64), clips


def _block_means(eta: np.ndarray, block: int) -> np.ndarray:
    """Nonoverlapping periodic block means, discarding at most block-1 sites."""
    nreal, nx = eta.shape
    nb = nx // block
    if nb < 1:
        raise ValueError("block larger than lattice")
    x = eta[:, :nb*block].reshape(nreal, nb, block)
    return x.mean(axis=2)


def simulate_frank_quench(
    g: float,
    tau_Q: float,
    *,
    params: FrankRDMEParams = FrankRDMEParams(),
    nx: int = 128,
    nreal: int = 128,
    dt: float = 0.01,
    a_i: float = -0.20,
    a_end: float | None = None,
    seed: int = 0,
    record_freeze: bool = True,
    initialize_steady: bool = True,
    burn_time: float = 0.0,
    switch_off_bias_at_incoming_freeze: bool = False,
) -> dict:
    """Simulate a linear quench a_eff=t/tau_Q through the Frank bifurcation.

    The initial state is a Poisson sample of the g=0 racemic branch.  The run
    records block-averaged sign statistics at the incoming freeze-out point
    a=-a_hat and, optionally, at +a_hat.  Blocks have length round(xi_hat).
    """
    if min(tau_Q, nx, nreal, dt, params.omega, params.s0) <= 0:
        raise ValueError("positive simulation parameters required")
    sc = kz_scales(tau_Q, params)
    ah = sc["a_hat"]
    if a_end is None:
        a_end = ah
    if a_i >= -ah:
        raise ValueError("a_i must start before incoming freeze-out")
    if a_end < -ah:
        raise ValueError("a_end must reach incoming freeze-out")

    rng = np.random.default_rng(seed)
    if initialize_steady:
        yss = steady_state_conc(a_i, g, params)
        L = rng.poisson(params.omega*yss[0], size=(int(nreal), int(nx))).astype(np.int64)
        Dn = rng.poisson(params.omega*yss[1], size=(int(nreal), int(nx))).astype(np.int64)
    else:
        mean_half = params.omega*params.s0/2.0
        L = rng.poisson(mean_half, size=(int(nreal), int(nx))).astype(np.int64)
        Dn = rng.poisson(mean_half, size=(int(nreal), int(nx))).astype(np.int64)

    # Optional fixed-control burn-in restores the spatial stationary noise
    # around the deterministic initialization before the quench begins.
    clips = 0
    burn_steps = 0
    if burn_time > 0:
        bt = 0.0
        while bt < burn_time - 0.5*dt:
            hburn = min(dt, burn_time-bt)
            L = _diffuse_periodic(L, params.D, 0.5*hburn, rng)
            Dn = _diffuse_periodic(Dn, params.D, 0.5*hburn, rng)
            L, Dn, c = _reaction_step(L, Dn, a_eff=a_i, g=g, dt=hburn, p=params, rng=rng)
            clips += c
            L = _diffuse_periodic(L, params.D, 0.5*hburn, rng)
            Dn = _diffuse_periodic(Dn, params.D, 0.5*hburn, rng)
            bt += hburn
            burn_steps += 1

    t = a_i*tau_Q
    t_end = a_end*tau_Q
    t_freeze = -sc["t_hat"]
    t_post = sc["t_hat"]
    block = max(1, int(np.rint(sc["xi_hat"])))
    rec = {}
    steps = 0

    def snapshot(label, a_now):
        total = L + Dn
        eta = (L - Dn) / (params.omega*params.s0)
        bm = _block_means(eta, block)
        per_real_wrong = (bm < 0).mean(axis=1) if g >= 0 else (bm > 0).mean(axis=1)
        # For g=0, define wrong as negative to preserve a symmetry diagnostic.
        wrong = float(per_real_wrong.mean())
        sem = float(per_real_wrong.std(ddof=1)/np.sqrt(nreal)) if nreal > 1 else 0.0
        z = float(norm.ppf(np.clip(1.0-wrong, 1e-12, 1-1e-12)))
        flat = bm.ravel()
        bmean = float(flat.mean())
        bstd = float(flat.std(ddof=1)) if flat.size > 1 else 0.0
        Rmom = bmean/bstd if bstd > 0 else np.inf
        if bstd > 0:
            zz = (flat-bmean)/bstd
            skew = float(np.mean(zz**3))
            excess = float(np.mean(zz**4)-3.0)
        else:
            skew = excess = 0.0
        rec[label] = {
            "a": float(a_now),
            "wrong_fraction": wrong,
            "wrong_sem": sem,
            "probit_R": z,
            "eta_mean": float(eta.mean()),
            "eta_var_site": float(eta.var()),
            "mean_total_count": float(total.mean()),
            "block_mean": bmean,
            "block_std": bstd,
            "moment_R": float(Rmom),
            "block_skew": skew,
            "block_excess_kurtosis": excess,
            "corr_length": float(correlation_length_fft(eta)),
            "block": int(block),
            "nblocks_per_real": int(nx//block),
            "per_realization_wrong": per_real_wrong,
        }

    freeze_done = False
    post_done = False
    while t < t_end - 0.5*dt:
        h = min(dt, t_end-t)
        tmid = t + 0.5*h
        a_now = tmid/tau_Q
        # Strang-ish split: half diffusion, reactions, half diffusion.
        L = _diffuse_periodic(L, params.D, 0.5*h, rng)
        Dn = _diffuse_periodic(Dn, params.D, 0.5*h, rng)
        g_now = 0.0 if (switch_off_bias_at_incoming_freeze and tmid >= t_freeze) else g
        L, Dn, c = _reaction_step(L, Dn, a_eff=a_now, g=g_now, dt=h, p=params, rng=rng)
        clips += c
        L = _diffuse_periodic(L, params.D, 0.5*h, rng)
        Dn = _diffuse_periodic(Dn, params.D, 0.5*h, rng)
        t += h
        steps += 1
        if record_freeze and (not freeze_done) and t >= t_freeze:
            snapshot("incoming_freeze", -ah)
            freeze_done = True
        if (not post_done) and t >= t_post and t_end >= t_post:
            snapshot("outgoing_freeze", +ah)
            post_done = True

    if record_freeze and not freeze_done:
        snapshot("incoming_freeze", -ah)
    if (not post_done) and t_end >= t_post - 1e-12:
        snapshot("outgoing_freeze", +ah)
        post_done = True
    snapshot("final", t/tau_Q)

    Lam = lambda_resp_racemic(-ah, params)
    Rmax_cont = predicted_Rmax(g, tau_Q, params)
    Rmax_block = predicted_Rmax(g, tau_Q, params, domain_length=block)
    return {
        "g": float(g), "tau_Q": float(tau_Q), "nx": int(nx), "nreal": int(nreal),
        "dt": float(dt), "a_i": float(a_i), "a_end": float(a_end), "seed": int(seed),
        "a_hat": ah, "xi_hat": sc["xi_hat"], "t_hat": sc["t_hat"], "block": block,
        "Lambda_resp_freeze": Lam, "Rmax_continuum": Rmax_cont, "Rmax_block": Rmax_block,
        "clip_events": int(clips), "steps": int(steps), "burn_steps": int(burn_steps),
        "burn_time": float(burn_time),
        "switch_off_bias_at_incoming_freeze": bool(switch_off_bias_at_incoming_freeze),
        "snapshots": rec,
    }


def correlation_length_fft(eta: np.ndarray, dx: float = 1.0, max_fraction: float = 0.25) -> float:
    """Second-moment-like integral correlation length from mean-subtracted fields.

    Returns the positive-lag integral of the normalized autocorrelation until its
    first zero crossing (or max_fraction*nx).  Intended as a diagnostic, not a
    precision critical-exponent estimator.
    """
    x = np.asarray(eta, float)
    if x.ndim == 1:
        x = x[None, :]
    x = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, axis=1)
    ac = np.fft.irfft(f*np.conj(f), n=x.shape[1], axis=1).real / x.shape[1]
    ac = ac.mean(axis=0)
    if ac[0] <= 0:
        return 0.0
    c = ac/ac[0]
    m = max(2, int(max_fraction*x.shape[1]))
    stop = m
    neg = np.where(c[1:m] <= 0)[0]
    if len(neg):
        stop = int(neg[0]+1)
    return float(dx*(0.5 + c[1:stop].sum()))


def local_rhs_conc(y, a_eff: float, g: float, p: FrankRDMEParams) -> np.ndarray:
    """Deterministic well-mixed concentration drift for the RDME reaction set."""
    l, d = np.asarray(y, float)
    k3 = k3p_from_a(a_eff, p)
    A = chemostat_A(k3, p)
    alpha = p.alpha
    k1p = p.kp*np.exp(alpha*g); k1m = p.km*np.exp(-(1-alpha)*g)
    k2p = p.kp*np.exp(-alpha*g); k2m = p.km*np.exp((1-alpha)*g)
    k4p = p.kr*np.exp(-g); k4m = p.kr*np.exp(g)
    j1 = k1p*A*l - k1m*l*l
    j2 = k2p*A*d - k2m*d*d
    j3 = k3*l*d - p.Q
    j4 = k4p*l - k4m*d
    return np.array([j1-j3-j4, j2-j3+j4])


def steady_state_conc(a_eff: float, g: float, p: FrankRDMEParams) -> np.ndarray:
    """Stable local steady state on the symmetric branch (used for initialization)."""
    from scipy.optimize import root
    # Linear-response starting guess around s0.
    Lam = lambda_resp_racemic(a_eff, p) if abs(p.alpha-0.5)<1e-12 else 0.5
    r = max(1e-6, -a_eff)
    eta0 = np.clip(g*Lam/r, -0.5, 0.5)
    y0 = np.array([0.5*p.s0*(1+eta0), 0.5*p.s0*(1-eta0)])
    sol = root(lambda y: local_rhs_conc(y, a_eff, g, p), y0)
    if not sol.success or np.any(sol.x <= 0):
        raise RuntimeError(f"failed to initialize steady state: {sol.message}")
    return np.asarray(sol.x, float)


def _pved_pair_propensities(L, Dn, *, a_eff: float, g: float, p: FrankRDMEParams):
    """Total affected directed propensities per realization for R1, R2, R4.

    Returns (plus, minus, B), where plus/minus have shape (nreal, 3) and B is
    [1,-1,-2].  Pair 3 (mutual inhibition) has B=0 and is omitted.
    """
    Om = p.omega
    k3 = k3p_from_a(a_eff, p)
    A = chemostat_A(k3, p)
    alpha = p.alpha
    B = np.array([1.0, -1.0, -2.0])
    SL = L.sum(axis=1).astype(float)
    SD = Dn.sum(axis=1).astype(float)
    LL = (L * np.maximum(L - 1, 0)).sum(axis=1).astype(float) / Om
    DD = (Dn * np.maximum(Dn - 1, 0)).sum(axis=1).astype(float) / Om
    plus = np.column_stack([
        p.kp * np.exp(alpha*g) * A * SL,
        p.kp * np.exp(-alpha*g) * A * SD,
        p.kr * np.exp(-g) * SL,
    ])
    minus = np.column_stack([
        p.km * np.exp(-(1.0-alpha)*g) * LL,
        p.km * np.exp((1.0-alpha)*g) * DD,
        p.kr * np.exp(+g) * SD,
    ])
    return plus, minus, B


def _pved_path_rates_per_real(L, Dn, *, a_eff: float, g: float, p: FrankRDMEParams, direction: str):
    """Instantaneous CTMC path-KL rate and B^2 activity for each realization.

    This diagnostic is exact for the underlying CTMC.  In the tau-leap solver
    it is integrated along the approximate trajectory; dt convergence is tested
    separately in Milestone 3.
    """
    if abs(p.alpha - 0.5) > 1e-12:
        raise ValueError("Milestone-3 closed path-activity relation requires alpha=1/2")
    if direction == "h||0":
        fp, fm, B = _pved_pair_propensities(L, Dn, a_eff=a_eff, g=g, p=p)
        u = 0.5 * g * B
        phi_p = u - 1.0 + np.exp(-u)
        phi_m = -u - 1.0 + np.exp(u)
    elif direction == "0||h":
        fp, fm, B = _pved_pair_propensities(L, Dn, a_eff=a_eff, g=0.0, p=p)
        u = 0.5 * g * B
        phi_p = -u - 1.0 + np.exp(u)
        phi_m = u - 1.0 + np.exp(-u)
    else:
        raise ValueError("direction must be 'h||0' or '0||h'")
    kl = np.sum(fp * phi_p[None, :] + fm * phi_m[None, :], axis=1)
    b2act = np.sum((B[None, :] ** 2) * (fp + fm), axis=1)
    return kl, b2act


def _advance_rdme_branch(L, Dn, *, a_eff: float, g: float, dt: float, p: FrankRDMEParams, rng):
    """Advance one operator-split RDME branch by a single step."""
    L = _diffuse_periodic(L, p.D, 0.5*dt, rng)
    Dn = _diffuse_periodic(Dn, p.D, 0.5*dt, rng)
    L, Dn, clips = _reaction_step(L, Dn, a_eff=a_eff, g=g, dt=dt, p=p, rng=rng)
    L = _diffuse_periodic(L, p.D, 0.5*dt, rng)
    Dn = _diffuse_periodic(Dn, p.D, 0.5*dt, rng)
    return L, Dn, clips


def _randomized_positive_event(L, Dn, rng):
    """Binary event that the global chiral inventory is positive; ties are fair coins."""
    m = (L - Dn).sum(axis=1)
    out = (m > 0).astype(np.int8)
    ties = np.where(m == 0)[0]
    if len(ties):
        out[ties] = rng.integers(0, 2, size=len(ties), dtype=np.int8)
    return out


def simulate_postfreeze_branches(
    g: float,
    tau_Q: float,
    *,
    params: FrankRDMEParams = FrankRDMEParams(),
    nx: int = 64,
    nreal: int = 256,
    dt: float = 0.01,
    a_i: float = -0.20,
    burn_time: float = 10.0,
    seed: int = 0,
    pre_g: float | None = None,
) -> dict:
    """Branch a quench at incoming freeze-out into bias-on and bias-off paths.

    The two post-freeze ensembles have *the same incoming-state distribution*.
    The on branch evolves with field g, while the off branch evolves with g=0.
    Along the on branch we integrate the CTMC path relative-entropy rate
    D(P_g||P_0); along the off branch we integrate D(P_0||P_g).  This directly
    tests the data-processing theorem for the final global-sign event.
    """
    if abs(params.alpha - 0.5) > 1e-12:
        raise ValueError("simulate_postfreeze_branches currently requires alpha=1/2")
    if min(tau_Q, nx, nreal, dt, params.omega, params.s0) <= 0:
        raise ValueError("positive simulation parameters required")
    sc = kz_scales(tau_Q, params)
    ah = sc["a_hat"]
    t_freeze = -sc["t_hat"]
    t_post = sc["t_hat"]
    if a_i*tau_Q >= t_freeze:
        raise ValueError("a_i must start before incoming freeze-out")

    rng_pre = np.random.default_rng(seed)
    g_pre = float(g if pre_g is None else pre_g)
    yss = steady_state_conc(a_i, g_pre, params)
    L = rng_pre.poisson(params.omega*yss[0], size=(int(nreal), int(nx))).astype(np.int64)
    Dn = rng_pre.poisson(params.omega*yss[1], size=(int(nreal), int(nx))).astype(np.int64)
    clips_pre = 0

    # Fixed-control burn-in at the starting point.
    bt = 0.0
    while bt < burn_time - 0.5*dt:
        h = min(dt, burn_time-bt)
        L, Dn, c = _advance_rdme_branch(L, Dn, a_eff=a_i, g=g_pre, dt=h, p=params, rng=rng_pre)
        clips_pre += c
        bt += h

    # Common biased history to incoming freeze-out.
    t = a_i*tau_Q
    while t < t_freeze - 0.5*dt:
        h = min(dt, t_freeze-t)
        a_now = (t + 0.5*h)/tau_Q
        L, Dn, c = _advance_rdme_branch(L, Dn, a_eff=a_now, g=g_pre, dt=h, p=params, rng=rng_pre)
        clips_pre += c
        t += h

    incoming_event_rng = np.random.default_rng(seed + 1777)
    event_in = _randomized_positive_event(L, Dn, incoming_event_rng)
    incoming_eta = (L-Dn).sum(axis=1) / (params.omega*params.s0*nx)

    # Branch independently from identical incoming states/distribution.
    L_on, D_on = L.copy(), Dn.copy()
    L_off, D_off = L.copy(), Dn.copy()
    rng_on = np.random.default_rng(seed + 1000003)
    rng_off = np.random.default_rng(seed + 2000003)
    kl_h0 = np.zeros(nreal, float)
    kl_0h = np.zeros(nreal, float)
    act_h = np.zeros(nreal, float)
    act_0 = np.zeros(nreal, float)
    clips_on = clips_off = 0

    t = t_freeze
    while t < t_post - 0.5*dt:
        hstep = min(dt, t_post-t)
        a_now = (t + 0.5*hstep)/tau_Q

        # Evaluate the CTMC information rates on the pre-reaction state after
        # the first diffusion half-step, matching the reaction operator timing.
        L_on = _diffuse_periodic(L_on, params.D, 0.5*hstep, rng_on)
        D_on = _diffuse_periodic(D_on, params.D, 0.5*hstep, rng_on)
        kr, ar = _pved_path_rates_per_real(L_on, D_on, a_eff=a_now, g=g, p=params, direction="h||0")
        kl_h0 += kr*hstep
        act_h += ar*hstep
        L_on, D_on, c = _reaction_step(L_on, D_on, a_eff=a_now, g=g, dt=hstep, p=params, rng=rng_on)
        clips_on += c
        L_on = _diffuse_periodic(L_on, params.D, 0.5*hstep, rng_on)
        D_on = _diffuse_periodic(D_on, params.D, 0.5*hstep, rng_on)

        L_off = _diffuse_periodic(L_off, params.D, 0.5*hstep, rng_off)
        D_off = _diffuse_periodic(D_off, params.D, 0.5*hstep, rng_off)
        kr, ar = _pved_path_rates_per_real(L_off, D_off, a_eff=a_now, g=g, p=params, direction="0||h")
        kl_0h += kr*hstep
        act_0 += ar*hstep
        L_off, D_off, c = _reaction_step(L_off, D_off, a_eff=a_now, g=0.0, dt=hstep, p=params, rng=rng_off)
        clips_off += c
        L_off = _diffuse_periodic(L_off, params.D, 0.5*hstep, rng_off)
        D_off = _diffuse_periodic(D_off, params.D, 0.5*hstep, rng_off)
        t += hstep

    event_rng_on = np.random.default_rng(seed + 3000017)
    event_rng_off = np.random.default_rng(seed + 4000037)
    event_on = _randomized_positive_event(L_on, D_on, event_rng_on)
    event_off = _randomized_positive_event(L_off, D_off, event_rng_off)
    eta_on = (L_on-D_on).sum(axis=1) / (params.omega*params.s0*nx)
    eta_off = (L_off-D_off).sum(axis=1) / (params.omega*params.s0*nx)

    block = max(1, int(np.rint(sc["xi_hat"])))
    Rmax_block = predicted_Rmax(g, tau_Q, params, domain_length=block)
    return {
        "g": float(g), "tau_Q": float(tau_Q), "nx": int(nx), "nreal": int(nreal),
        "dt": float(dt), "a_i": float(a_i), "burn_time": float(burn_time), "seed": int(seed), "pre_g": g_pre,
        "a_hat": float(ah), "t_hat": float(sc["t_hat"]), "xi_hat": float(sc["xi_hat"]),
        "block": int(block), "Rmax_block": float(Rmax_block),
        "event_in": event_in, "event_on": event_on, "event_off": event_off,
        "incoming_eta": incoming_eta, "eta_on": eta_on, "eta_off": eta_off,
        "kl_h0": kl_h0, "kl_0h": kl_0h,
        "b2_activity_h": act_h, "b2_activity_0": act_0,
        "clip_events_pre": int(clips_pre), "clip_events_on": int(clips_on), "clip_events_off": int(clips_off),
    }
