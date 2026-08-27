"""Theorem tests.

The previous version of this suite tested only the Frank network and only at
operating points reachable by integrating it, so every alpha_j was in [0,1] by
construction.  It therefore could not detect that T1 and T2 REQUIRE
alpha_j in [0,1] -- an assumption local detailed balance does not supply.
This version:

  * asserts the activity/frenetic IDENTITY for arbitrary alpha_j;
  * asserts the unconditional bound T2u for arbitrary alpha_j;
  * asserts that T1 and T2 FAIL outside [0,1], turning the assumption into a
    characterised boundary rather than a hidden premise;
  * keeps the 81-point Frank grid as the convex-domain check.
"""
import itertools
import numpy as np
import pytest
from pvchiral import (bound_T2, bound_unconditional, convex_barrier,
                      custom_network, eps_decomposition, eps_eff,
                      frank_network, Lambda, Theta_k)
from pvchiral.network import frank_chemostat_a

G = 1e-6
RNG = np.random.default_rng(20260731)


def _random_network(n_rx=None, alpha_lo=0.0, alpha_hi=1.0, equilibrium=False):
    n_sp = int(RNG.integers(2, 5))
    n_rx = int(RNG.integers(2, 7)) if n_rx is None else n_rx
    chi = RNG.integers(-2, 3, size=n_sp).astype(float)
    if not np.any(chi):
        chi[0] = 1.0
    nu = [RNG.integers(-2, 3, size=n_sp).astype(float) for _ in range(n_rx)]
    Jp = RNG.exponential(size=n_rx)
    Jm = Jp.copy() if equilibrium else RNG.exponential(size=n_rx)
    al = RNG.uniform(alpha_lo, alpha_hi, size=n_rx)
    units = np.abs(chi) + 1.0
    return custom_network(chi, units, nu, Jp, Jm, al,
                          conc=RNG.uniform(0.5, 2.0, size=n_sp))


# --------------------------------------------------------------- the identity
@pytest.mark.parametrize("trial", range(200))
def test_activity_frenetic_identity_arbitrary_alpha(trial):
    """eps = (g/N)[ (1/2) sum P B a + sum P B (alpha-1/2) j ] is an identity."""
    net = _random_network(alpha_lo=-4.0, alpha_hi=5.0)
    d = eps_decomposition(net, g=G)
    assert d["total"] == pytest.approx(eps_eff(net, g=G), rel=1e-11, abs=1e-30)


@pytest.mark.parametrize("alpha_max", [1.0, 5.0, 50.0])
def test_frenetic_term_vanishes_at_equilibrium(alpha_max):
    """j_j = 0 at equilibrium, so the frenetic term vanishes for EVERY alpha.

    This is exactly why the old equilibrium test passed for all alpha and was
    blind to the convexity gap.  Asserted here so the blindness is documented.
    """
    for _ in range(20):
        net = _random_network(alpha_lo=-alpha_max, alpha_hi=alpha_max,
                              equilibrium=True)
        assert eps_decomposition(net, g=G)["frenetic_term"] == pytest.approx(0.0, abs=1e-30)


# --------------------------------------------------- unconditional bound (T2u)
@pytest.mark.parametrize("trial", range(200))
def test_unconditional_bound_arbitrary_alpha(trial):
    net = _random_network(alpha_lo=-4.0, alpha_hi=5.0)
    assert abs(eps_eff(net, g=G)) <= bound_unconditional(net, g=G)*(1 + 1e-10)


@pytest.mark.parametrize("trial", range(100))
def test_T2u_reduces_to_T2_on_convex_domain(trial):
    """For alpha in [0,1] the frenetic term cannot exceed the activity term."""
    net = _random_network(alpha_lo=0.0, alpha_hi=1.0)
    assert convex_barrier(net)
    assert bound_unconditional(net, g=G) <= bound_T2(net, g=G)*(1 + 1e-10)


# ------------------------------------ T1/T2 fail outside the convex domain
def test_T1_fails_outside_convex_domain():
    """alpha = 2, J+ = 1, J- = 100  =>  weight = 2 - 100 = -98, so the
    thermodynamically favoured enantiomer is the one that LOSES.
    (alpha = -1 on the same fluxes gives +199: it breaks T2, not T1.)"""
    net = custom_network(chi=[1.0, -1.0], units=[1.0, 1.0],
                         nu=[[1.0, 0.0]], Jp=[1.0], Jm=[100.0], alpha=[2.0])
    assert not convex_barrier(net)
    assert eps_eff(net, g=G) < 0.0


@pytest.mark.parametrize("alpha,Jp,Jm", [(5.0, 2.0, 1.0), (-1.0, 1.0, 100.0)])
def test_T2_fails_outside_convex_domain(alpha, Jp, Jm):
    """alpha = 5 on (2,1) gives weight 6 vs activity 3; alpha = -1 on (1,100)
    gives 199 vs 101.  Both exceed T2; both satisfy the unconditional T2u."""
    net = custom_network(chi=[1.0, -1.0], units=[1.0, 1.0],
                         nu=[[1.0, 0.0]], Jp=[Jp], Jm=[Jm], alpha=[alpha])
    assert not convex_barrier(net)
    assert abs(eps_eff(net, g=G)) > bound_T2(net, g=G)
    assert abs(eps_eff(net, g=G)) <= bound_unconditional(net, g=G)*(1 + 1e-10)


@pytest.mark.parametrize("trial", range(300))
def test_search_for_T2_violation_inside_convex_domain(trial):
    """Adversarial search: T2 must never fail when alpha in [0,1]."""
    net = _random_network(alpha_lo=0.0, alpha_hi=1.0)
    assert abs(eps_eff(net, g=G)) <= bound_T2(net, g=G)*(1 + 1e-10)


# ------------------------------------------------ Frank grid (convex domain)
GRID = list(itertools.product((0.0, 0.5, 1.0), (1.5, 2.0, 3.0),
                              (0.02, 0.04, 0.08), (0.1, 0.4, 0.8)))


def _physical_states():
    for alpha, k3p, s_t, Qf in GRID:
        Q = Qf*((1.0 + k3p)*s_t/2)*s_t/2
        a = frank_chemostat_a(s_t, Q, k3p)
        if a <= 0:
            continue
        net = frank_network(a, Q, k3p, alpha, g=G)
        if net.s < 1e-8:
            continue
        yield alpha, k3p, Q, net


ALL = list(_physical_states())


def test_grid_is_populated():
    assert len(ALL) == 81


@pytest.mark.parametrize("case", ALL, ids=lambda c: f"a{c[0]}_k{c[1]}")
def test_frank_T1_and_T2(case):
    _, _, _, net = case
    assert convex_barrier(net)
    assert eps_eff(net, g=G) >= 0.0
    assert abs(eps_eff(net, g=G)) <= bound_T2(net, g=G)*(1 + 1e-12)


def test_amplification_approaches_asymptote():
    alpha, s_t, km = 1.0, 0.04, 1.0
    for k3p, tol in ((1.1, 0.05), (1.01, 0.02), (1.001, 0.02)):
        Q = ((k3p - km)*s_t/2 - 2*1e-4)*s_t/2
        a = frank_chemostat_a(s_t, Q, k3p)
        net = frank_network(a, Q, k3p, alpha, g=G)
        b = (k3p - km)*net.s/2
        assert eps_eff(net, g=G)/b/G == pytest.approx(
            (1 + alpha)*km/(k3p - km), rel=tol)
