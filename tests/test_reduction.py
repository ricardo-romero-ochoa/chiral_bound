"""Equilibrium identities, and which of them are network-specific.

An earlier formulation claimed eta_eq = tanh(g) and Lambda = lambda_eq as
general results.  Both are Frank-specific.  The L_n <-> D_n network is kept
here as a permanent regression test against that over-generalisation.
"""
import numpy as np
import pytest
from pvchiral import (eps_eff, frank_reduction, g_from_splitting, Lambda,
                      Theta_k, two_state_network)
from pvchiral.network import frank_equilibrium, frank_network

G = 1e-6


# ------------------------------------------------------- what IS general
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_eps_equals_g_Lambda_at_equilibrium_frank(alpha):
    a_eq, Q_eq = frank_equilibrium(alpha, g=G)
    net = frank_network(a_eq, Q_eq, 2.0, alpha, g=G)
    assert eps_eff(net, g=G) == pytest.approx(G*Lambda(net), rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_eps_equals_g_Lambda_at_equilibrium_two_state(n, alpha):
    """Survives the counterexample that breaks the normalisation claims."""
    net = two_state_network(n=n, g=G, alpha=alpha)
    assert eps_eff(net, g=G) == pytest.approx(G*Lambda(net), rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_Theta_equals_Lambda_over_s(n):
    """FDT/shot-noise identity, aligned coordinate. Holds across networks."""
    net = two_state_network(n=n, g=G)
    assert Theta_k(net) == pytest.approx(Lambda(net)/net.s, rel=1e-12)


def test_Theta_equals_Lambda_over_s_frank():
    a_eq, Q_eq = frank_equilibrium(0.5, g=G)
    net = frank_network(a_eq, Q_eq, 2.0, 0.5, g=G)
    assert Theta_k(net) == pytest.approx(Lambda(net)/net.s, rel=1e-12)


# ------------------------------------------------ what is NETWORK-SPECIFIC
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_frank_equilibrium_gives_tanh_g(alpha):
    a_eq, Q_eq = frank_equilibrium(alpha, g=G)
    net = frank_network(a_eq, Q_eq, 2.0, alpha, g=G)
    l, d = net.conc
    assert (l - d)/(l + d) == pytest.approx(np.tanh(G), rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_eta_eq_normalisation_is_network_specific(n):
    """REGRESSION: L_n <-> D_n gives tanh(n g), not tanh(g)."""
    net = two_state_network(n=n, g=G)
    LL, DD = net.conc
    eta = (n*LL - n*DD)/net.s
    assert eta == pytest.approx(np.tanh(n*G), rel=1e-9)
    if n > 1:
        assert eta != pytest.approx(np.tanh(G), rel=1e-3)


def test_Lambda_is_not_the_relaxation_rate_in_general():
    """REGRESSION: for L_2 <-> D_2, Lambda = 2 * lambda_relax."""
    k = 1.0
    net = two_state_network(n=2, g=G, k=k)
    lam_relax = 2*k                      # dm/dt = -2k m for m = 2([L2]-[D2])
    assert Lambda(net) == pytest.approx(2*lam_relax, rel=1e-6)


def test_Lambda_equals_relaxation_rate_for_frank():
    a_eq, Q_eq = frank_equilibrium(0.5, g=G)
    net = frank_network(a_eq, Q_eq, 2.0, 0.5, g=G)
    red = frank_reduction(net, k3p=2.0, Q=Q_eq)
    assert Lambda(net) == pytest.approx(red["lam_eq"], rel=1e-9)


# --------------------------------------------------------- g convention
def test_g_is_the_half_splitting():
    """mu_i -> mu_i - g kT chi_i with chi = +-1 gives E_D - E_L = 2 g kT."""
    got = g_from_splitting(1e-16)
    assert got / (0.5e-16) == pytest.approx(1.0, rel=1e-15, abs=0.0)
    net = two_state_network(n=1, g=G)
    LL, DD = net.conc
    assert np.log(DD/LL) == pytest.approx(-2*G, rel=1e-6)
