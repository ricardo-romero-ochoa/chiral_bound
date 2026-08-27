"""Order-parameter independence, and the inertness of B_j = 0 reactions."""
import numpy as np
import pytest
from pvchiral import Lambda, Theta_k, bound_T2, eps_eff
from pvchiral.network import oligomer_network, OLIG_CHI

G = 1e-6
DIMER_EE = np.array([0.0, 0.0, 1.0, 0.0, -1.0])


def test_polymerisation_steps_carry_no_bias():
    """Additive PVED => dimerisation conserves sum nu chi, so B_j = 0."""
    net = oligomer_network(g=G)
    zero = [r.B for i, r in enumerate(net.rxns) if i in (2, 3, 4, 8)]
    assert all(b == pytest.approx(0.0, abs=1e-12) for b in zero)


@pytest.mark.parametrize("mult", [1.0, 10.0, 100.0])
def test_lambda_invariant_under_polymerisation_rate(mult):
    """100x faster (de)polymerisation must not change Lambda."""
    ref = Lambda(oligomer_network(g=G))
    got = Lambda(oligomer_network(g=G, kd=5.0*mult, kdm=0.5*mult))
    assert got == pytest.approx(ref, rel=2e-3)


def test_bound_holds_for_misaligned_order_parameter():
    net = oligomer_network(g=G)
    N2 = float(net.conc[2] + net.conc[3] + net.conc[4])
    for e_k, N_k in ((OLIG_CHI, net.s), (DIMER_EE, N2)):
        assert abs(eps_eff(net, e_k, N_k, g=G)) <= bound_T2(net, e_k, N_k, g=G)


def test_misalignment_costs_selection_power():
    """The R-relevant combination eps/sqrt(Theta) is largest when P ~ B."""
    net = oligomer_network(g=G)
    N2 = float(net.conc[2] + net.conc[3] + net.conc[4])
    aligned = eps_eff(net, g=G)/np.sqrt(Theta_k(net))
    misaligned = eps_eff(net, DIMER_EE, N2, g=G)/np.sqrt(Theta_k(net, DIMER_EE, N2))
    common = 2*G*np.sqrt(net.s*Lambda(net))
    assert aligned > misaligned
    assert max(aligned, misaligned) <= common
