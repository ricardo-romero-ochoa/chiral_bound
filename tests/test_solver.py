"""Numerical network builders expose validated steady-state metadata."""
import numpy as np
from pvchiral import frank_network, oligomer_network
from pvchiral.network import frank_chemostat_a


def test_frank_solver_metadata_and_nonnegative_state():
    s = 0.04; k3p = 2.0; Q = 0.001
    A = frank_chemostat_a(s, Q, k3p)
    net = frank_network(A, Q, k3p, 0.5)
    assert net.metadata["solver_success"]
    assert net.metadata["drift_inf_norm"] < 1e-8
    assert np.all(net.conc >= 0)


def test_oligomer_solver_metadata_and_nonnegative_state():
    net = oligomer_network()
    assert net.metadata["solver_success"]
    assert net.metadata["drift_inf_norm"] < 1e-8
    assert np.all(net.conc >= 0)
