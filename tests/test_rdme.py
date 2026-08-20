import numpy as np
from pvchiral.rdme import (
    FrankRDMEParams, critical_k3p, k3p_from_a, chemostat_A,
    lambda_resp_racemic, kz_scales, predicted_Rmax, simulate_frank_quench,
)


def test_critical_mapping():
    p = FrankRDMEParams()
    kc = critical_k3p(p)
    assert np.isclose(k3p_from_a(0.0, p), kc)
    A = chemostat_A(kc, p)
    # racemic ds/dt=0 at s=s0
    l = d = p.s0/2
    ds = 2*(p.kp*A*l - p.km*l*l) - 2*(kc*l*d - p.Q)
    assert abs(ds) < 1e-12


def test_lambda_positive_and_R_linear():
    p = FrankRDMEParams()
    L = lambda_resp_racemic(-0.05, p)
    assert L > 0
    r1 = predicted_Rmax(0.01, 400.0, p)
    r2 = predicted_Rmax(0.02, 400.0, p)
    assert np.isclose(r2, 2*r1)


def test_small_quench_runs_and_preserves_symmetry_at_zero_bias():
    p = FrankRDMEParams(omega=40, D=0.5)
    out = simulate_frank_quench(0.0, 25.0, params=p, nx=16, nreal=24,
                                dt=0.02, a_i=-0.25, seed=123)
    w = out['snapshots']['incoming_freeze']['wrong_fraction']
    assert 0.25 < w < 0.75
    assert out['snapshots']['incoming_freeze']['mean_total_count'] > 0
