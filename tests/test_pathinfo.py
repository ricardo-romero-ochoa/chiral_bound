import math
import numpy as np
import pytest

from pvchiral.pathinfo import (
    bernoulli_kl,
    bernoulli_probability_interval,
    favored_probability_upper_from_symmetric_baseline,
    wrong_probability_lower_from_reverse_kl,
    symmetric_pved_kl_rate,
    symmetric_pved_activity_kl_ceiling,
    symmetric_pved_weak_kl,
    kz_cell_path_information,
    kz_impulse_constant_activity,
    gaussian_action_from_R,
)


def test_bernoulli_kl_basic():
    assert bernoulli_kl(0.5, 0.5) == 0.0
    assert bernoulli_kl(1.0, 0.5) == pytest.approx(math.log(2.0))
    assert bernoulli_kl(0.0, 0.5) == pytest.approx(math.log(2.0))


def test_bernoulli_interval_inverts_exactly():
    q = 0.37
    D = 0.08
    lo, hi = bernoulli_probability_interval(q, D)
    assert lo < q < hi
    assert bernoulli_kl(lo, q) == pytest.approx(D, rel=1e-10, abs=1e-12)
    assert bernoulli_kl(hi, q) == pytest.approx(D, rel=1e-10, abs=1e-12)


def test_symmetric_baseline_bound_hits_one_at_ln2():
    assert favored_probability_upper_from_symmetric_baseline(math.log(2.0)) == 1.0


def test_reverse_wrong_closed_form():
    for C in [0.0, 0.1, 1.0, 4.0]:
        q = wrong_probability_lower_from_reverse_kl(C)
        assert 0.0 <= q <= 0.5
        assert bernoulli_kl(0.5, 1.0-q) <= C + 1e-12
        if C > 0:
            assert bernoulli_kl(0.5, 1.0-q) == pytest.approx(C, rel=1e-10, abs=1e-12)


def test_exact_kl_rate_is_quadratic_to_leading_order():
    fp = np.array([3.0, 2.0, 0.7])
    fm = np.array([1.2, 4.0, 0.3])
    B = np.array([1.0, -1.0, 2.0])
    g = 1e-5
    exact = symmetric_pved_kl_rate(fp, fm, B, g, direction='h||0')
    weak = (g*g/8.0) * np.sum(B*B*(fp+fm))
    assert exact == pytest.approx(weak, rel=3e-5, abs=1e-15)


def test_activity_ceiling_bounds_exact_rate_both_directions_randomized():
    rng = np.random.default_rng(123)
    for _ in range(300):
        n = rng.integers(1, 20)
        B = rng.integers(-3, 4, n).astype(float)
        B[B == 0] = 1
        g = rng.uniform(-0.3, 0.3)
        # h-measure rates for forward direction.
        fp_h = rng.lognormal(size=n)
        fm_h = rng.lognormal(size=n)
        exact_h0 = symmetric_pved_kl_rate(fp_h, fm_h, B, g, direction='h||0')
        A_h = np.sum(B*B*(fp_h+fm_h))
        cap_h0 = symmetric_pved_activity_kl_ceiling(g, A_h, np.max(np.abs(B)))
        assert exact_h0 <= cap_h0 * (1 + 1e-12) + 1e-14

        fp_0 = rng.lognormal(size=n)
        fm_0 = rng.lognormal(size=n)
        exact_0h = symmetric_pved_kl_rate(fp_0, fm_0, B, g, direction='0||h')
        A_0 = np.sum(B*B*(fp_0+fm_0))
        cap_0h = symmetric_pved_activity_kl_ceiling(g, A_0, np.max(np.abs(B)))
        assert exact_0h <= cap_0h * (1 + 1e-12) + 1e-14


def test_kz_impulse_equals_gaussian_action_for_constant_activity():
    g = 0.012
    N = 350.0
    Lam = 0.73
    ah = 0.08
    R = abs(g) * math.sqrt(Lam*N/ah)
    D = kz_impulse_constant_activity(g, N, Lam, ah)
    assert D == pytest.approx(gaussian_action_from_R(R), rel=1e-14)


def test_weak_activity_formula():
    assert symmetric_pved_weak_kl(0.1, 80.0) == pytest.approx(0.1)
    assert kz_cell_path_information(0.1, 20.0, 2.0) == pytest.approx(0.1)

from pvchiral.pathinfo import (
    exponential_tilt_kl_rate,
    response_activity_kl_ceiling,
    response_activity_weak_kl,
    pved_directed_response_coefficients,
    convex_pved_activity_kl_ceiling,
    required_kl_for_wrong_probability,
    required_integrated_nlambda_for_fidelity,
)


def test_generic_exponential_tilt_ceiling_randomized():
    rng = np.random.default_rng(20260818)
    for _ in range(300):
        n = int(rng.integers(1, 30))
        rates = rng.lognormal(size=n)
        r = rng.normal(size=n)
        h = float(rng.uniform(-0.4, 0.4))
        for direction in ["h||0", "0||h"]:
            exact = exponential_tilt_kl_rate(rates, r, h, direction=direction)
            Ar = float(np.sum(r*r*rates))
            cap = response_activity_kl_ceiling(h, Ar, float(np.max(np.abs(r))))
            assert exact <= cap * (1.0 + 1e-12) + 1e-14


def test_generic_weak_kl_matches_quadratic_limit():
    rates = np.array([1.3, 4.2, 0.8])
    r = np.array([-0.7, 0.2, 1.1])
    h = 1e-6
    exact = exponential_tilt_kl_rate(rates, r, h, direction="0||h")
    weak = response_activity_weak_kl(h, np.sum(r*r*rates))
    assert exact == pytest.approx(weak, rel=2e-5, abs=1e-16)


def test_pved_directed_coefficients_and_convex_ceiling():
    B = np.array([1.0, -2.0, 0.5])
    alpha = np.array([0.0, 0.4, 1.0])
    rp, rm = pved_directed_response_coefficients(B, alpha)
    assert np.allclose(rp-rm, B)
    assert np.max(np.abs(np.concatenate([rp, rm]))) <= np.max(np.abs(B))
    Achi = 17.3
    g = 0.03
    cap_sym = symmetric_pved_activity_kl_ceiling(g, Achi, 2.0)
    cap_conv = convex_pved_activity_kl_ceiling(g, Achi, 2.0)
    assert cap_sym < cap_conv


def test_required_event_budget_inverts_final_fidelity_condition():
    g = 5e-17
    q = 0.05
    d = required_kl_for_wrong_probability(q)
    I = required_integrated_nlambda_for_fidelity(g, q, Bmax=1.0, symmetric_split=True)
    # In the weak-field limit C = g^2 I / 4 for symmetric splitting.
    assert (g*g/4.0)*I == pytest.approx(d, rel=1e-14)
    I_conv = required_integrated_nlambda_for_fidelity(g, q, Bmax=1.0, symmetric_split=False)
    assert I_conv == pytest.approx(I/4.0, rel=1e-14)
