"""Milestone 1: paired response activity and critical-selection fidelity bounds."""
from __future__ import annotations

import numpy as np
import pytest

from pvchiral import (
    Lambda,
    Lambda_response,
    R_for_global_confidence,
    Theta_k,
    all_domains_favored_probability_upper_bound,
    bound_total_response,
    decompose,
    directed_fisher_information_rate,
    effective_response_coefficient,
    eps_channels,
    eps_from_responses,
    frank_network,
    general_selection_ratio_bound,
    paired_fisher_slack,
    pved_activity_required_for_global_confidence,
    pved_response_activity,
    response_activity_required_for_global_confidence,
    wrong_sign_probability_lower_bound,
)
from pvchiral.center_manifold import projected_field_with_stable_shift
from pvchiral.network import frank_chemostat_a

RNG = np.random.default_rng(20260818)


def _draw(n=None):
    n = int(RNG.integers(2, 9)) if n is None else n
    P = RNG.normal(size=n)
    rp = RNG.normal(scale=3.0, size=n)
    rm = RNG.normal(scale=3.0, size=n)
    Jp = RNG.exponential(size=n)
    Jm = RNG.exponential(size=n)
    return P, rp, rm, Jp, Jm


@pytest.mark.parametrize("trial", range(300))
def test_complete_response_obeys_paired_activity_bound(trial):
    P, rp, rm, Jp, Jm = _draw()
    A, F = decompose(rp, rm)
    N = float(RNG.uniform(0.2, 4.0))
    s = float(RNG.uniform(0.2, 5.0))
    h = float(RNG.normal())
    theta = float(np.sum(P**2 * (Jp + Jm))) / (2.0 * N**2)
    eps = eps_from_responses(P, rp, rm, Jp, Jm, N, h=h)
    bound = bound_total_response(theta, s, A, F, Jp, Jm, h=h)
    assert abs(eps) <= bound * (1.0 + 1e-11) + 1e-30


def test_paired_activity_bound_is_sharp():
    """Choose P_j proportional to Q_j so the Cauchy step saturates exactly."""
    a = np.array([0.7, 1.3, 2.1, 0.4])
    rho = np.array([-0.8, -0.1, 0.5, 0.9])
    Jp, Jm = 0.5 * a * (1 + rho), 0.5 * a * (1 - rho)
    A = np.array([1.2, -0.4, 0.7, 2.0])
    F = np.array([-0.2, 1.1, -0.8, 0.3])
    Q = effective_response_coefficient(A, F, Jp, Jm)
    P = Q.copy()
    rp, rm = F + 0.5 * A, F - 0.5 * A
    N, s, h = 2.3, 1.7, -0.6
    theta = float(np.sum(P**2 * a)) / (2.0 * N**2)
    eps = eps_from_responses(P, rp, rm, Jp, Jm, N, h=h)
    bound = bound_total_response(theta, s, A, F, Jp, Jm, h=h)
    assert abs(eps) == pytest.approx(bound, rel=1e-13, abs=1e-14)


@pytest.mark.parametrize("trial", range(300))
def test_pair_contraction_is_tighter_than_directed_fisher_bound(trial):
    _, rp, rm, Jp, Jm = _draw()
    A, F = decompose(rp, rm)
    s = float(RNG.uniform(0.2, 5.0))
    lhs = s * Lambda_response(A, F, Jp, Jm, s)
    rhs = 2.0 * directed_fisher_information_rate(rp, rm, Jp, Jm)
    assert lhs <= rhs * (1.0 + 1e-12) + 1e-14
    assert paired_fisher_slack(rp, rm, Jp, Jm, s=s) >= -1e-12


def test_pair_contraction_saturates_for_pure_affinity_response():
    A = np.array([0.4, -1.2, 2.0])
    F = np.zeros_like(A)
    rp, rm = 0.5 * A, -0.5 * A
    Jp = np.array([1.0, 0.4, 3.0])
    Jm = np.array([0.7, 2.0, 0.2])
    s = 2.1
    assert s * Lambda_response(A, F, Jp, Jm, s) == pytest.approx(
        2 * directed_fisher_information_rate(rp, rm, Jp, Jm), rel=1e-13
    )


@pytest.mark.parametrize("trial", range(200))
def test_pved_convex_domain_contracts_to_original_activity(trial):
    n = int(RNG.integers(2, 10))
    B = RNG.integers(-3, 4, size=n).astype(float)
    alpha = RNG.uniform(0.0, 1.0, size=n)
    Jp, Jm = RNG.exponential(size=n), RNG.exponential(size=n)
    s = float(RNG.uniform(0.1, 4.0))
    Lresp = pved_response_activity(B, alpha, Jp, Jm, s)
    L = float(np.sum(B**2 * (Jp + Jm))) / (2 * s)
    assert Lresp <= 4.0 * L * (1 + 1e-12) + 1e-14


def test_pved_response_activity_equals_original_activity_at_detailed_balance():
    B = np.array([1.0, -2.0, 0.5, 3.0])
    alpha = np.array([-5.0, 0.0, 1.0, 7.0])  # alpha drops out when j=0
    Jp = np.array([0.2, 1.0, 3.0, 0.7])
    Jm = Jp.copy()
    s = 1.9
    Lresp = pved_response_activity(B, alpha, Jp, Jm, s)
    L = float(np.sum(B**2 * (Jp + Jm))) / (2 * s)
    assert Lresp == pytest.approx(L, rel=1e-13)


def test_pved_four_lambda_ceiling_can_fail_outside_convex_domain():
    B = np.array([1.0])
    alpha = np.array([5.0])
    Jp, Jm, s = np.array([2.0]), np.array([1.0]), 1.0
    Lresp = pved_response_activity(B, alpha, Jp, Jm, s)
    L = float(np.sum(B**2 * (Jp + Jm))) / (2 * s)
    assert Lresp > 4.0 * L


def test_frank_symmetric_partition_saturates_paired_bound_for_aligned_coordinate():
    """For alpha=1/2, Q_j=B_j=P_j for the aligned chiral coordinate."""
    alpha, k3p, s_t, Qf, g = 0.5, 2.0, 0.04, 0.4, 1e-6
    Qchemo = Qf * ((1.0 + k3p) * s_t / 2) * s_t / 2
    a = frank_chemostat_a(s_t, Qchemo, k3p)
    net = frank_network(a, Qchemo, k3p, alpha, g=g)
    B = np.array([r.B for r in net.rxns])
    al = np.array([r.alpha for r in net.rxns])
    Jp = np.array([r.Jp for r in net.rxns])
    Jm = np.array([r.Jm for r in net.rxns])
    Lresp = pved_response_activity(B, al, Jp, Jm, net.s)
    new_bound = g * np.sqrt(net.s * Theta_k(net) * Lresp)
    from pvchiral.reduction import eps_eff
    assert abs(eps_eff(net, g=g)) == pytest.approx(new_bound, rel=1e-11)
    # The old convex-domain theorem is a factor-two envelope here.
    assert Lresp == pytest.approx(Lambda(net), rel=1e-11)


def test_critical_projection_identity_survives_stable_mode_shift_for_generic_bias():
    # First species is the critical mode; second is stable.
    J = np.array([[0.0, 0.0], [0.0, -3.0]])
    u = np.array([1.0, 0.0])
    nu = np.array([[1.0, 2.0], [-1.0, 1.0], [2.0, -0.5]])
    P = nu @ u
    rp = np.array([0.7, -0.4, 1.2])
    rm = np.array([-0.2, 0.8, -0.1])
    Jp = np.array([1.5, 0.7, 2.2])
    Jm = np.array([0.4, 1.1, 0.6])
    h = 0.03
    microscopic_q = np.sum(
        nu * (rp * Jp - rm * Jm)[:, None], axis=0
    )
    expected = eps_from_responses(P, rp, rm, Jp, Jm, N=1.0, h=h)
    projected = h * projected_field_with_stable_shift(
        J, microscopic_q, np.array([2.0, -4.0]), left_mode=u
    )
    assert projected == pytest.approx(expected, rel=1e-13)


def test_general_KZ_fidelity_bounds_are_exactly_invertible_at_the_envelope():
    h, Lresp, ahat, Nxi, M = 2e-4, 0.8, 3e-3, 2e6, 1000
    Rmax = general_selection_ratio_bound(h, Lresp, ahat, Nxi)
    assert wrong_sign_probability_lower_bound(h, Lresp, ahat, Nxi) < 0.5
    upper = all_domains_favored_probability_upper_bound(h, Lresp, ahat, Nxi, M)
    assert 0.0 <= upper <= 1.0
    req = response_activity_required_for_global_confidence(h, ahat, Nxi, M, upper)
    assert req == pytest.approx(Lresp, rel=2e-11)
    assert Rmax >= 0


@pytest.mark.parametrize("M", [1, 10, 10**3, 10**5])
def test_required_response_activity_matches_required_R(M):
    h, ahat, Nxi, q = 1e-5, 2e-4, 8e8, 0.95
    req = response_activity_required_for_global_confidence(h, ahat, Nxi, M, q)
    Rreq = R_for_global_confidence(M, q)
    assert general_selection_ratio_bound(h, req, ahat, Nxi) == pytest.approx(Rreq)


@pytest.mark.parametrize("M", [1, 100, 10**5])
def test_pved_activity_requirement_has_factor_four_convex_envelope(M):
    g, ahat, Nxi, q = 5e-17, 1e-10, 1e28, 0.95
    req_pv = pved_activity_required_for_global_confidence(g, ahat, Nxi, M, q)
    req_resp = response_activity_required_for_global_confidence(g, ahat, Nxi, M, q)
    assert req_pv == pytest.approx(req_resp / 4.0, rel=1e-13)
