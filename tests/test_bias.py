"""General weak-bias channels: identity, equilibrium behavior, and bounds."""
import numpy as np
import pytest

from pvchiral.bias import (
    Lambda_A,
    bound_affinity,
    bound_barrier,
    current_dissipation_slack,
    decompose,
    eps_channels,
    eps_from_responses,
    entropy_production,
    ldb_response_coefficients,
)

RNG = np.random.default_rng(31415)


def _draw(n=None, equilibrium=False, scale=3.0):
    n = int(RNG.integers(2, 8)) if n is None else n
    P = RNG.normal(size=n)
    rp, rm = RNG.normal(size=n) * scale, RNG.normal(size=n) * scale
    Jp = RNG.exponential(size=n)
    Jm = Jp.copy() if equilibrium else RNG.exponential(size=n)
    return P, rp, rm, Jp, Jm


@pytest.mark.parametrize("trial", range(300))
def test_general_identity_arbitrary_response_coefficients(trial):
    """The affinity/barrier identity is exact for arbitrary derivative coefficients."""
    P, rp, rm, Jp, Jm = _draw()
    A, F = decompose(rp, rm)
    h = RNG.normal()
    direct = eps_from_responses(P, rp, rm, Jp, Jm, 1.0, h=h)
    assert eps_channels(P, A, F, Jp, Jm, 1.0, h=h)["total"] == pytest.approx(
        direct, rel=1e-11, abs=1e-30
    )


def test_ldb_response_coefficients_do_not_absorb_perturbation_amplitude():
    B = np.array([1.0, -2.0, 0.5])
    alpha = np.array([0.0, 0.5, 1.0])
    A, F = ldb_response_coefficients(B, alpha)
    assert np.allclose(A, B)
    assert np.allclose(F, B * (alpha - 0.5))


@pytest.mark.parametrize("trial", range(100))
def test_general_identity_reduces_to_species_energy_case(trial):
    """For h=g, A=B and F=B(alpha-1/2) reproduce the PVED identity."""
    g, n = 1e-6, 5
    P, B = RNG.normal(size=n), RNG.normal(size=n)
    alpha = RNG.uniform(-3, 4, size=n)
    Jp, Jm = RNG.exponential(size=n), RNG.exponential(size=n)
    A, F = ldb_response_coefficients(B, alpha)
    ref = g * float(np.sum(P * B * (alpha * Jp + (1 - alpha) * Jm)))
    got = eps_channels(P, A, F, Jp, Jm, 1.0, h=g)["total"]
    assert got == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("trial", range(100))
def test_barrier_channel_vanishes_at_equilibrium(trial):
    """A pure barrier perturbation cannot shift this instantaneous drift at detailed balance."""
    P, rp, rm, Jp, Jm = _draw(equilibrium=True, scale=10.0)
    A, F = decompose(rp, rm)
    ch = eps_channels(P, A, F, Jp, Jm, 1.0)
    assert ch["barrier"] == pytest.approx(0.0, abs=1e-30)
    assert ch["total"] == pytest.approx(ch["affinity"], rel=1e-12)


@pytest.mark.parametrize("trial", range(100))
def test_affinity_channel_can_survive_at_equilibrium(trial):
    P, rp, rm, Jp, Jm = _draw(equilibrium=True)
    A, _ = decompose(rp, rm)
    if np.allclose(A, 0.0):
        pytest.skip("degenerate draw")
    assert eps_channels(P, A, np.zeros_like(A), Jp, Jm, 1.0)["affinity"] != 0.0


@pytest.mark.parametrize("edge_affinity", [1e-8, 1e-4, 0.1, 1.0, 5.0, 30.0])
def test_current_dissipation_inequality(edge_affinity):
    """j^2 <= a sigma/2 for finite reversible one-way fluxes."""
    activity = 1.0
    Jp = activity / (1 + np.exp(-edge_affinity))
    Jm = activity - Jp
    assert current_dissipation_slack([Jp], [Jm])[0] >= -1e-15


@pytest.mark.parametrize("trial", range(200))
def test_barrier_channel_is_conditioned_by_dissipation(trial):
    """The barrier-sector bound holds for finite reversible channels."""
    P, rp, rm, Jp, Jm = _draw()
    _, F = decompose(rp, rm)
    N, h = 1.0, RNG.normal()
    Theta = float(np.sum(P**2 * (Jp + Jm))) / (2 * N**2)
    eb = eps_channels(P, np.zeros_like(F), F, Jp, Jm, N, h=h)["barrier"]
    assert abs(eb) <= bound_barrier(Theta, F, Jp, Jm, h=h) * (1 + 1e-9)


@pytest.mark.parametrize("trial", range(200))
def test_affinity_channel_is_activity_bounded(trial):
    P, rp, rm, Jp, Jm = _draw()
    A, _ = decompose(rp, rm)
    N = s = 1.0
    h = RNG.normal()
    Theta = float(np.sum(P**2 * (Jp + Jm))) / (2 * N**2)
    ea = eps_channels(P, A, np.zeros_like(A), Jp, Jm, N, h=h)["affinity"]
    assert abs(ea) <= bound_affinity(Theta, s, A, Jp, Jm, h=h) * (1 + 1e-9)


def test_Lambda_A_is_independent_of_perturbation_amplitude():
    A = np.array([1.0, -2.0])
    Jp, Jm = np.array([2.0, 3.0]), np.array([1.0, 4.0])
    expected = np.sum(A**2 * (Jp + Jm)) / 2
    assert Lambda_A(A, Jp, Jm, s=1.0) == pytest.approx(expected)


def test_entropy_production_nonnegative():
    Jp, Jm = RNG.exponential(size=50), RNG.exponential(size=50)
    assert np.all(entropy_production(Jp, Jm) >= -1e-15)


def test_strictly_irreversible_channel_makes_reversible_pair_dissipation_bound_trivial():
    assert np.isinf(entropy_production([1.0], [0.0])[0])


# Structural sanity check used only to illustrate why a biased adsorption step
# need not directly move the *total* chiral inventory.  This is not a generic
# model of CISS and carries no quantitative energetic interpretation.
SURFACE_CHI = np.array([0.0, 0.0, 1.0, -1.0])       # L_sol D_sol L_ads D_ads
TOTAL_EXCESS = np.array([1.0, -1.0, 1.0, -1.0])
ADSORB_L = np.array([-1.0, 0.0, 1.0, 0.0])
CREATE_L_ADS = np.array([0.0, 0.0, 1.0, 0.0])


def test_biased_adsorption_can_carry_surface_bias_but_not_change_total_inventory():
    assert float(ADSORB_L @ SURFACE_CHI) == pytest.approx(1.0)
    assert float(ADSORB_L @ TOTAL_EXCESS) == pytest.approx(0.0)


def test_downstream_inventory_changing_step_can_project_on_total_excess():
    assert float(CREATE_L_ADS @ TOTAL_EXCESS) == pytest.approx(1.0)
