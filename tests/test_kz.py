"""Freeze-out, local/global selection, and field-simulation checks."""
import csv
from pathlib import Path
import numpy as np
import pytest
from pvchiral import (
    R_for_global_confidence,
    a_hat,
    all_domains_favored_probability,
    local_sign_probability,
    rescue_factors,
    selection_ratio,
    selection_ratio_tauQ,
    tauQ_eff,
)
from pvchiral.field1d import bootstrap_exponent_from_scan, exponent_from_scan, quench_fL

YR = 3.156e7
K_RAC = np.log(2) / (1e5 * YR)
PARAMS = dict(g=5e-17, Lam=1e-3, b=1e-4, rho=6.022e23, D=1e-9, d=3)
DATA = Path(__file__).resolve().parents[1] / "data" / "kz_scan_1d.csv"


@pytest.mark.parametrize("ah", [1e-12, 1e-9, 1e-6, 1e-3])
def test_two_routes_to_R_agree(ah):
    p = dict(PARAMS); p.pop("d")
    r1 = selection_ratio(a_hat_=ah, d=3, **p)
    r2 = selection_ratio_tauQ(a_hat_=ah, d=3, **p)
    assert r1 == pytest.approx(r2, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("bfac", [0.1, 1.0, 10.0])
def test_R_independent_of_b(bfac):
    p = dict(PARAMS); p.pop("d"); p["b"] *= bfac
    ref = dict(p); ref["b"] = PARAMS["b"]
    assert selection_ratio_tauQ(a_hat_=1e-9, d=3, **p) == pytest.approx(
        selection_ratio_tauQ(a_hat_=1e-9, d=3, **ref), rel=1e-12, abs=0.0)


def test_a_hat_floors():
    b = 1e-4
    assert a_hat(b, delta=0.0, k_rac=K_RAC) == pytest.approx(2 * K_RAC)
    assert a_hat(b, tau_c=3.16e7, delta=1.0, k_rac=K_RAC) == pytest.approx(
        np.sqrt(2 * b / 3.16e7), rel=1e-12)
    assert tauQ_eff(b, delta=0.0, k_rac=K_RAC) == pytest.approx((b / (2 * K_RAC)) ** 2)


def test_corrected_half_splitting_numbers():
    p = dict(PARAMS); p.pop("d")
    R_rac = selection_ratio(a_hat_=2 * K_RAC, d=3, **p)
    R_ann = selection_ratio(a_hat_=np.sqrt(2 * PARAMS["b"] / YR), d=3, **p)
    assert R_rac == pytest.approx(1.2203e3, rel=1e-3)
    assert R_ann == pytest.approx(4.352e-6, rel=1e-3)


def test_rescue_scaling():
    factors = rescue_factors(4.351983597844203e-6, d=3)
    assert factors["rho"] == pytest.approx(5.2799e10, rel=1e-4)
    assert factors["D"] == pytest.approx(1.4074e7, rel=1e-4)


def test_R_one_is_not_deterministic():
    assert local_sign_probability(1.0) == pytest.approx(0.841344746, rel=1e-9)
    assert all_domains_favored_probability(1.0, 100) < 1e-7
    assert R_for_global_confidence(100000, 0.95) > 4.8


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/run_kz_scan.py first")
def test_measured_exponent_is_consistent_but_not_overclaimed():
    rows = list(csv.DictReader(DATA.open()))
    tq = [float(r["tau_Q"]) for r in rows]
    eps = [float(r["eps"]) for r in rows]
    f = [float(r["f_L"]) for r in rows]
    sem = [float(r["f_L_sem"]) for r in rows]
    fit = exponent_from_scan(tq, eps, f)
    boot = bootstrap_exponent_from_scan(tq, eps, f, sem, n_boot=500)
    assert fit["n_used"] == 5
    assert abs(fit["slope"] + 0.375) < 0.08
    assert boot["slope_ci95"][0] < -0.375 < boot["slope_ci95"][1]
    # The broad OLS interval is reported rather than using the residual scatter
    # as an uncertainty estimate.
    assert fit["slope_ci95"][0] < -0.25 < fit["slope_ci95"][1]


def test_quench_selects_when_bias_dominates():
    assert quench_fL(eps=2e-2, tau_Q=100.0, theta=1e-6, nx=64, nreal=4, seed=1) > 0.99
