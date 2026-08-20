"""Mean-field validity and the corrected closed-form Ginzburg scale."""
import numpy as np
import pytest
from pvchiral import a_ginzburg, gi_closed_form

YR = 3.156e7
X, KM1, RHO, D = 1e-3, 0.1, 6.022e23, 1e-9
B = KM1 * X
K_RAC = np.log(2) / (1e5 * YR)
N_D = RHO * (D / B) ** 1.5
THETA = X / RHO


def test_closed_form_matches_direct_without_vacuous_absolute_tolerance():
    direct = a_ginzburg(THETA, B, D, d=3) / B
    closed = gi_closed_form(KM1, N_D)
    assert direct / closed == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_previous_factor_of_four_is_absent():
    correct = gi_closed_form(KM1, N_D)
    old = (2.0 / (KM1 * N_D)) ** 2
    assert old / correct == pytest.approx(4.0, rel=1e-15, abs=0.0)


def test_upper_critical_dimension():
    base = THETA * B / D**1.5
    assert base < 1.0
    assert a_ginzburg(THETA, B, D, d=3.9) < a_ginzburg(THETA, B, D, d=3.0)


@pytest.mark.parametrize(
    "ah,floor",
    [(2 * K_RAC, 1e20), (np.sqrt(2 * B / 3.16e7), 1e26)],
)
def test_system_is_far_outside_the_critical_region(ah, floor):
    assert ah / a_ginzburg(THETA, B, D, d=3) > floor


def test_finetuning_rescue_does_not_reach_criticality():
    km1 = 8e-10
    b2 = km1 * X
    nd2 = RHO * (D / b2) ** 1.5
    assert gi_closed_form(km1, nd2) < gi_closed_form(KM1, N_D)
